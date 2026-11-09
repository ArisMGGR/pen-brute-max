"""
Smoke tests for the password cracker — standard-library unittest, no extra deps.

Run from the project root:
    python -m unittest discover -s tests

These cover one path per attack mode plus the specific defects fixed in the
audit (see tests/FINDINGS.md): the hash-algorithm guard (F-001) and that Markov
generation is finite and terminates on a real corpus (F-002).
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest

# Make the project root importable however the suite is launched.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import build_verifier, make_hash_verifier, run_attack, format_duration
from bruteforce import build_charset, iter_bruteforce
from wordlist import build_rules, count_words, iter_wordlist, _insert_variants
from markov import iter_markov, train_markov, train_or_load_markov


class HashVerifierTests(unittest.TestCase):
    def test_md5_matches_correct_and_rejects_wrong(self):
        digest = hashlib.md5(b"abc").hexdigest()
        verify = make_hash_verifier("md5", digest)
        self.assertTrue(verify(b"abc"))
        self.assertFalse(verify(b"abd"))

    def test_unknown_algorithm_raises_valueerror(self):
        with self.assertRaises(ValueError):
            build_verifier({"kind": "hash", "algo": "nope256", "digest": "00"})

    def test_non_constructor_attrs_raise_valueerror(self):
        # F-001: these names exist in hashlib but aren't zero-arg constructors;
        # they must raise ValueError, not crash with an uncaught TypeError.
        for algo in ("new", "pbkdf2_hmac", "scrypt"):
            with self.assertRaises(ValueError):
                build_verifier({"kind": "hash", "algo": algo, "digest": "00"})

    def test_wrong_length_digest_raises(self):
        with self.assertRaises(ValueError):
            make_hash_verifier("md5", "aa")  # 1 byte; md5 digests are 16

    def test_variable_length_hash_rejected(self):
        # F-006: shake_128/shake_256 have digest_size 0 and need a length arg at
        # .digest(); they must raise ValueError (not crash in the hot loop),
        # for both an empty and a non-empty target.
        for target in ("", "aa"):
            with self.assertRaises(ValueError):
                make_hash_verifier("shake_128", target)


class BruteForceTests(unittest.TestCase):
    def test_build_charset_bans(self):
        self.assertEqual(build_charset("abcabc", "b"), "acac")

    def test_run_attack_finds_plaintext(self):
        verify = build_verifier({"kind": "plaintext", "plaintext": "cab"})
        candidates = iter_bruteforce("abc", 1, 3, verbose=False)
        pw, attempts, _ = run_attack(verify, candidates, verbose=False)
        self.assertEqual(pw, "cab")
        self.assertGreater(attempts, 0)

    def test_run_attack_reports_not_found(self):
        verify = build_verifier({"kind": "plaintext", "plaintext": "zz"})
        candidates = iter_bruteforce("ab", 1, 2, verbose=False)
        pw, _, _ = run_attack(verify, candidates, verbose=False)
        self.assertIsNone(pw)


class WordlistTests(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("dragon\npassword\n\niloveyou\n")  # blank line on purpose
        self.addCleanup(os.remove, self.path)

    def test_best_rule_count_is_101(self):
        # README advertises ~101 guesses/word for the 'best' rule set.
        self.assertEqual(len(build_rules("best")), 101)

    def test_count_words_skips_blank_lines(self):
        words, chars = count_words(self.path)
        self.assertEqual(words, 3)
        self.assertEqual(chars, len("dragon") + len("password") + len("iloveyou"))

    def test_rules_produce_year_composition(self):
        verify = build_verifier({"kind": "plaintext", "plaintext": "Dragon2024"})
        candidates = iter_wordlist(self.path, build_rules("best"))
        pw, _, _ = run_attack(verify, candidates, verbose=False)
        self.assertEqual(pw, "Dragon2024")

    def test_insert_variants_are_interior_only(self):
        self.assertEqual(list(_insert_variants("abc", "!")), ["a!bc", "ab!c"])


class MarkovTests(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("dragon\npassword\niloveyou\n")
        self.addCleanup(os.remove, self.path)

    def test_generates_a_training_word(self):
        model, n = train_markov(self.path, order=2)
        self.assertEqual(n, 3)
        verify = build_verifier({"kind": "plaintext", "plaintext": "dragon"})
        pw, _, _ = run_attack(verify, iter_markov(model, 2, 1, 15), verbose=False)
        self.assertEqual(pw, "dragon")

    def test_terminates_on_finite_corpus(self):
        # F-002: the generator is finite. Fully draining it must return, not
        # loop forever, even though it emits strings beyond the training words.
        model, _ = train_markov(self.path, order=2)
        count = sum(1 for _ in iter_markov(model, 2, 1, 15))
        self.assertGreater(count, 0)  # produced candidates AND stopped

    def test_cache_is_json_and_round_trips(self):
        # F-008: the on-disk model cache is JSON (a non-executable format), so a
        # cache file that happens to sit next to the corpus cannot run code on
        # load. The cache must parse as JSON and reload to an equivalent model.
        cache = self.path + ".markov-o2.json"
        self.addCleanup(lambda: os.path.exists(cache) and os.remove(cache))
        model = train_or_load_markov(self.path, order=2, max_lines=0,
                                     verbose=False)
        self.assertTrue(os.path.isfile(cache))
        with open(cache, "r", encoding="utf-8") as f:
            self.assertEqual(json.load(f), model)      # parses as JSON
        # Second call loads the cache (newer than corpus) → equivalent model.
        self.assertEqual(
            train_or_load_markov(self.path, order=2, max_lines=0, verbose=False),
            model)


class CliValidationTests(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("dragon\npassword\niloveyou\n")
        self.addCleanup(os.remove, self.path)

    def test_markov_rejects_min_greater_than_max(self):
        # F-005: the markov branch must validate min<=max like brute mode does,
        # instead of silently running empty (0 attempts, "Not found").
        from brute import main
        with self.assertRaises(SystemExit):
            main(["--plaintext", "dragon", "--wordlist", self.path, "--markov",
                  "--min", "10", "--max", "5", "--quiet"])

    def test_markov_rejects_negative_top(self):
        # F-007: a negative --markov-top must be rejected, not silently emit
        # zero candidates and report "Not found. 0 attempts".
        from brute import main
        with self.assertRaises(SystemExit):
            main(["--plaintext", "dragon", "--wordlist", self.path, "--markov",
                  "--markov-top", "-1", "--quiet"])

    def test_markov_rejects_negative_train_lines(self):
        # F-007: a negative --markov-train-lines must be rejected, not silently
        # train on a single line.
        from brute import main
        with self.assertRaises(SystemExit):
            main(["--plaintext", "dragon", "--wordlist", self.path, "--markov",
                  "--markov-train-lines", "-1", "--quiet"])


class FormatDurationTests(unittest.TestCase):
    def test_units(self):
        self.assertEqual(format_duration(float("nan")), "inf")
        self.assertEqual(format_duration(float("inf")), "inf")
        self.assertTrue(format_duration(5).endswith("s"))
        self.assertIn("min", format_duration(120))
        self.assertIn("years", format_duration(5 * 365.25 * 86400))


if __name__ == "__main__":
    unittest.main(verbosity=2)
