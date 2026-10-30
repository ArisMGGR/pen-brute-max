"""
brute.py - command-line interface for the password cracker.

A thin CLI layer: it parses arguments, builds the target + per-mode parameters,
prints the plan, and dispatches to the right generator through the shared
run_attack loop (or run_parallel for multi-core). The actual attack logic lives
in the mode modules:

    core.py        shared verifiers, run_attack loop, ETA formatting
    bruteforce.py  exhaustive brute force over a character set
    wordlist.py    dictionary attack: rules (none/basic/best) + insertion
    markov.py      statistical, likelihood-ordered generation
    parallel.py    shared multiprocessing orchestration

Usage examples:
    # Brute force an MD5 hash over lowercase, lengths 1-6
    python brute.py --hash md5:5f4dcc... --preset lower --min 1 --max 6

    # Wordlist attack with the default ("best") rule set, all cores
    python brute.py --hash md5:5f4dcc... --wordlist rockyou.txt --workers 0

    # Markov mode (order 3), trained on the wordlist
    python brute.py --hash md5:5f4dcc... --wordlist rockyou.txt --markov \
                    --markov-order 3 --workers 0 --max-attempts 50000000
"""

import argparse
import os
import sys

from core import build_verifier, run_attack
from bruteforce import (CHARSET_PRESETS, DEFAULT_CHARSET, DEFAULT_MAX_LENGTH,
                        DEFAULT_MIN_LENGTH, build_charset, iter_bruteforce)
from wordlist import build_rules, count_words, iter_wordlist
from markov import iter_markov, train_or_load_markov
from parallel import run_parallel


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Password cracker: brute-force, wordlist+rules, or markov.")

    # What are we looking for?
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--hash", "-H",
                     help="Target hash as ALGO:HEX (e.g. md5:5f4dcc...)")
    src.add_argument("--plaintext", "-p",
                     help="Plaintext target (for testing)")

    # Mode: wordlist (if given) else brute force; --markov overrides to markov
    p.add_argument("--wordlist", "-w",
                   help="Path to a wordlist; enables wordlist mode")
    p.add_argument("--rules", choices=["none", "basic", "best"], default="best",
                   help="Mangling rule set for wordlist mode (default: best)")
    p.add_argument("--insert", default="", metavar="CHARS",
                   help="Insertion attack (wordlist mode): insert each of "
                        "these chars at every interior position of each word, "
                        "e.g. --insert '!-_'. Adds num_chars*(avg_len-1) "
                        "candidates per word - see the banner.")

    # Markov mode (statistical, likelihood-ordered)
    p.add_argument("--markov", action="store_true",
                   help="Markov mode: train on --wordlist and emit guesses in "
                        "likelihood order (a statistically-ordered brute force)")
    p.add_argument("--markov-order", type=int, default=2, metavar="K",
                   help="Markov context length (default: %(default)s)")
    p.add_argument("--markov-train-lines", type=int, default=0, metavar="N",
                   help="Train on only the first N corpus lines (0 = all)")
    p.add_argument("--markov-top", type=int, default=0, metavar="N",
                   help="Keep only the N most likely next-chars per context "
                        "(0 = all seen); smaller = faster, more focused")

    # Brute-force length bounds
    p.add_argument("--min", type=int, default=DEFAULT_MIN_LENGTH,
                   dest="min_length",
                   help="Minimum length, brute mode (default: %(default)s)")
    p.add_argument("--max", type=int, default=DEFAULT_MAX_LENGTH,
                   dest="max_length",
                   help="Maximum length, brute mode (default: %(default)s)")

    # Brute-force character set selection
    p.add_argument("--preset", choices=sorted(CHARSET_PRESETS),
                   help="Named character set preset (default: printable)")
    p.add_argument("--charset", "-c",
                   help="Literal character set (overrides --preset)")
    p.add_argument("--ban", "-b", default="",
                   help="Characters to remove from the character set")

    # Performance
    p.add_argument("--workers", "-j", type=int, default=1,
                   help="Parallel worker processes; 0 = all cores "
                        "(default: %(default)s)")

    # Optional safety net
    p.add_argument("--max-attempts", type=int, default=None,
                   help="Optional cap on total attempts")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress progress output")

    args = p.parse_args(argv)
    verbose = not args.quiet

    # ---- Worker count ----
    workers = args.workers
    if workers == 0:
        workers = os.cpu_count() or 1
    if workers < 1:
        p.error("--workers must be >= 0")

    # ---- Build target params (picklable; also validates early) ----
    if args.hash:
        if ":" not in args.hash:
            p.error("--hash must be in the form ALGO:HEX")
        algo, digest = args.hash.split(":", 1)
        params = {'kind': 'hash', 'algo': algo.lower(), 'digest': digest}
        target_desc = f"{algo.lower()} hash {digest}"
    else:
        params = {'kind': 'plaintext', 'plaintext': args.plaintext}
        target_desc = f"plaintext {args.plaintext!r}"

    try:
        build_verifier(params)          # validate algo / hex / digest length now
    except ValueError as e:
        p.error(str(e))

    markov_candidates = None

    # ====================================================================
    # Markov mode (statistical, likelihood-ordered)
    # ====================================================================
    if args.markov:
        if not args.wordlist:
            p.error("--markov needs a training corpus: pass --wordlist FILE")
        if not os.path.isfile(args.wordlist):
            p.error(f"Training corpus not found: {args.wordlist}")
        if args.markov_order < 1:
            p.error("--markov-order must be >= 1")

        mode = 'markov'
        total = None
        params['wordlist'] = args.wordlist
        params['order'] = args.markov_order
        params['train_lines'] = args.markov_train_lines
        params['top'] = args.markov_top
        params['min_length'] = args.min_length
        params['max_length'] = args.max_length

        print(f"[*] Target        : {target_desc}")
        print(f"[*] Mode          : markov (order {args.markov_order})")
        print(f"[*] Corpus        : {args.wordlist}")
        print(f"[*] Length range  : {args.min_length}..{args.max_length}")
        # Train/load once in the parent so workers just load the cache.
        model = train_or_load_markov(args.wordlist, args.markov_order,
                                     args.markov_train_lines, verbose)
        if workers == 1:
            markov_candidates = iter_markov(model, args.markov_order,
                                            args.min_length, args.max_length,
                                            args.markov_top)
        print(f"[*] Enumeration   : likelihood bands (OMEN-style)")
        print(f"[*] Workers       : {workers}")
        print()

    # ====================================================================
    # Wordlist mode
    # ====================================================================
    elif args.wordlist:
        if not os.path.isfile(args.wordlist):
            p.error(f"Wordlist not found: {args.wordlist}")

        mode = 'wordlist'
        rules = build_rules(args.rules)
        word_count, char_count = count_words(args.wordlist)
        inserts = ''.join(dict.fromkeys(args.insert))
        # Upper bound: dedup per word only reduces the real total.
        rule_total = word_count * len(rules)
        insert_total = len(inserts) * max(char_count - word_count, 0)
        total = rule_total + insert_total
        params['wordlist'] = args.wordlist
        params['rules'] = args.rules
        params['insert'] = args.insert

        print(f"[*] Target        : {target_desc}")
        print(f"[*] Mode          : wordlist ({args.wordlist})")
        print(f"[*] Words         : {word_count:,}")
        print(f"[*] Rule set      : {args.rules} ({len(rules)} rules/word)")
        if inserts:
            print(f"[*] Insert attack : {inserts!r} "
                  f"(+{insert_total:,} candidates)")
        print(f"[*] Candidates    : up to {total:,} (before dedup)")
        print(f"[*] Workers       : {workers}")
        print()

    # ====================================================================
    # Brute-force mode
    # ====================================================================
    else:
        mode = 'brute'
        if args.charset is not None:
            base = args.charset
        elif args.preset is not None:
            base = CHARSET_PRESETS[args.preset]
        else:
            base = DEFAULT_CHARSET

        charset = build_charset(base, args.ban)
        if not charset:
            p.error("Character set is empty after applying --ban")

        if args.max_length > DEFAULT_MAX_LENGTH:
            print(f"[!] --max {args.max_length} exceeds the configured ceiling "
                  f"({DEFAULT_MAX_LENGTH}); clamping.")
            args.max_length = DEFAULT_MAX_LENGTH
        if args.min_length < 1:
            p.error("--min must be >= 1")
        if args.max_length < args.min_length:
            p.error("--max must be >= --min")

        total = sum(len(charset) ** L
                    for L in range(args.min_length, args.max_length + 1))
        params['charset'] = charset
        params['min_length'] = args.min_length
        params['max_length'] = args.max_length

        print(f"[*] Target        : {target_desc}")
        print(f"[*] Mode          : brute force")
        print(f"[*] Length range  : {args.min_length}..{args.max_length}")
        print(f"[*] Char set      : {len(charset)} chars -> {charset!r}")
        print(f"[*] Candidates    : {total:,}")
        print(f"[*] Workers       : {workers}")
        print()

    # ---- Run ----
    try:
        if workers > 1:
            password, attempts, elapsed = run_parallel(
                mode, params, total, workers,
                max_attempts=args.max_attempts, verbose=verbose)
        elif mode == 'markov':
            verify = build_verifier(params)
            password, attempts, elapsed = run_attack(
                verify, markov_candidates,
                total_space=None,              # true total is unknown/pruned
                max_attempts=args.max_attempts,
                verbose=verbose,
            )
        else:
            verify = build_verifier(params)
            if mode == 'wordlist':
                candidates = iter_wordlist(args.wordlist,
                                           build_rules(args.rules), args.insert)
            else:
                candidates = iter_bruteforce(params['charset'],
                                             params['min_length'],
                                             params['max_length'], verbose)
            password, attempts, elapsed = run_attack(
                verify, candidates,
                total_space=total,
                max_attempts=args.max_attempts,
                verbose=verbose,
            )
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
        return 130

    print()
    if password is not None:
        print(f"[+] FOUND after {attempts:,} attempts in {elapsed:.2f}s: "
              f"{password!r}")
        return 0

    print(f"[-] Not found. {attempts:,} attempts in {elapsed:.2f}s.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
