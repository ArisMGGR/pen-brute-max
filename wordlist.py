"""
wordlist.py — dictionary attack with rule-based mangling and an insertion attack.

The "upgraded" list mode. Each wordlist entry is tried as-is and transformed by
a chosen rule set (capitalize, leet, append year/symbol, reverse, ...), and
optionally with symbols inserted at interior positions. Provides:

  * build_rules         — the none/basic/best rule sets,
  * count_words         — (word_count, total_chars) for sizing the run,
  * iter_wordlist       — single-process generator,
  * iter_wordlist_slice — per-worker slice for parallel runs.

Candidates are yielded as bytes for the shared verify loop in core.run_attack.
"""

# Years appended by the rule engine, e.g. "password" -> "password2024"
YEAR_MIN = 1990
YEAR_MAX = 2030

# Leet substitutions applied all-at-once by the 'leet' rule
LEET_MAP = {'a': '@', 'o': '0', 'e': '3', 'i': '1', 's': '5', 't': '7'}


# ----------------------------------------------------------------------
# Rules (mangling)
#
# Each rule is a (name, func) pair where func maps ONE word -> ONE candidate.
# This mirrors how hashcat/john rule files work: fan-out comes from having many
# rules (e.g. one per year), not from a rule returning a list. Rules are ordered
# cheap-and-high-hit first, so common passwords fall early.
# ----------------------------------------------------------------------
def _capitalize(word):
    return word[:1].upper() + word[1:]


def _leet(word):
    return ''.join(LEET_MAP.get(c.lower(), c) for c in word)


def build_rules(level):
    """Return the ordered list of (name, func) rules for a given level."""
    rules = []

    def add(name, fn):
        rules.append((name, fn))

    # The word itself is always tried first.
    add("as-is", lambda w: w)

    if level == "none":
        return rules

    # --- basic single transforms (1 candidate each) ---
    add("capitalize", _capitalize)
    add("upper",      lambda w: w.upper())
    add("lower",      lambda w: w.lower())
    add("toggle",     lambda w: w.swapcase())
    add("reverse",    lambda w: w[::-1])
    add("duplicate",  lambda w: w + w)
    add("leet",       _leet)

    # --- common suffixes ---
    for suf in ("1", "123", "1234", "!", "!!", ".", "?", "1!"):
        add(f"+'{suf}'", lambda w, s=suf: w + s)

    if level == "basic":
        return rules

    # --- 'best': year suffixes + a few high-hit compositions ---
    for year in range(YEAR_MIN, YEAR_MAX + 1):
        y = str(year)
        add(f"+{y}",     lambda w, y=y: w + y)
        add(f"cap+{y}",  lambda w, y=y: _capitalize(w) + y)

    add("cap+'!'",   lambda w: _capitalize(w) + "!")
    add("leet+'!'",  lambda w: _leet(w) + "!")
    add("cap+leet",  lambda w: _leet(_capitalize(w)))

    return rules


# ----------------------------------------------------------------------
# Sizing + insertion helpers
# ----------------------------------------------------------------------
def count_words(path):
    """
    Return (word_count, total_chars) for the non-blank lines in a wordlist.

    total_chars (the sum of word lengths) lets the caller size the insertion
    fan-out: inserting a symbol at every interior position adds (len - 1)
    candidates per word, so the total added is num_symbols * (chars - words).
    """
    words = 0
    chars = 0
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            word = line.rstrip('\r\n')
            if not word:
                continue
            words += 1
            chars += len(word)
    return words, chars


def _insert_variants(word, insert_chars):
    """
    Yield `word` with each char in `insert_chars` placed at every INTERIOR
    position (between two existing characters). The ends are intentionally
    skipped because prepend/append already cover them. A word of length L
    yields (L - 1) variants per insert char.
    """
    length = len(word)
    for ch in insert_chars:
        for i in range(1, length):
            yield word[:i] + ch + word[i:]


# ----------------------------------------------------------------------
# Candidate generators
# ----------------------------------------------------------------------
def iter_wordlist(path, rules, insert_chars="", dedup=True):
    """
    Yield each word in the wordlist transformed by every rule, then (if
    insert_chars is given) the word with each symbol inserted at every
    interior position.

    Duplicates produced for the SAME word (e.g. capitalize of an already
    capitalized word) are skipped, so no candidate is hashed twice per
    word — this keeps the work close to the ETA's upper bound.
    """
    inserts = ''.join(dict.fromkeys(insert_chars))   # unique, order-preserving
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            word = line.rstrip('\r\n')
            if not word:
                continue
            seen = set()
            for _name, fn in rules:
                candidate = fn(word)
                if dedup:
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                yield candidate.encode('utf-8', 'ignore')
            for candidate in _insert_variants(word, inserts):
                if dedup:
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                yield candidate.encode('utf-8', 'ignore')


def iter_wordlist_slice(path, level, insert_chars, worker_id, num_workers,
                        dedup=True):
    """Wordlist candidates (bytes) owned by one worker (round-robin lines)."""
    rules = build_rules(level)
    inserts = ''.join(dict.fromkeys(insert_chars))
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        idx = -1
        for line in f:
            word = line.rstrip('\r\n')
            if not word:
                continue
            idx += 1
            if idx % num_workers != worker_id:
                continue
            seen = set()
            for _name, fn in rules:
                candidate = fn(word)
                if dedup:
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                yield candidate.encode('utf-8', 'ignore')
            for candidate in _insert_variants(word, inserts):
                if dedup:
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                yield candidate.encode('utf-8', 'ignore')
