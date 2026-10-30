"""
bruteforce.py - exhaustive brute-force candidate generation.

Enumerates every string of length MIN..MAX over a character set. Provides the
charset presets + construction helper, the single-process generator, and the
per-worker slice used by parallel runs. Candidates are yielded as bytes for the
shared verify loop in core.run_attack.
"""

import itertools
import string

# ----------------------------------------------------------------------
# Configuration / character sets
# ----------------------------------------------------------------------
DEFAULT_MIN_LENGTH = 1
DEFAULT_MAX_LENGTH = 15          # <-- hard ceiling for brute-force mode

# Default character set: all printable ASCII (code points 32-126)
DEFAULT_CHARSET = ''.join(chr(c) for c in range(32, 127))

# Convenience presets selectable with --preset
CHARSET_PRESETS = {
    'lower':     string.ascii_lowercase,
    'upper':     string.ascii_uppercase,
    'letters':   string.ascii_letters,
    'digits':    string.digits,
    'alnum':     string.ascii_letters + string.digits,
    'hex':       '0123456789abcdef',
    'punct':     string.punctuation,
    'printable': DEFAULT_CHARSET,
}


def build_charset(base, banned):
    """Remove every character in `banned` from `base`."""
    banned_set = set(banned)
    return ''.join(c for c in base if c not in banned_set)


# ----------------------------------------------------------------------
# Candidate generators
# ----------------------------------------------------------------------
def iter_bruteforce(charset, min_length, max_length, verbose=True):
    """
    Yield every string of length min..max over `charset`, as bytes.

    When the charset is pure ASCII (the common case) we enumerate over the
    byte values directly and build each candidate with bytes(), avoiding a
    string join + UTF-8 encode per guess. Non-ASCII charsets fall back to
    joining characters and encoding, which keeps multi-byte chars intact.
    """
    ascii_only = all(ord(c) < 128 for c in charset)

    for length in range(min_length, max_length + 1):
        if verbose:
            space = len(charset) ** length
            print(f"[*] Length {length}: up to {space:,} candidates "
                  f"({len(charset)} chars in set)")
        if ascii_only:
            cs = charset.encode('ascii')
            for combo in itertools.product(cs, repeat=length):
                yield bytes(combo)
        else:
            for combo in itertools.product(charset, repeat=length):
                yield ''.join(combo).encode('utf-8', 'ignore')


def iter_bruteforce_slice(charset, min_length, max_length,
                          worker_id, num_workers):
    """
    Brute-force candidates (bytes) owned by one worker.

    The space is partitioned by the FIRST character: worker w owns every
    candidate starting with charset[w], charset[w + num_workers], ... The
    union across workers is the full space, with no overlap.
    """
    ascii_only = all(ord(c) < 128 for c in charset)
    if ascii_only:
        cs = charset.encode('ascii')
        firsts = cs[worker_id::num_workers]
        for length in range(min_length, max_length + 1):
            for first in firsts:
                head = bytes((first,))
                for combo in itertools.product(cs, repeat=length - 1):
                    yield head + bytes(combo)
    else:
        chars = list(charset)
        firsts = chars[worker_id::num_workers]
        for length in range(min_length, max_length + 1):
            for first in firsts:
                for combo in itertools.product(chars, repeat=length - 1):
                    yield (first + ''.join(combo)).encode('utf-8', 'ignore')
