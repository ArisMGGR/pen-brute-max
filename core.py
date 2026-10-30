"""
core.py — shared engine for the password cracker.

Everything the attack modes (bruteforce, wordlist, markov) and the CLI have in
common:

  * verifiers       — turn a target (hash or plaintext) into a fast check,
  * run_attack      — the single-process loop that feeds candidates to a verifier,
  * format_duration — human-readable ETAs.

Candidates flow through this module as *bytes* (already encoded by whichever
generator produced them), which keeps the hot loop free of per-guess encoding.
"""

import hashlib
import time

# How often run_attack prints a progress line (attempts).
PROGRESS_EVERY = 250_000


# ----------------------------------------------------------------------
# Verifiers
# ----------------------------------------------------------------------
def make_hash_verifier(algo, target_hex):
    """
    Return a callable(candidate_bytes) -> bool that hashes the candidate.

    The candidate is passed in as *bytes* (already encoded by the generator)
    and compared against the raw target digest — not the hex string. Skipping
    the hex-encode on every guess is a measurable speedup in the hot loop.
    """
    try:
        hasher = getattr(hashlib, algo)
    except AttributeError:
        raise ValueError(f"Unknown hash algorithm: {algo}")

    try:
        target = bytes.fromhex(target_hex.strip())
    except ValueError:
        raise ValueError(f"Target hash is not valid hex: {target_hex!r}")

    digest_size = hasher().digest_size
    if len(target) != digest_size:
        raise ValueError(
            f"{algo} produces {digest_size}-byte digests, but the target is "
            f"{len(target)} bytes - wrong algorithm for this hash?")

    def verify(candidate_bytes):
        return hasher(candidate_bytes).digest() == target

    return verify


def make_plaintext_verifier(target):
    target_bytes = target.encode('utf-8')

    def verify(candidate_bytes):
        return candidate_bytes == target_bytes

    return verify


def build_verifier(params):
    """(Re)build the verify callable from picklable params (used by workers too)."""
    if params['kind'] == 'hash':
        return make_hash_verifier(params['algo'], params['digest'])
    return make_plaintext_verifier(params['plaintext'])


# ----------------------------------------------------------------------
# Duration formatting
# ----------------------------------------------------------------------
def format_duration(seconds):
    """Render a number of seconds using the largest sensible unit."""
    if seconds != seconds or seconds == float('inf'):   # NaN or inf
        return "inf"

    minute = 60
    hour   = 60 * minute
    day    = 24 * hour
    week   = 7 * day
    month  = 30.44 * day        # average Gregorian month
    year   = 365.25 * day       # average Gregorian year

    if seconds < minute:
        return f"{seconds:,.1f}s"
    if seconds < hour:
        return f"{seconds / minute:,.1f} min"
    if seconds < day:
        return f"{seconds / hour:,.1f}h"
    if seconds < week:
        return f"{seconds / day:,.1f} days"
    if seconds < month:
        return f"{seconds / week:,.1f} weeks"
    if seconds < year:
        return f"{seconds / month:,.1f} months"
    return f"{seconds / year:,.1f} years"


# ----------------------------------------------------------------------
# Core attack loop (shared by every mode)
# ----------------------------------------------------------------------
def run_attack(verify, candidates, total_space=None,
               max_attempts=None, verbose=True):
    """
    Feed each candidate to `verify` until it returns True.

    Parameters
    ----------
    verify       : callable(candidate_bytes) -> bool
    candidates   : iterable of candidate byte strings
    total_space  : optional total count, used only for the ETA
    max_attempts : optional safety cap on total attempts
    verbose      : print periodic progress

    Returns (password, attempts, elapsed_seconds); password is None if no
    match was found (or max_attempts was hit).
    """
    attempts = 0
    start = time.time()
    next_report = PROGRESS_EVERY        # cheaper than a modulo every iteration

    for candidate in candidates:
        attempts += 1

        if verify(candidate):
            password = candidate.decode('utf-8', 'replace')
            return password, attempts, time.time() - start

        if attempts == next_report:
            next_report += PROGRESS_EVERY
            if verbose:
                elapsed = time.time() - start
                rate = attempts / elapsed if elapsed > 0 else 0.0
                if total_space and rate > 0:
                    remaining = max(total_space - attempts, 0)
                    eta = f"{format_duration(remaining / rate)} to exhaust all combos"
                else:
                    eta = "? to exhaust all combos"
                latest = candidate.decode('utf-8', 'replace')
                print(f"    [{attempts:,} tried | {rate:,.0f}/s | "
                      f"{elapsed:.1f}s] latest: {latest!r}  |  {eta}")

        if max_attempts is not None and attempts >= max_attempts:
            if verbose:
                print(f"[!] Reached max_attempts ({max_attempts:,}); stopping.")
            return None, attempts, time.time() - start

    return None, attempts, time.time() - start
