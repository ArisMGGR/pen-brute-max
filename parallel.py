"""
parallel.py — multiprocessing orchestration shared by every mode.

A single Python process tops out around ~2M guesses/sec regardless of how fast
the hash itself is — the limit is interpreter loop overhead, not the C hash
function. To go faster we run N worker processes, each owning a disjoint slice
of the search space, so throughput scales with cores.

Everything a worker needs is passed as plain data (a `params` dict), not as
closures or lambdas, because those can't be pickled to child processes on
Windows (spawn). Each worker rebuilds its verifier and candidate slice from the
mode modules.
"""

import multiprocessing as mp
import time

from core import build_verifier, format_duration, PROGRESS_EVERY
from bruteforce import iter_bruteforce_slice
from wordlist import iter_wordlist_slice
from markov import iter_markov, train_or_load_markov

# Workers report progress by adding to this counter every COUNT_BATCH tries
# (batched to keep lock contention low) and check the stop flag at the same
# cadence, so a hit in one worker halts the rest within a batch.
COUNT_BATCH = 50_000


def _worker(mode, params, worker_id, num_workers, found, result, counter):
    """One worker process: hash its slice until a hit or the slice is done."""
    verify = build_verifier(params)
    if mode == 'brute':
        candidates = iter_bruteforce_slice(
            params['charset'], params['min_length'], params['max_length'],
            worker_id, num_workers)
    elif mode == 'markov':
        model = train_or_load_markov(params['wordlist'], params['order'],
                                     params['train_lines'], verbose=False)
        candidates = iter_markov(
            model, params['order'], params['min_length'], params['max_length'],
            params['top'], worker_id=worker_id, num_workers=num_workers)
    else:
        candidates = iter_wordlist_slice(
            params['wordlist'], params['rules'], params.get('insert', ''),
            worker_id, num_workers)

    local = 0
    try:
        for candidate in candidates:
            if verify(candidate):
                result.put(candidate.decode('utf-8', 'replace'))
                found.set()
                return
            local += 1
            if local % COUNT_BATCH == 0:
                with counter.get_lock():
                    counter.value += COUNT_BATCH
                if found.is_set():
                    return
    finally:
        with counter.get_lock():
            counter.value += local % COUNT_BATCH


def run_parallel(mode, params, total_space, num_workers,
                 max_attempts=None, verbose=True):
    """
    Run the attack across `num_workers` processes.

    Returns (password, attempts, elapsed_seconds). `attempts` is the aggregate
    across workers (approximate, batched in COUNT_BATCH chunks).
    """
    found = mp.Event()
    result = mp.Queue()
    counter = mp.Value('q', 0)

    procs = [mp.Process(target=_worker,
                        args=(mode, params, wid, num_workers,
                              found, result, counter))
             for wid in range(num_workers)]

    start = time.time()
    for pr in procs:
        pr.start()

    next_report = PROGRESS_EVERY
    try:
        while any(pr.is_alive() for pr in procs):
            time.sleep(0.2)
            done = counter.value

            if found.is_set():
                break
            if max_attempts is not None and done >= max_attempts:
                if verbose:
                    print(f"[!] Reached max_attempts ({max_attempts:,}); "
                          f"stopping.")
                found.set()
                break
            if verbose and done >= next_report:
                while next_report <= done:
                    next_report += PROGRESS_EVERY
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0.0
                if total_space and rate > 0:
                    remaining = max(total_space - done, 0)
                    eta = f"{format_duration(remaining / rate)} to exhaust all combos"
                else:
                    eta = "? to exhaust all combos"
                print(f"    [{done:,} tried | {rate:,.0f}/s | {elapsed:.1f}s] "
                      f"|  {eta} ({num_workers} workers)")
    except KeyboardInterrupt:
        found.set()
        for pr in procs:
            pr.join(timeout=5)
        raise

    password = None
    if found.is_set():
        try:
            password = result.get(timeout=2)
        except Exception:
            password = None

    found.set()                      # signal any stragglers to stop
    for pr in procs:
        pr.join(timeout=5)
        if pr.is_alive():
            pr.terminate()

    return password, counter.value, time.time() - start
