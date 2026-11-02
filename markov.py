"""
markov.py — statistical, likelihood-ordered candidate generation.

Rather than walking a keyspace alphabetically, learn character-transition
frequencies from a real password corpus and emit human-like strings first,
following only transitions the corpus actually contains. Provides:

  * train_markov         — count order-k transitions into a model,
  * train_or_load_markov — train (or load a cached model) for reuse,
  * iter_markov          — OMEN-style likelihood-band generator (+ per-worker
                           first-character partition for parallel runs).

Order-k model: the next character depends on the previous k characters. A
special END symbol marks end-of-password, so a walk knows where a word may
plausibly stop. Confining the search to observed transitions keeps it on
realistic strings and bounds the branching.
"""

import json
import math
import os
import time

MARKOV_START = '\x02'      # sentinel for positions before the first char
MARKOV_END = '\x03'        # sentinel marking end-of-password


def train_markov(path, order, max_lines=0):
    """Count order-k character transitions in a corpus -> nested-dict model."""
    from collections import defaultdict
    counts = defaultdict(lambda: defaultdict(int))
    pad = MARKOV_START * order
    n = 0
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            word = line.rstrip('\r\n')
            if not word:
                continue
            ctx = pad
            for ch in word:
                counts[ctx][ch] += 1
                ctx = (ctx + ch)[-order:]
            counts[ctx][MARKOV_END] += 1
            n += 1
            if max_lines and n >= max_lines:
                break
    # Plain dicts serialize cleanly to JSON and load fast.
    return {ctx: dict(nexts) for ctx, nexts in counts.items()}, n


def train_or_load_markov(path, order, max_lines, verbose=True):
    """Train the model, caching it next to the corpus for fast reuse.

    The cache is JSON, not pickle: the model is a plain dict of string contexts
    -> {char: count}, which JSON round-trips exactly. Using a non-executable
    format means loading a cache file that happens to sit next to the corpus
    can never run arbitrary code, even if it was placed there by someone else.
    """
    tag = f".markov-o{order}" + (f"-n{max_lines}" if max_lines else "") + ".json"
    cache = path + tag
    if (os.path.isfile(cache)
            and os.path.getmtime(cache) >= os.path.getmtime(path)):
        if verbose:
            print(f"[*] Loading cached Markov model: {cache}")
        try:
            with open(cache, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (ValueError, OSError):
            # Corrupt or unreadable cache: fall through and retrain.
            if verbose:
                print("[!] Cached model unreadable; retraining.")

    if verbose:
        print(f"[*] Training Markov model (order {order}) from {path} ...")
    t0 = time.time()
    model, n = train_markov(path, order, max_lines)
    if verbose:
        print(f"[*] Trained on {n:,} words in {time.time() - t0:.1f}s "
              f"({len(model):,} contexts)")
    try:
        with open(cache, 'w', encoding='utf-8') as f:
            json.dump(model, f)
        if verbose:
            print(f"[*] Cached model -> {cache}")
    except OSError:
        pass
    return model


def iter_markov(model, order, min_length, max_length, top=0, levels=10,
                worker_id=0, num_workers=1):
    """
    Yield candidates (bytes) in likelihood-band order (OMEN-style).

    Each transition gets an integer "level" from its probability: level 0 is
    the most likely, higher levels are progressively rarer. A string's total
    level is the sum of its transition levels plus the end marker. We then
    enumerate strings by *increasing total level*, so the most probable band
    comes out first, then the next, and so on — a fast, well-ordered
    approximation to a full probability sort (this is how JtR's incremental
    and the OMEN tool work).

    A plain depth-first walk is fast but badly ordered (it exhausts one
    first-letter subtree before reaching others, so common passwords surface
    late); best-first is correctly ordered but far too slow in pure Python.
    Level-banding gives both: near-probability order and millions/sec.

    Only transitions seen in training are followed; `top` (> 0) keeps just the
    N most likely successors per context. `levels` sets the quantisation
    granularity (more = finer ordering, more passes). For parallel runs,
    (worker_id, num_workers) give this worker a round-robin slice of the
    observed first characters.
    """
    pad = MARKOV_START * order

    # Build per-context successor lists, then convert -log(prob) to an integer
    # level scaled so the rarest observed transition sits at `levels - 1`.
    trans = {}
    max_nll = 0.0
    for ctx, nexts in model.items():
        log_total = math.log(sum(nexts.values()))
        lst = [[ch, log_total - math.log(cnt)] for ch, cnt in nexts.items()]
        for _ch, nll in lst:
            if nll > max_nll:
                max_nll = nll
        trans[ctx] = lst
    scale = (levels - 1) / max_nll if max_nll > 0 else 0.0
    for lst in trans.values():
        for item in lst:
            item[1] = int(item[1] * scale)        # nll -> integer level
        lst.sort(key=lambda it: it[1])            # most likely (level 0) first
        if top:
            del lst[top:]

    # For parallel runs, each worker owns a round-robin slice of the observed
    # first characters. The start context is level-sorted, so the heaviest
    # first letters spread across workers rather than piling on worker 0.
    allowed_first = None
    if num_workers > 1:
        firsts = [ch for ch, _lvl in trans.get(pad, []) if ch != MARKOV_END]
        allowed_first = set(firsts[worker_id::num_workers])

    def emit(total):
        # Iterative depth-first walk that yields in the same order a recursive
        # descent would, but keeps its own explicit stack so a large max_length
        # cannot overflow the interpreter's recursion limit (emit used to recurse
        # once per candidate character; a --max above ~1000 crashed).
        root_node = trans.get(pad)
        if not root_node:
            return
        # Each frame: [s, remaining, iterator over that context's (ch, lvl)].
        stack = [["", total, iter(root_node)]]
        while stack:
            s, remaining, it = stack[-1]
            slen = len(s)
            descended = False
            for ch, lvl in it:
                if lvl > remaining:
                    break                          # sorted: the rest are rarer
                if ch == MARKOV_END:
                    if lvl == remaining and slen >= min_length:
                        yield s.encode('utf-8', 'ignore')
                elif slen < max_length:
                    if slen == 0 and allowed_first is not None and ch not in allowed_first:
                        continue                   # another worker owns this first char
                    child = s + ch
                    child_node = trans.get((pad + child)[-order:])
                    if child_node:
                        stack.append([child, remaining - lvl, iter(child_node)])
                        descended = True
                        break                      # walk the child before the rest
            if not descended:
                stack.pop()

    max_total = (levels - 1) * (max_length + 1)
    for total in range(max_total + 1):
        yield from emit(total)
