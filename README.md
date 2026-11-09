# brute

A learning-oriented password cracker in pure Python. It generates candidate
passwords and checks each against a target hash (or plaintext) until it finds a
match. Four candidate-generation strategies are included, plus multi-core
parallelism.

> **Authorized use only.** Use this against your own hashes, test data, CTF
> targets, or systems you have explicit permission to test — never against
> accounts or data that aren't yours.

No third-party dependencies — Python 3.8+ standard library only.

> **Wordlist not included.** The examples below use `rockyou.txt`, which is
> gitignored and not redistributed here. Download it separately (it ships with
> Kali at `/usr/share/wordlists/rockyou.txt.gz`, or fetch it from the SecLists
> project) and place it in the project root — or pass any wordlist path you like.

---

## Quick start

```bash
# Brute force an MD5 over lowercase letters, lengths 1–6
python brute.py --hash md5:5f4dcc3b5aa765d61d8327deb882cf99 --preset lower --min 1 --max 6

# Dictionary attack with mangling rules, using all CPU cores
python brute.py --hash md5:... --wordlist rockyou.txt --rules best --workers 0

# Markov (statistical) attack, order-3 model trained on the wordlist
python brute.py --hash md5:... --wordlist rockyou.txt --markov --markov-order 3 \
                --workers 0 --max-attempts 50000000

# Test without a hash (handy for trying things out)
python brute.py --plaintext iloveyou --wordlist rockyou.txt --markov
```

Run `python brute.py -h` for the full option reference.

---

## Attack modes

The target is always `--hash ALGO:HEX` (e.g. `md5:...`, `sha256:...`) or
`--plaintext VALUE`. The mode is chosen by which flags you pass:

### 1. Brute force (default)
Enumerates **every** string of length `--min`..`--max` over a character set.

```bash
python brute.py --hash md5:... --preset alnum --ban "0O1l" --min 4 --max 8
```

- `--preset` — named set: `lower upper letters digits alnum hex punct printable`
- `--charset "abc123"` — a literal set (overrides `--preset`)
- `--ban "chars"` — remove characters from the set

Complete but only feasible for short passwords (keyspace is `charset^length`).

### 2. Wordlist + rules (`--wordlist`)
Tries each line of a wordlist, transformed by a rule set that models human
habits (capitalize, leet, append year/symbol, reverse, …).

```bash
python brute.py --hash md5:... --wordlist rockyou.txt --rules best
```

- `--rules none` — words verbatim
- `--rules basic` — case/leet/reverse + common suffixes
- `--rules best` *(default)* — adds year suffixes (1990–2030) and compositions (~101 guesses/word)

### 3. Insertion attack (`--insert`, with a wordlist)
Layers on top of the rules: inserts each given character at every **interior**
position of each word (`dragon` + `!` → `d!ragon … drago!n`).

```bash
python brute.py --hash md5:... --wordlist rockyou.txt --insert "!-_"
```

The banner shows how many candidates this adds.

### 4. Markov (`--markov`, with a wordlist as training data)
Trains a character-transition model on the wordlist, then emits guesses in
**likelihood order** (OMEN-style). Finds human-like passwords early *without*
needing them to be in a list — on a real corpus it won't finish in any
practical time, so cap it.

```bash
python brute.py --hash md5:... --wordlist rockyou.txt --markov --markov-order 3 \
                --workers 0 --max-attempts 50000000
```

- `--markov-order K` — context length (2 default; 3 is sharper; 4+ overfits)
- `--markov-top N` — keep only the N likeliest next-chars per context
- `--markov-train-lines N` — train on the first N lines only
- The trained model is cached next to the corpus (`rockyou.txt.markov-o2.json`)

> In Markov mode `--wordlist` is **training data**, not a guess list: the tool
> learns what human passwords look like, then generates its own.

---

## Shared options

| Flag | Meaning |
|---|---|
| `--workers N` / `-j N` | parallel worker processes; `0` = all cores (default 1) |
| `--max-attempts N` | cap the run (recommended for Markov, whose ordered keyspace is astronomically large) |
| `--quiet` / `-q` | suppress progress lines |

Parallel runs partition the search (by first character, or round-robin lines)
across processes. Note: the attempt **count** in parallel mode is a batched
aggregate across workers, so it's approximate; single-worker counts are exact.

---

## Module layout

The code is split into focused modules:

| File | Role |
|---|---|
| `brute.py` | **CLI only** — argument parsing, plan banner, dispatch |
| `core.py` | *shared* — verifiers, the `run_attack` loop, ETA formatting |
| `bruteforce.py` | brute-force generators + character-set handling |
| `wordlist.py` | dictionary mode — rule sets + insertion |
| `markov.py` | Markov training, caching, likelihood-ordered generator |
| `parallel.py` | *shared* — multiprocessing orchestration for any mode |

### Dependency graph

```
brute.py (CLI)
   ├─ core        (verifiers, run_attack, ETA)
   ├─ bruteforce  (brute generators + charset)
   ├─ wordlist    (rules + insert)
   ├─ markov      (train + ordered generator)
   └─ parallel ──┬─ core
                 ├─ bruteforce
                 ├─ wordlist
                 └─ markov
```

Acyclic: `core` and the three mode modules stand alone; `parallel` pulls each
mode's per-worker slice generator; `brute.py` wires everything together. The
shared code (`core`, `parallel`) is common to *all* modes, which is why there's
one engine rather than per-pair shared files.

---

## How it works

Every mode produces **byte** candidates that flow into one shared loop
(`core.run_attack`), which hashes each and compares the raw digest to the
target. That decoupling of *candidate generation* from *checking* is why the
same loop serves brute force, wordlists, and Markov alike.

---

## Tests

A small smoke-test suite (standard-library `unittest`, no extra dependencies)
covers one path per mode plus the audited edge cases. From the project root:

```bash
python -m unittest discover -s tests
```
