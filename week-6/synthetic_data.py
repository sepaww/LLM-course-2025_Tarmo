#from llama_index.llms.ollama import Ollama
#import dspy

#llm = Ollama(model="llama3", request_timeout=60.0)

# you can use DSPY (https://github.com/stanfordnlp/dspy), but you can also choose another method of interacting with an LLM
#dspy.settings.configure(lm=llm)

# Task: implement a method, that will take a query string as input and produce N misspelling variants of the query.
# These variants with typos will be used to test a search engine quality.
# Example
# Query: machine learning applications
# Possible Misspellings:
# "machin learning applications" (missing "e" in "machine")
# "mashine learning applications" (phonetically similar spelling of "machine")
# "machine lerning aplications" (missing "a" in "learning" and "p" in "applications")
# "machin lerning aplications" (combining multiple typos)
# "mahcine learing aplication" (transposed letters in "machine" and typos in "learning" and "applications")
#
# Questions:
# 1. Does the search engine produce the same results for all the variants?
# 2. Do all variants make sense?
# 3. How to improve robustness of the method, for example, skip known abbreviations, like JFK or NBC.
# 4. Can you test multiple LLMs and figure out which one is the best?
# 5. Do the misspellings capture a variety of error types (phonetic, omission, transposition, repetition)?

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import pandas as pd

N_VARIANTS_PER_QUERY = 2
RNG_SEED = 7


TYPO_PROBABILITY = 10.0

DEFAULT_ABBREV_WHITELIST: Set[str] = {
    "JFK", "LAX", "SFO", "LHR", "CDG", "NYC", "USA", "UK", "EU", "UN",
    "NBA", "NFL", "NHL", "MLB", "FBI", "CIA", "NASA", "WHO", "BBC", "NBC",
}

PHONETIC_SWAPS: Sequence[Tuple[str, str]] = (
    ("ph", "f"),
    ("ck", "k"),
    ("c", "k"),
    ("q", "k"),
    ("x", "ks"),
    ("tion", "shun"),
    ("sion", "zhun"),
    ("v", "w"),
    ("y", "i"),
    ("oo", "u"),
    ("ee", "i"),
)


@dataclass(frozen=True)
class Variant:
    original: str
    misspelled: str
    error_types: Tuple[str, ...]


def _is_abbrev(token: str, whitelist: Set[str]) -> bool:
    if token in whitelist:
        return True
    if token.isupper() and 2 <= len(token) <= 6 and token.isalpha():
        return True
    return False


def _tokenize(query: str) -> List[str]:
    parts = query.split()
    tokens: List[str] = []
    for p in parts:
        if re.match(r"^(https?://|www\.)", p, flags=re.I):
            tokens.append(p)
        else:
            tokens.extend(re.findall(r"[A-Za-z]+|\d+|[^\w\s]", p))
    return tokens


def _join_tokens(tokens: List[str]) -> str:
    out: List[str] = []
    for t in tokens:
        if out and re.fullmatch(r"[^\w\s]", t):
            out[-1] = out[-1] + t
        else:
            out.append(t)
    return " ".join(out).replace("  ", " ").strip()


def _omission(word: str, rng: random.Random) -> Optional[str]:
    if len(word) < 3:
        return None
    i = rng.randrange(0, len(word))
    return word[:i] + word[i + 1 :]


def _transposition(word: str, rng: random.Random) -> Optional[str]:
    if len(word) < 3:
        return None
    i = rng.randrange(0, len(word) - 1)
    if word[i] == word[i + 1]:
        return None
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def _repetition(word: str, rng: random.Random) -> Optional[str]:
    if len(word) < 2:
        return None
    i = rng.randrange(0, len(word))
    return word[:i] + word[i] + word[i:]


def _substitution(word: str, rng: random.Random) -> Optional[str]:
    if len(word) < 2:
        return None
    letters = "abcdefghijklmnopqrstuvwxyz"
    i = rng.randrange(0, len(word))
    ch = word[i].lower()
    repl = rng.choice([c for c in letters if c != ch])
    if word[i].isupper():
        repl = repl.upper()
    return word[:i] + repl + word[i + 1 :]


def _phonetic(word: str, rng: random.Random) -> Optional[str]:
    w = word.lower()
    swaps = list(PHONETIC_SWAPS)
    rng.shuffle(swaps)
    for a, b in swaps:
        if a in w and len(word) >= 3:
            return re.sub(a, b, word, count=1, flags=re.IGNORECASE)
    return None


ERROR_FUNCS = (
    ("omission", _omission),
    ("transposition", _transposition),
    ("repetition", _repetition),
    ("substitution", _substitution),
    ("phonetic", _phonetic),
)

def generate_misspellings(
    query: str,
    n: int,
    *,
    rng: Optional[random.Random] = None,
    max_typos_per_query: int = 30,
    abbrev_whitelist: Optional[Set[str]] = None,
    typo_probability: float = TYPO_PROBABILITY,
) -> List[Variant]:
    rng = rng or random.Random()
    abbrev_whitelist = abbrev_whitelist or set(DEFAULT_ABBREV_WHITELIST)

    tokens = _tokenize(query)
    word_idxs = [
        i for i, t in enumerate(tokens)
        if t.isalpha() and not _is_abbrev(t, abbrev_whitelist)
    ]
    if not word_idxs:
        return []

    variants: List[Variant] = []
    seen: Set[str] = {query}

    attempts = 0
    max_attempts = max(50, n * 20)

    while len(variants) < n and attempts < max_attempts:
        attempts += 1

        if rng.random() > typo_probability:
            out = query
            applied: List[str] = []
        else:
            new_tokens = tokens[:]
            k = rng.randint(1, max_typos_per_query)
            chosen = rng.sample(word_idxs, k=min(k, len(word_idxs)))

            applied = []
            for idx in chosen:
                w = new_tokens[idx]
                for _ in range(3):
                    etype, fn = rng.choice(ERROR_FUNCS)
                    mutated = fn(w, rng)
                    if mutated and mutated.lower() != w.lower():
                        new_tokens[idx] = mutated
                        applied.append(etype)
                        break

            out = _join_tokens(new_tokens)
            out = re.sub(r"\s+", " ", out).strip()

        if not applied or out in seen:
            continue
        if sum(1 for t in _tokenize(out) if t.isalpha() and len(t) == 1) > 1:
            continue

        seen.add(out)
        variants.append(Variant(original=query, misspelled=out, error_types=tuple(applied)))

    return variants



script_dir = Path(__file__).resolve().parent
csv_path = script_dir / "web_search_queries.csv"

out_dir = script_dir / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "misspelled_queries.csv"

rng = random.Random(RNG_SEED)

rows = []
df = pd.read_csv(csv_path)

for _, row in df.iterrows():
    topic = row["Topic"] if "Topic" in df.columns else ""
    query = str(row["Query"]).strip()

    for v in generate_misspellings(query, N_VARIANTS_PER_QUERY, rng=rng):
        rows.append(
            {
                "Topic": topic,
                "OriginalQuery": v.original,
                "VariantQuery": v.misspelled,
                "ErrorTypes": ",".join(v.error_types),
            }
        )

out_df = pd.DataFrame(rows)
out_df.to_csv(out_path, index=False)

print(f"Wrote {len(out_df)} misspelled queries to {out_path}")
