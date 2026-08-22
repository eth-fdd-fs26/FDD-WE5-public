"""Measuring retrieval quality — the part of RAG that turns opinion into evidence.

Every choice on the read side (keyword vs embedding vs hybrid, how wide to
over-retrieve) and on the write side (chunk size, overlap, strategy) changes
which chunks reach the LLM. Eyeballing one query cannot tell you whether a
change helped: the only honest way to compare is a fixed set of questions whose
correct answer you already know, scored the same way every time.

That is what this module is:

    EVAL_SET     — 12 questions over ``data/``, each labelled with the document
                   that answers it (its "gold" source) and with a *kind*.
    recall_at_k  — for what fraction of questions does the gold *document*
                   appear anywhere in the top k results?
    answer_recall_at_k
                 — stricter: does the retrieved *text* actually contain the
                   passage that answers the question?
    context_cost — how many words of context that answer cost you.

**Recall@k**, the standard retrieval metric, is deliberately lenient about
ranking: it asks only "did the right document make the cut?". k matters — a RAG
system feeds its top few chunks to the LLM, so recall@3 is close to "will the
model actually see the answer?", while recall@1 is the strict version and
recall@20 tells you whether the document is even reachable.

The two **kinds** exist to expose the tradeoff that motivates hybrid search:

    "lexical"  — the question contains the rare, exact term the document uses
                 (MFA, PIP, MNPI). Keyword search is very strong here.
    "semantic" — the question is phrased the way a person would ask it, sharing
                 little vocabulary with the document. Keyword search struggles;
                 a real embedding model is what rescues these.

A single averaged number would hide exactly that split, which is why
``recall_at_k`` reports per kind *and* overall.

Why three metrics and not one: document recall alone cannot tell chunking
strategies apart. Retrieve one un-chunked 1,200-word document and you "hit" the
gold document every time — while handing the LLM ten times the context, most of
it irrelevant. ``answer_recall_at_k`` checks the answer is really in what you
retrieved, and ``context_cost`` prices it. Read all three together: the goal is
high answer recall at low context cost.

Caveat worth repeating to students: these questions were written by hand against
this corpus. A dozen questions is enough to see a trend, not enough to certify a
system. Real evaluation sets are larger, and are built from questions users
actually asked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from models import Chunk

#: One retrieval call: (query) -> ranked [(Chunk, score)].
RetrieveFn = Callable[[str], list[tuple[Chunk, float]]]


@dataclass(frozen=True)
class EvalQuery:
    """One graded question.

    Attributes:
        question: what the user asks.
        gold_source: the ``source`` metadata value of the document that answers
            it. Grading at *document* level (rather than a specific chunk) keeps
            the set valid when you change chunk_size — which is the whole point,
            since sweeping chunk_size is one of the things we want to measure.
        kind: ``"lexical"`` or ``"semantic"`` — see the module docstring.
        answer_phrase: a short verbatim string from the passage that actually
            answers the question. Retrieving the right *document* is not the
            same as retrieving the right *passage*, and only this can tell them
            apart — see ``answer_recall_at_k``.
        note: why this question is in the set (what it is meant to probe).
    """

    question: str
    gold_source: str
    kind: str
    answer_phrase: str = ""
    note: str = ""


#: Questions whose wording overlaps the source document — often a rare term that
#: appears almost nowhere else in the corpus. BM25's IDF makes these easy.
LEXICAL_QUERIES: list[EvalQuery] = [
    EvalQuery("Is MFA mandatory for VPN connections?",
              "information_security_policy.md", "lexical",
              "Multi-factor authentication (MFA) is mandatory",
              "MFA / VPN are near-unique tokens in this corpus"),
    EvalQuery("What is a PIP and when is one issued?",
              "performance_review_policy.md", "lexical",
              "Performance Improvement Plan (PIP)",
              "PIP appears only in the performance policy"),
    EvalQuery("What are the rules about trading while holding MNPI?",
              "code_of_conduct.md", "lexical",
              "must not trade in the company's securities",
              "MNPI is a single-document acronym"),
    EvalQuery("How much is the one-time home office setup allowance in CHF?",
              "expense_reimbursement_policy.md", "lexical",
              "up to CHF 1,500",
              "'home office setup' is verbatim in the expense policy"),
    EvalQuery("What is the AI Tools Register?",
              "ai_usage_policy.md", "lexical",
              "AI Tools Register",
              "a proper noun that occurs once"),
    EvalQuery("How long are customer records retained after the last transaction?",
              "data_retention_regulation.md", "lexical",
              "retained for ten years",
              "'customer records' + 'retained' are verbatim"),
]

#: Questions phrased the way a person actually asks, sharing little vocabulary
#: with the document. This is where lexical matching runs out of road.
SEMANTIC_QUERIES: list[EvalQuery] = [
    EvalQuery("How much time off do I get each year?",
              "leave_and_absence_policy.md", "semantic",
              "25 days of paid annual leave",
              "'time off' vs the document's 'annual leave entitlement'"),
    EvalQuery("Someone keeps making comments about my background — what do I do?",
              "code_of_conduct.md", "semantic",
              "free from harassment",
              "describes harassment without using the word"),
    EvalQuery("Can I do my job from home a few days a week?",
              "remote_work_policy.md", "semantic",
              "probation period of three months",
              "'from home' vs 'remote work' / 'other locations'"),
    EvalQuery("Someone emailed asking me to urgently wire money — is that legitimate?",
              "information_security_policy.md", "semantic",
              "verify the request through a separate channel",
              "describes phishing without naming it"),
    EvalQuery("Can I paste our customer list into ChatGPT to summarise it?",
              "ai_usage_policy.md", "semantic",
              "enterprise-tier access",
              "a concrete scenario, not the policy's vocabulary"),
    EvalQuery("A supplier offered me tickets to a football match — may I accept?",
              "code_of_conduct.md", "semantic",
              "CHF 100 in value per person",
              "'gifts and entertainment' expressed as a situation"),
]

#: The full evaluation set: lexical questions first, then semantic ones.
EVAL_SET: list[EvalQuery] = LEXICAL_QUERIES + SEMANTIC_QUERIES


def is_hit(results: Iterable[tuple[Chunk, float]], gold_source: str) -> bool:
    """Did any retrieved chunk come from the gold document?"""
    return any(chunk.metadata.get("source") == gold_source for chunk, _score in results)


def recall_at_k(
    retrieve: RetrieveFn,
    k: int = 3,
    eval_set: list[EvalQuery] | None = None,
) -> dict[str, float]:
    """Fraction of questions whose gold document appears in the top ``k``.

    Args:
        retrieve: called as ``retrieve(question)``; must already be configured to
            return at least ``k`` results (e.g. ``lambda q: rc.hybrid_search(q, top_k=k)``).
        k: how many results count as "retrieved". Only used for slicing here —
            pass a ``retrieve`` that returns at least this many.
        eval_set: defaults to ``EVAL_SET``.

    Returns:
        ``{"lexical": float, "semantic": float, "overall": float}`` — each the
        fraction of that kind's questions that hit. Kinds absent from the eval
        set are omitted.
    """
    eval_set = EVAL_SET if eval_set is None else eval_set
    hits: dict[str, list[bool]] = {}
    for query in eval_set:
        results = list(retrieve(query.question))[:k]
        hits.setdefault(query.kind, []).append(is_hit(results, query.gold_source))

    scores = {kind: sum(values) / len(values) for kind, values in hits.items()}
    every = [hit for values in hits.values() for hit in values]
    scores["overall"] = sum(every) / len(every) if every else 0.0
    return scores


def per_query_hits(
    retrieve: RetrieveFn,
    k: int = 3,
    eval_set: list[EvalQuery] | None = None,
) -> list[tuple[EvalQuery, bool, str | None]]:
    """Per-question detail behind a recall number: (query, hit, top result's source).

    A single recall figure tells you *how many* questions failed; this tells you
    *which* ones and what was retrieved instead — which is what you actually need
    to decide what to change.
    """
    eval_set = EVAL_SET if eval_set is None else eval_set
    rows = []
    for query in eval_set:
        results = list(retrieve(query.question))[:k]
        top = results[0][0].metadata.get("source") if results else None
        rows.append((query, is_hit(results, query.gold_source), top))
    return rows


def recall_curve(
    retrieve_for_k: Callable[[int], RetrieveFn],
    ks: Iterable[int] = (1, 2, 3, 5, 10, 20),
    eval_set: list[EvalQuery] | None = None,
) -> dict[int, dict[str, float]]:
    """Recall@k for several k values.

    ``retrieve_for_k(k)`` must return a retrieve function configured for that k
    (retrievers cap their own output, so k cannot just be a slice).
    """
    return {k: recall_at_k(retrieve_for_k(k), k=k, eval_set=eval_set) for k in ks}


def answer_recall_at_k(
    retrieve: RetrieveFn,
    k: int = 3,
    eval_set: list[EvalQuery] | None = None,
) -> dict[str, float]:
    """Fraction of questions whose *answer passage* is inside the retrieved text.

    Stricter than ``recall_at_k``, and the metric that can actually see chunking
    quality: retrieving the right document says nothing about whether the
    sentence that answers the question came with it. Chunks that are too small
    can split the answer across a boundary and fail this while still passing
    document recall.

    Matching is a case-insensitive substring test against the concatenated
    retrieved text. Crude, but unambiguous and free — and unlike an LLM judge it
    gives the same verdict every run. Questions with no ``answer_phrase`` are
    skipped.
    """
    eval_set = EVAL_SET if eval_set is None else eval_set
    hits: dict[str, list[bool]] = {}
    for query in eval_set:
        if not query.answer_phrase:
            continue
        context = " ".join(chunk.text for chunk, _ in list(retrieve(query.question))[:k]).lower()
        hits.setdefault(query.kind, []).append(query.answer_phrase.lower() in context)

    scores = {kind: sum(values) / len(values) for kind, values in hits.items()}
    every = [hit for values in hits.values() for hit in values]
    scores["overall"] = sum(every) / len(every) if every else 0.0
    return scores


def context_cost(
    retrieve: RetrieveFn,
    k: int = 3,
    eval_set: list[EvalQuery] | None = None,
) -> float:
    """Mean number of words handed to the LLM per question, at this ``k``.

    The price of recall. Context is not free: it costs tokens, money, latency,
    and — because models attend worse to long inputs — often accuracy too. A
    configuration that wins on recall while tripling this has not obviously won.
    """
    eval_set = EVAL_SET if eval_set is None else eval_set
    if not eval_set:
        return 0.0
    totals = [
        sum(len(chunk.text.split()) for chunk, _ in list(retrieve(query.question))[:k])
        for query in eval_set
    ]
    return sum(totals) / len(totals)


def scoreboard(
    retrieve: RetrieveFn,
    k: int = 3,
    eval_set: list[EvalQuery] | None = None,
) -> dict:
    """All three metrics for one configuration, ready to tabulate."""
    doc = recall_at_k(retrieve, k=k, eval_set=eval_set)
    ans = answer_recall_at_k(retrieve, k=k, eval_set=eval_set)
    return {
        "doc_recall": doc,
        "answer_recall": ans,
        "context_words": context_cost(retrieve, k=k, eval_set=eval_set),
    }
