"""Tests for evaluation.py — the retrieval metrics, grouped by function.

Run it:

    uv run python -m tests.test_evaluation
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluation as ev
from models import Chunk
from tests.harness import TestSuite

suite = TestSuite("Evaluation — Test Results")


def _chunk(source: str, text: str = "some text") -> Chunk:
    return Chunk(document_id="d", index=0, text=text, metadata={"source": source})


_GOLD = _chunk("gold.md", "the answer is 25 days of paid annual leave per year")
_OTHER = _chunk("other.md", "something entirely unrelated about firewalls")

_ONE = [ev.EvalQuery("q?", "gold.md", "lexical", "25 days of paid annual leave")]
_TWO_KINDS = [
    ev.EvalQuery("lex?", "gold.md", "lexical", "25 days"),
    ev.EvalQuery("sem?", "gold.md", "semantic", "25 days"),
]


# --------------------------------------------------------------------------- #
# EVAL_SET
# --------------------------------------------------------------------------- #
@suite.case("EVAL_SET", "every gold_source names a file that exists in data/")
def _():
    available = {path.name for path in Path("data").iterdir() if path.is_file()}
    missing = {q.gold_source for q in ev.EVAL_SET} - available
    assert not missing, f"eval set points at documents that aren't in data/: {sorted(missing)}"


@suite.case("EVAL_SET", "every answer_phrase really appears in its gold document")
def _():
    for query in ev.EVAL_SET:
        if not query.answer_phrase:
            continue
        text = Path("data", query.gold_source).read_text(encoding="utf-8").lower()
        assert query.answer_phrase.lower() in text, (
            f"'{query.answer_phrase}' is not in {query.gold_source} — the gold answer is wrong, "
            "so every score computed against it is meaningless"
        )


@suite.case("EVAL_SET", "both kinds are represented")
def _():
    kinds = {q.kind for q in ev.EVAL_SET}
    assert kinds == {"lexical", "semantic"}, f"expected both kinds, got {kinds}"


# --------------------------------------------------------------------------- #
# is_hit
# --------------------------------------------------------------------------- #
@suite.case("is_hit", "true when any result comes from the gold document")
def _():
    assert ev.is_hit([(_OTHER, 1.0), (_GOLD, 0.1)], "gold.md")


@suite.case("is_hit", "false when the gold document is absent")
def _():
    assert not ev.is_hit([(_OTHER, 1.0)], "gold.md")


@suite.case("is_hit", "false on empty results")
def _():
    assert not ev.is_hit([], "gold.md")


# --------------------------------------------------------------------------- #
# recall_at_k
# --------------------------------------------------------------------------- #
@suite.case("recall_at_k", "perfect retrieval scores 1.0")
def _():
    scores = ev.recall_at_k(lambda q: [(_GOLD, 1.0)], k=3, eval_set=_ONE)
    assert scores["overall"] == 1.0, scores


@suite.case("recall_at_k", "retrieving only the wrong document scores 0.0")
def _():
    scores = ev.recall_at_k(lambda q: [(_OTHER, 1.0)], k=3, eval_set=_ONE)
    assert scores["overall"] == 0.0, scores


@suite.case("recall_at_k", "k truncates: a hit below the cut does not count")
def _():
    retrieve = lambda q: [(_OTHER, 1.0), (_OTHER, 0.9), (_GOLD, 0.8)]
    assert ev.recall_at_k(retrieve, k=2, eval_set=_ONE)["overall"] == 0.0
    assert ev.recall_at_k(retrieve, k=3, eval_set=_ONE)["overall"] == 1.0


@suite.case("recall_at_k", "reports per kind as well as overall")
def _():
    scores = ev.recall_at_k(lambda q: [(_GOLD, 1.0)], k=1, eval_set=_TWO_KINDS)
    assert set(scores) == {"lexical", "semantic", "overall"}, scores


# --------------------------------------------------------------------------- #
# answer_recall_at_k
# --------------------------------------------------------------------------- #
@suite.case("answer_recall_at_k", "hit when the retrieved text contains the answer phrase")
def _():
    assert ev.answer_recall_at_k(lambda q: [(_GOLD, 1.0)], k=1, eval_set=_ONE)["overall"] == 1.0


@suite.case("answer_recall_at_k", "right document, wrong passage still counts as a miss")
def _():
    stub = _chunk("gold.md", "an unrelated paragraph of the same document")
    assert ev.recall_at_k(lambda q: [(stub, 1.0)], k=1, eval_set=_ONE)["overall"] == 1.0
    assert ev.answer_recall_at_k(lambda q: [(stub, 1.0)], k=1, eval_set=_ONE)["overall"] == 0.0


@suite.case("answer_recall_at_k", "queries with no answer_phrase are skipped")
def _():
    no_phrase = [ev.EvalQuery("q?", "gold.md", "lexical")]
    assert ev.answer_recall_at_k(lambda q: [], k=1, eval_set=no_phrase)["overall"] == 0.0


# --------------------------------------------------------------------------- #
# context_cost
# --------------------------------------------------------------------------- #
@suite.case("context_cost", "counts the words actually handed to the LLM")
def _():
    five = _chunk("gold.md", "one two three four five")
    cost = ev.context_cost(lambda q: [(five, 1.0), (five, 0.5)], k=2, eval_set=_ONE)
    assert cost == 10.0, f"two 5-word chunks should cost 10 words, got {cost}"


@suite.case("context_cost", "respects k")
def _():
    five = _chunk("gold.md", "one two three four five")
    assert ev.context_cost(lambda q: [(five, 1.0), (five, 0.5)], k=1, eval_set=_ONE) == 5.0


# --------------------------------------------------------------------------- #
# scoreboard
# --------------------------------------------------------------------------- #
@suite.case("scoreboard", "bundles all three metrics")
def _():
    board = ev.scoreboard(lambda q: [(_GOLD, 1.0)], k=1, eval_set=_ONE)
    assert set(board) == {"doc_recall", "answer_recall", "context_words"}, board
    assert board["doc_recall"]["overall"] == 1.0 and board["answer_recall"]["overall"] == 1.0


if __name__ == "__main__":
    raise SystemExit(0 if suite.run() else 1)
