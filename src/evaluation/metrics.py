"""
Metric calculations for the RAG evaluation.

Everything here is a pure function over already-collected results, so the
numbers can be unit-tested without loading a model or an index.
"""
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationMetrics:
    """
    Quality of the in-scope / out-of-scope decision.

    The positive class is "answered". Precision therefore reads as "of the
    questions it chose to answer, how many should it have answered", and recall
    as "of the questions it should have answered, how many did it".
    """
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def total(self) -> int:
        return (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
        )

    @property
    def accuracy(self) -> float:
        return _ratio(self.true_positives + self.true_negatives, self.total)

    @property
    def precision(self) -> float:
        return _ratio(self.true_positives, self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        return _ratio(self.true_positives, self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        if denominator == 0:
            return 0.0
        return 2 * self.precision * self.recall / denominator


def classify(answered: Sequence[bool], should_answer: Sequence[bool]) -> ClassificationMetrics:
    """Builds a confusion matrix from paired decisions and ground truth."""
    if len(answered) != len(should_answer):
        raise ValueError("answered and should_answer must be the same length")

    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for did_answer, expected in zip(answered, should_answer):
        if expected and did_answer:
            counts["tp"] += 1
        elif expected and not did_answer:
            counts["fn"] += 1
        elif not expected and did_answer:
            counts["fp"] += 1
        else:
            counts["tn"] += 1

    return ClassificationMetrics(
        true_positives=counts["tp"],
        false_positives=counts["fp"],
        true_negatives=counts["tn"],
        false_negatives=counts["fn"],
    )


def is_hit(retrieved_titles: Sequence[str], expected_sources: Sequence[str]) -> bool:
    """True when any expected source appears in any retrieved title."""
    haystack = " || ".join(title.lower() for title in retrieved_titles)
    return any(expected.lower() in haystack for expected in expected_sources)


def reciprocal_rank(retrieved_titles: Sequence[str], expected_sources: Sequence[str]) -> float:
    """
    1/rank of the first relevant document, or 0.0 if none is relevant.

    Averaged over a question set this is Mean Reciprocal Rank, which rewards
    putting the right document near the top rather than merely including it.
    """
    for position, title in enumerate(retrieved_titles, start=1):
        if is_hit([title], expected_sources):
            return 1.0 / position
    return 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], fraction: float) -> float:
    """
    Nearest-rank percentile, e.g. fraction=0.95 for p95.

    Nearest-rank avoids interpolating between latency samples, so every reported
    figure is a value that was actually observed.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(fraction * len(ordered) + 0.5) - 1))
    return ordered[index]


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Renders a fixed-width table for terminal and Markdown-friendly output."""
    widths = [
        max(len(str(headers[column])), *(len(str(row[column])) for row in rows))
        if rows
        else len(str(headers[column]))
        for column in range(len(headers))
    ]
    divider = "-+-".join("-" * width for width in widths)
    lines: list[str] = [
        " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))),
        divider,
    ]
    lines.extend(
        " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) for row in rows
    )
    return "\n".join(lines)
