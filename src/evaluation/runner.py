"""
Evaluation harness for the RAG pipeline.

Measures three things the architecture claims to do:

1. Retrieval - does the corpus surface the right documents for a question?
2. Guardrail - does it answer in-scope questions and decline everything else?
3. Generation - how long an answered question takes end to end.

Retrieval and guardrail scoring need only the embedding model, so they are
deterministic and cost nothing per run. The language model is used only to
condense context-dependent follow-ups and to time generation, which keeps the
evaluation cheap enough to run against a metered hosted API.
"""
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

from src.evaluation.dataset import ConversationCase, EvalDataset
from src.evaluation.metrics import (
    ClassificationMetrics,
    classify,
    is_hit,
    mean,
    percentile,
    reciprocal_rank,
)
from src.rag_pipeline.bot import QueryBot

logger = logging.getLogger(__name__)


@dataclass
class ScoredCase:
    """A question with the retrieval distances that drive the scope decision."""
    question: str
    should_answer: bool
    bare_distance: float
    condensed_distance: float | None = None
    retrieved_titles: Sequence[str] = ()
    retrieved_lengths: Sequence[int] = ()
    expected_sources: Sequence[str] = ()

    def answered_at(self, threshold: float) -> bool:
        """Mirrors QueryBot.ask: a follow-up gets a second chance once condensed."""
        if self.bare_distance <= threshold:
            return True
        if self.condensed_distance is not None:
            return self.condensed_distance <= threshold
        return False


@dataclass
class RetrievalReport:
    hit_rate: float
    mean_reciprocal_rank: float
    misses: list[str]
    total: int
    # Hit rate alone says only that the right document was found. It is
    # satisfied by a chunk containing nothing but that document's title, which
    # is exactly the failure these two catch: retrieval looking perfect while
    # handing the model no material to answer from.
    mean_context_chars: float = 0.0
    thin_chunk_rate: float = 0.0


# A chunk shorter than this carries a heading at best, not an argument.
THIN_CHUNK_CHARS = 200


@dataclass
class LatencyReport:
    samples: list[float]

    @property
    def mean(self) -> float:
        return mean(self.samples)

    @property
    def p50(self) -> float:
        return percentile(self.samples, 0.50)

    @property
    def p95(self) -> float:
        return percentile(self.samples, 0.95)


@dataclass
class EvalReport:
    provider: str
    retrieval: RetrievalReport
    guardrail: ClassificationMetrics
    guardrail_breakdown: dict[str, str]
    threshold_sweep: list[tuple]
    latency: LatencyReport
    configured_threshold: float


def _min_distance(bot: QueryBot, query: str) -> tuple:
    """Returns the best distance, the titles retrieved, and their content lengths."""
    scored = bot.retrieve(query)
    if not scored:
        return float("inf"), [], []
    titles = [doc.metadata.get("title", "") for doc, _ in scored]
    lengths = [len(doc.page_content) for doc, _ in scored]
    return min(score for _, score in scored), titles, lengths


def _score_conversation(bot: QueryBot, case: ConversationCase, should_answer: bool) -> ScoredCase:
    """Scores a follow-up both as written and after condensation against history."""
    bare_distance, titles, lengths = _min_distance(bot, case.question)
    history = [(role, content) for role, content in case.history]
    condensed_query = bot.condense(case.question, history)
    condensed_distance, _, _ = _min_distance(bot, condensed_query)
    return ScoredCase(
        question=case.question,
        should_answer=should_answer,
        bare_distance=bare_distance,
        condensed_distance=condensed_distance,
        retrieved_titles=titles,
        retrieved_lengths=lengths,
    )


def collect_scores(bot: QueryBot, dataset: EvalDataset) -> dict[str, list[ScoredCase]]:
    """Runs retrieval once per case so thresholds can be swept without re-embedding."""
    groups: dict[str, list[ScoredCase]] = {}

    logger.info("Scoring in-scope questions...")
    groups["in_scope"] = []
    for case in dataset.in_scope:
        distance, titles, lengths = _min_distance(bot, case.question)
        groups["in_scope"].append(
            ScoredCase(
                question=case.question,
                should_answer=True,
                bare_distance=distance,
                retrieved_titles=titles,
                retrieved_lengths=lengths,
                expected_sources=case.expected_sources,
            )
        )

    for group_name, questions in (
        ("out_of_scope", dataset.out_of_scope),
        ("adversarial_out_of_scope", dataset.adversarial_out_of_scope),
    ):
        logger.info(f"Scoring {group_name.replace('_', ' ')} questions...")
        groups[group_name] = []
        for question in questions:
            distance, titles, lengths = _min_distance(bot, question)
            groups[group_name].append(
                ScoredCase(
                    question=question,
                    should_answer=False,
                    bare_distance=distance,
                    retrieved_titles=titles,
                    retrieved_lengths=lengths,
                )
            )

    logger.info("Scoring context-dependent follow-ups (uses the LLM to condense)...")
    groups["follow_ups"] = [
        _score_conversation(bot, case, should_answer=True) for case in dataset.follow_ups
    ]

    logger.info("Scoring off-topic follow-ups...")
    groups["off_topic_follow_ups"] = [
        _score_conversation(bot, case, should_answer=False)
        for case in dataset.off_topic_follow_ups
    ]

    return groups


def evaluate_retrieval(scored_in_scope: Sequence[ScoredCase]) -> RetrievalReport:
    """Hit rate and MRR over questions that have labelled expected sources."""
    hits, ranks, misses = [], [], []
    context_sizes, every_chunk = [], []
    for case in scored_in_scope:
        hit = is_hit(case.retrieved_titles, case.expected_sources)
        hits.append(1.0 if hit else 0.0)
        ranks.append(reciprocal_rank(case.retrieved_titles, case.expected_sources))
        if not hit:
            misses.append(case.question)
        context_sizes.append(float(sum(case.retrieved_lengths)))
        every_chunk.extend(case.retrieved_lengths)

    thin = [1.0 if length < THIN_CHUNK_CHARS else 0.0 for length in every_chunk]

    return RetrievalReport(
        hit_rate=mean(hits),
        mean_reciprocal_rank=mean(ranks),
        misses=misses,
        total=len(scored_in_scope),
        mean_context_chars=mean(context_sizes),
        thin_chunk_rate=mean(thin),
    )


def evaluate_guardrail(
    groups: dict[str, list[ScoredCase]], threshold: float
) -> ClassificationMetrics:
    """Confusion matrix of the answer/decline decision at a given threshold."""
    every_case = [case for group in groups.values() for case in group]
    return classify(
        answered=[case.answered_at(threshold) for case in every_case],
        should_answer=[case.should_answer for case in every_case],
    )


def sweep_thresholds(
    groups: dict[str, list[ScoredCase]], thresholds: Sequence[float]
) -> list[tuple]:
    """Recomputes guardrail quality across candidate thresholds, without re-embedding."""
    sweep = []
    for threshold in thresholds:
        metrics = evaluate_guardrail(groups, threshold)
        sweep.append((threshold, metrics.accuracy, metrics.precision, metrics.recall, metrics.f1))
    return sweep


def group_accuracy(group: Sequence[ScoredCase], threshold: float) -> str:
    """Correct-count for one labelled group, e.g. '18/18'."""
    correct = sum(
        1 for case in group if case.answered_at(threshold) == case.should_answer
    )
    return f"{correct}/{len(group)}"


def measure_latency(
    bot: QueryBot, questions: Sequence[str], num_predict_tokens: int
) -> LatencyReport:
    """Times full answered requests, including retrieval and generation."""
    samples = []
    for question in questions:
        started = time.perf_counter()
        bot.ask(question, num_predict_tokens=num_predict_tokens)
        samples.append(time.perf_counter() - started)
    return LatencyReport(samples=samples)
