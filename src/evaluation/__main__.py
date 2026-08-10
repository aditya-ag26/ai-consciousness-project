"""
Command-line entry point for the evaluation harness.

    python -m src.evaluation                  # evaluate the configured provider
    LLM_PROVIDER=ollama python -m src.evaluation
    python -m src.evaluation --latency-samples 0   # skip generation timing
"""
import argparse
import json
import logging
import os
from pathlib import Path

from src.config import config
from src.evaluation.dataset import load_dataset
from src.evaluation.metrics import format_table
from src.evaluation.runner import (
    collect_scores,
    evaluate_guardrail,
    evaluate_retrieval,
    group_accuracy,
    measure_latency,
    sweep_thresholds,
)
from src.rag_pipeline.bot import QueryBot

SWEEP_THRESHOLDS = [0.85, 0.95, 1.00, 1.02, 1.05, 1.08, 1.10, 1.15, 1.25, 1.40]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument(
        "--latency-samples",
        type=int,
        default=5,
        help="Number of in-scope questions to time end to end (0 to skip).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write the report as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # The pipeline's own logs would drown out the report.
    logging.getLogger("src.rag_pipeline.bot").setLevel(logging.WARNING)

    args = parse_args()
    provider = os.getenv("LLM_PROVIDER", config.rag_application.llm.provider)
    threshold = config.rag_application.relevance_threshold

    dataset = load_dataset()
    bot = QueryBot(config.rag_application)

    groups = collect_scores(bot, dataset)
    retrieval = evaluate_retrieval(groups["in_scope"])
    guardrail = evaluate_guardrail(groups, threshold)
    sweep = sweep_thresholds(groups, SWEEP_THRESHOLDS)

    latency = None
    if args.latency_samples > 0:
        questions = [case.question for case in dataset.in_scope[: args.latency_samples]]
        latency = measure_latency(
            bot, questions, config.rag_application.answer_length_map["short"]
        )

    report = _render(
        provider, threshold, dataset, groups, retrieval, guardrail, sweep, latency
    )
    print(report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                _as_dict(provider, threshold, retrieval, guardrail, sweep, latency),
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nJSON report written to {args.json}")


def _render(provider, threshold, dataset, groups, retrieval, guardrail, sweep, latency) -> str:
    embedding = config.rag_application.embedding_model
    model = (
        config.rag_application.llm.gemini.model_name
        if provider == "gemini"
        else config.rag_application.llm.ollama.model_name
    )

    lines = [
        "",
        "=" * 68,
        "  RAG PIPELINE EVALUATION",
        "=" * 68,
        f"  LLM provider      : {provider} ({model})",
        f"  Embedding model   : {embedding}",
        f"  Relevance cutoff  : {threshold}",
        f"  Labelled cases    : {dataset.total_cases}",
        "",
        "-" * 68,
        "  1. RETRIEVAL QUALITY   (does the right document come back?)",
        "-" * 68,
        f"  Hit rate @{config.rag_application.retrieval_k}        : {retrieval.hit_rate:.1%}  "
        f"({round(retrieval.hit_rate * retrieval.total)}/{retrieval.total} questions)",
        f"  Mean reciprocal rank: {retrieval.mean_reciprocal_rank:.3f}",
        f"  Context per question: {retrieval.mean_context_chars:,.0f} chars",
        f"  Thin chunks (<200ch): {retrieval.thin_chunk_rate:.1%}"
        + ("   <-- retrieval is returning headings, not content" if retrieval.thin_chunk_rate > 0.1 else ""),
    ]
    if retrieval.misses:
        lines.append(f"  Missed ({len(retrieval.misses)}):")
        lines.extend(f"    - {question}" for question in retrieval.misses)

    lines += [
        "",
        "-" * 68,
        "  2. GUARDRAIL ACCURACY  (answer in scope, decline everything else)",
        "-" * 68,
        f"  Accuracy  : {guardrail.accuracy:.1%}",
        f"  Precision : {guardrail.precision:.1%}   (answers that should have been given)",
        f"  Recall    : {guardrail.recall:.1%}   (in-scope questions actually answered)",
        f"  F1        : {guardrail.f1:.3f}",
        "",
        f"  Confusion : TP={guardrail.true_positives}  FP={guardrail.false_positives}  "
        f"TN={guardrail.true_negatives}  FN={guardrail.false_negatives}",
        "",
        "  By category:",
        f"    In-scope questions           : {group_accuracy(groups['in_scope'], threshold)} answered",
        f"    Out-of-scope questions       : {group_accuracy(groups['out_of_scope'], threshold)} declined",
        f"    Adversarial near-miss topics : "
        f"{group_accuracy(groups['adversarial_out_of_scope'], threshold)} declined",
        f"    Context-dependent follow-ups : {group_accuracy(groups['follow_ups'], threshold)} answered",
        f"    Off-topic mid-conversation   : {group_accuracy(groups['off_topic_follow_ups'], threshold)} declined",
        "",
        "-" * 68,
        "  3. THRESHOLD SENSITIVITY  (why the cutoff sits where it does)",
        "-" * 68,
    ]

    rows = [
        [
            f"{value:.2f}" + ("  <-- configured" if abs(value - threshold) < 1e-9 else ""),
            f"{accuracy:.1%}",
            f"{precision:.1%}",
            f"{recall:.1%}",
            f"{f1:.3f}",
        ]
        for value, accuracy, precision, recall, f1 in sweep
    ]
    lines.append(
        format_table(["Threshold", "Accuracy", "Precision", "Recall", "F1"], rows)
    )

    if latency:
        lines += [
            "",
            "-" * 68,
            "  4. LATENCY  (end to end, retrieval plus generation)",
            "-" * 68,
            f"  Samples : {len(latency.samples)}",
            f"  Mean    : {latency.mean:.2f}s",
            f"  p50     : {latency.p50:.2f}s",
            f"  p95     : {latency.p95:.2f}s",
        ]

    lines += ["", "=" * 68, ""]
    return "\n".join(lines)


def _as_dict(provider, threshold, retrieval, guardrail, sweep, latency) -> dict:
    return {
        "provider": provider,
        "relevance_threshold": threshold,
        "retrieval": {
            "hit_rate": retrieval.hit_rate,
            "mean_reciprocal_rank": retrieval.mean_reciprocal_rank,
            "misses": retrieval.misses,
            "total": retrieval.total,
            "mean_context_chars": retrieval.mean_context_chars,
            "thin_chunk_rate": retrieval.thin_chunk_rate,
        },
        "guardrail": {
            "accuracy": guardrail.accuracy,
            "precision": guardrail.precision,
            "recall": guardrail.recall,
            "f1": guardrail.f1,
            "true_positives": guardrail.true_positives,
            "false_positives": guardrail.false_positives,
            "true_negatives": guardrail.true_negatives,
            "false_negatives": guardrail.false_negatives,
        },
        "threshold_sweep": [
            {
                "threshold": t,
                "accuracy": a,
                "precision": p,
                "recall": r,
                "f1": f,
            }
            for t, a, p, r, f in sweep
        ],
        "latency_seconds": (
            {
                "mean": latency.mean,
                "p50": latency.p50,
                "p95": latency.p95,
                "samples": latency.samples,
            }
            if latency
            else None
        ),
    }


if __name__ == "__main__":
    main()
