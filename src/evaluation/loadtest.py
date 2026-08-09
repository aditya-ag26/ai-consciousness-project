"""
Load test for the deployed retrieval path.

    python -m src.evaluation.loadtest --url https://... --concurrency 50 --requests 500

Sends out-of-scope questions to `/ask`. Those are declined by the relevance
guardrail *before* the language model is called, so every request still
exercises the full request path - HTTP, middleware, query embedding, FAISS
search, and response serialisation - without consuming model quota. That makes
it possible to measure the service under real concurrency for free.

What this does NOT measure is generation latency, which is dominated by the
language model and has to be reported separately from a small sample. Quoting
these numbers as end-to-end chat latency would overstate them.
"""
import argparse
import asyncio
import time
from dataclasses import dataclass, field

import httpx

from src.evaluation.metrics import mean, percentile

# Deliberately out of scope: refused by retrieval, so no model call is made.
PROBE = "How do I cook pasta?"


@dataclass
class Results:
    latencies: list[float] = field(default_factory=list)
    statuses: dict[int, int] = field(default_factory=dict)
    failures: int = 0
    wall_time: float = 0.0

    @property
    def total(self) -> int:
        return len(self.latencies) + self.failures

    @property
    def throughput(self) -> float:
        return self.total / self.wall_time if self.wall_time else 0.0

    @property
    def error_rate(self) -> float:
        ok = self.statuses.get(200, 0)
        return 1.0 - (ok / self.total) if self.total else 0.0


async def _worker(
    client: httpx.AsyncClient, url: str, queue: asyncio.Queue, results: Results
) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        started = time.perf_counter()
        try:
            response = await client.post(
                f"{url}/ask", json={"query": PROBE, "length": "short"}
            )
            results.latencies.append(time.perf_counter() - started)
            results.statuses[response.status_code] = (
                results.statuses.get(response.status_code, 0) + 1
            )
        except Exception:
            results.failures += 1
        finally:
            queue.task_done()


async def run(url: str, concurrency: int, total: int, timeout: float) -> Results:
    queue: asyncio.Queue = asyncio.Queue()
    for _ in range(total):
        queue.put_nowait(None)

    results = Results()
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        started = time.perf_counter()
        await asyncio.gather(
            *(_worker(client, url, queue, results) for _ in range(concurrency))
        )
        results.wall_time = time.perf_counter() - started

    return results


def report(results: Results, concurrency: int) -> str:
    lines = [
        "",
        "=" * 60,
        "  RETRIEVAL PATH LOAD TEST",
        "=" * 60,
        f"  Concurrency      : {concurrency}",
        f"  Requests sent    : {results.total}",
        f"  Wall time        : {results.wall_time:.2f}s",
        f"  Throughput       : {results.throughput:.1f} req/s",
        f"  Error rate       : {results.error_rate:.2%}",
        "",
        "  Latency (seconds)",
        f"    mean : {mean(results.latencies):.3f}",
        f"    p50  : {percentile(results.latencies, 0.50):.3f}",
        f"    p95  : {percentile(results.latencies, 0.95):.3f}",
        f"    p99  : {percentile(results.latencies, 0.99):.3f}",
        f"    max  : {max(results.latencies):.3f}" if results.latencies else "",
        "",
        f"  Status codes     : {dict(sorted(results.statuses.items()))}",
        f"  Transport errors : {results.failures}",
        "=" * 60,
        "",
    ]
    return "\n".join(line for line in lines if line != "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Base URL of the deployed API")
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument("--requests", type=int, default=250)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    results = asyncio.run(
        run(args.url.rstrip("/"), args.concurrency, args.requests, args.timeout)
    )
    print(report(results, args.concurrency))


if __name__ == "__main__":
    main()
