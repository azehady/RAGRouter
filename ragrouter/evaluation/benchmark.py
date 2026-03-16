"""Benchmark script: runs eval dataset through arbiter, compares engine results."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

from ragrouter.schemas import ArbiterResult, EngineRequest


async def run_benchmark(
    eval_path: str,
    arbiter_url: str = "http://localhost:8000",
    corpus_id: str = "ciroos-docs",
    output_path: str | None = None,
) -> list[dict]:
    """Run all queries in the eval dataset through the arbiter."""
    eval_data = json.loads(Path(eval_path).read_text())
    questions = eval_data if isinstance(eval_data, list) else eval_data.get("questions", eval_data)

    results = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, item in enumerate(questions):
            query = item.get("question", item.get("query", ""))
            expected = item.get("expected_answer", item.get("answer", ""))
            expected_docs = item.get("expected_sources", item.get("sources", []))

            print(f"[{i + 1}/{len(questions)}] {query[:80]}...")

            start = time.monotonic()
            try:
                resp = await client.post(
                    f"{arbiter_url}/ask",
                    json=EngineRequest(query=query, corpus_id=corpus_id).model_dump(),
                )
                resp.raise_for_status()
                arbiter_result = ArbiterResult(**resp.json())
                elapsed = (time.monotonic() - start) * 1000

                result = {
                    "query": query,
                    "expected_answer": expected,
                    "expected_sources": expected_docs,
                    "chosen_engine": arbiter_result.chosen_engine,
                    "answer": arbiter_result.answer,
                    "citations": [c.model_dump() for c in arbiter_result.citations],
                    "total_latency_ms": elapsed,
                    "engine_scores": [
                        {
                            "engine": sr.response.engine,
                            "arbiter_score": sr.arbiter_score,
                            "chosen": sr.chosen,
                            "confidence": sr.response.scores.confidence,
                            "groundedness": sr.response.scores.groundedness,
                            "latency_ms": sr.response.usage.latency_ms,
                        }
                        for sr in arbiter_result.engine_responses
                    ],
                    "routing": {
                        "intent": arbiter_result.routing_decision.query_signals.intent.value,
                        "mode": arbiter_result.routing_decision.mode.value,
                        "candidates": [
                            c.engine.value for c in arbiter_result.routing_decision.candidates
                        ],
                    },
                }
                results.append(result)
                print(
                    f"  -> {arbiter_result.chosen_engine} "
                    f"(score={arbiter_result.engine_responses[0].arbiter_score:.3f}, "
                    f"{elapsed:.0f}ms)"
                )
            except Exception as e:
                print(f"  -> ERROR: {e}")
                results.append({"query": query, "error": str(e)})

    # Summary
    total = len(results)
    successful = [r for r in results if "error" not in r]
    engines_chosen = {}
    for r in successful:
        eng = r["chosen_engine"]
        engines_chosen[eng] = engines_chosen.get(eng, 0) + 1

    avg_latency = sum(r["total_latency_ms"] for r in successful) / len(successful) if successful else 0
    avg_score = (
        sum(r["engine_scores"][0]["arbiter_score"] for r in successful if r["engine_scores"])
        / len(successful)
        if successful
        else 0
    )

    summary = {
        "total_queries": total,
        "successful": len(successful),
        "failed": total - len(successful),
        "avg_latency_ms": round(avg_latency, 1),
        "avg_arbiter_score": round(avg_score, 3),
        "engine_distribution": engines_chosen,
    }

    print("\n--- Benchmark Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    output = {"summary": summary, "results": results}

    if output_path:
        Path(output_path).write_text(json.dumps(output, indent=2))
        print(f"\nResults written to {output_path}")

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ragrouter.evaluation.benchmark <eval_dataset.json> [arbiter_url]")
        sys.exit(1)

    eval_path = sys.argv[1]
    arbiter_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    asyncio.run(run_benchmark(eval_path, arbiter_url, output_path=output_path))


if __name__ == "__main__":
    main()
