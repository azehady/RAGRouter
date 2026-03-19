"""
RAGRouter evaluation runner.

Sends each question from the eval dataset through RAGRouter (and optionally
direct engines) then scores retrieval accuracy and answer quality.

Usage:
    # Run against arbiter (port-forwarded or in-cluster)
    python eval/run_eval.py --arbiter-url http://localhost:8000

    # Run against arbiter + direct hybrid for comparison
    python eval/run_eval.py --arbiter-url http://localhost:8000 \
                            --hybrid-url http://localhost:8001

    # Limit to N questions (for quick test)
    python eval/run_eval.py --arbiter-url http://localhost:8000 --limit 10

    # Use LLM judge for answer quality scoring
    python eval/run_eval.py --arbiter-url http://localhost:8000 \
                            --judge-url http://litellm:80 --judge-model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

EVAL_DIR = Path(__file__).parent
DATA_DIR = EVAL_DIR / "data"
DATASET_FILE = DATA_DIR / "eval_dataset.jsonl"
RESULTS_DIR = EVAL_DIR / "results"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RetrievalMetrics:
    """Retrieval quality for a single question."""

    hit: bool = False  # any ground-truth chunk in retrieved set
    reciprocal_rank: float = 0.0  # 1/rank of first relevant chunk
    precision_at_k: float = 0.0  # relevant / retrieved (at k)
    recall: float = 0.0  # relevant retrieved / total relevant
    retrieved_ids: list[str] = field(default_factory=list)
    retrieved_scores: list[float] = field(default_factory=list)


@dataclass
class AnswerMetrics:
    """Answer quality for a single question."""

    faithfulness: float = 0.0  # grounded in retrieved context (0-1)
    relevance: float = 0.0  # answers the question (0-1)
    correctness: float = 0.0  # matches ground truth (0-1)
    completeness: float = 0.0  # covers all aspects (0-1)


@dataclass
class EngineMetrics:
    """Per-engine scores for a single question."""

    engine_name: str
    arbiter_score: float = 0.0
    confidence: float = 0.0
    groundedness: float = 0.0
    relevance: float = 0.0
    coverage: float = 0.0
    latency_ms: float = 0.0


@dataclass
class EvalResult:
    """Full evaluation result for a single question."""

    id: str
    question: str
    ground_truth_answer: str
    question_type: str
    difficulty: str
    doc_types: list[str]

    # Arbiter results
    arbiter_answer: str = ""
    arbiter_engine: str = ""
    arbiter_latency_ms: float = 0.0
    arbiter_citations: list[dict] = field(default_factory=list)
    arbiter_retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    arbiter_answer_metrics: AnswerMetrics = field(default_factory=AnswerMetrics)
    arbiter_engine_metrics: list[EngineMetrics] = field(default_factory=list)
    arbiter_routing_mode: str = ""
    arbiter_routing_intent: str = ""

    # Direct hybrid results (optional comparison)
    hybrid_answer: str = ""
    hybrid_latency_ms: float = 0.0
    hybrid_retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    hybrid_answer_metrics: AnswerMetrics = field(default_factory=AnswerMetrics)

    # Error tracking
    arbiter_error: str = ""
    hybrid_error: str = ""

    timestamp: str = ""


# ---------------------------------------------------------------------------
# API callers
# ---------------------------------------------------------------------------


def call_arbiter(
    url: str,
    question: str,
    corpus_id: str = "ciroos-docs",
    timeout: float = 30.0,
) -> dict:
    """Call arbiter /ask endpoint and return raw response dict."""
    resp = httpx.post(
        f"{url.rstrip('/')}/ask",
        json={
            "query": question,
            "corpus_id": corpus_id,
            "chat_history": [],
            "constraints": {"max_latency_ms": 30000, "must_cite": True, "max_tokens": 2000},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def call_hybrid_search(
    url: str,
    question: str,
    limit: int = 10,
    timeout: float = 30.0,
) -> dict:
    """Call hybrid engine /search endpoint."""
    resp = httpx.post(
        f"{url.rstrip('/')}/search",
        json={"query": question, "limit": limit, "score_threshold": 0.0},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def call_hybrid_query(
    url: str,
    question: str,
    timeout: float = 30.0,
) -> dict:
    """Call hybrid engine /query endpoint (full answer generation)."""
    resp = httpx.post(
        f"{url.rstrip('/')}/query",
        json={
            "query": question,
            "corpus_id": "ciroos-docs",
            "chat_history": [],
            "constraints": {"max_latency_ms": 15000, "must_cite": True},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Retrieval scoring
# ---------------------------------------------------------------------------


def score_retrieval(
    retrieved_ids: list[str],
    retrieved_scores: list[float],
    ground_truth_ids: list[str],
    ground_truth_contexts: list[str],
    answer_text: str,
) -> RetrievalMetrics:
    """Score retrieval quality against ground truth."""
    metrics = RetrievalMetrics(
        retrieved_ids=retrieved_ids,
        retrieved_scores=retrieved_scores,
    )

    if not ground_truth_ids and not ground_truth_contexts:
        # No ground truth chunks — use text overlap heuristic
        if answer_text and ground_truth_contexts:
            # Check if retrieved contexts overlap with ground truth
            pass
        metrics.hit = len(retrieved_ids) > 0
        metrics.reciprocal_rank = 1.0 if retrieved_ids else 0.0
        return metrics

    # Standard retrieval metrics against known chunk IDs
    gt_set = set(ground_truth_ids)
    if not gt_set:
        return metrics

    first_hit_rank = 0
    hits = 0
    for i, rid in enumerate(retrieved_ids):
        if rid in gt_set:
            hits += 1
            if first_hit_rank == 0:
                first_hit_rank = i + 1

    metrics.hit = hits > 0
    metrics.reciprocal_rank = 1.0 / first_hit_rank if first_hit_rank > 0 else 0.0
    metrics.precision_at_k = hits / len(retrieved_ids) if retrieved_ids else 0.0
    metrics.recall = hits / len(gt_set) if gt_set else 0.0

    return metrics


# ---------------------------------------------------------------------------
# LLM judge for answer quality
# ---------------------------------------------------------------------------


JUDGE_SYSTEM = """You are evaluating a RAG system's answer quality. Score each dimension 0.0-1.0.

Dimensions:
- faithfulness: Is the answer grounded in the provided context/citations? (0=hallucinated, 1=fully grounded)
- relevance: Does the answer address the question? (0=off-topic, 1=directly answers)
- correctness: Does the answer match the ground truth? (0=wrong, 1=matches)
- completeness: Does the answer cover all aspects of the ground truth? (0=misses key info, 1=comprehensive)

Return ONLY JSON: {"faithfulness": 0.X, "relevance": 0.X, "correctness": 0.X, "completeness": 0.X}"""

JUDGE_USER = """Question: {question}

Ground Truth Answer: {ground_truth}

System Answer: {answer}

Citations: {citations}

Score the system answer on faithfulness, relevance, correctness, and completeness (0.0-1.0 each)."""


def judge_answer(
    question: str,
    ground_truth: str,
    answer: str,
    citations: list[dict],
    judge_url: str,
    judge_model: str = "gpt-4o-mini",
) -> AnswerMetrics:
    """Use LLM to judge answer quality."""
    try:
        citations_text = json.dumps(citations[:5], indent=1) if citations else "None"
        resp = httpx.post(
            f"{judge_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": judge_model,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {
                        "role": "user",
                        "content": JUDGE_USER.format(
                            question=question,
                            ground_truth=ground_truth,
                            answer=answer[:1500],
                            citations=citations_text,
                        ),
                    },
                ],
                "temperature": 0.0,
                "max_tokens": 200,
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)

        scores = json.loads(content)
        return AnswerMetrics(
            faithfulness=float(scores.get("faithfulness", 0)),
            relevance=float(scores.get("relevance", 0)),
            correctness=float(scores.get("correctness", 0)),
            completeness=float(scores.get("completeness", 0)),
        )
    except Exception as e:
        print(f"    Judge error: {e}", file=sys.stderr)
        return AnswerMetrics()


# ---------------------------------------------------------------------------
# Text-overlap answer scoring (no LLM needed)
# ---------------------------------------------------------------------------


def score_answer_overlap(ground_truth: str, answer: str) -> AnswerMetrics:
    """Score answer quality using text overlap heuristics (no LLM needed)."""
    if not answer or not ground_truth:
        return AnswerMetrics()

    # Tokenize
    gt_words = set(re.findall(r"\w+", ground_truth.lower()))
    ans_words = set(re.findall(r"\w+", answer.lower()))

    # Remove stopwords
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "and", "but", "or",
        "not", "no", "so", "if", "then", "than", "that", "this", "it", "its",
    }
    gt_words -= stopwords
    ans_words -= stopwords

    if not gt_words:
        return AnswerMetrics(relevance=0.5)

    overlap = gt_words & ans_words
    precision = len(overlap) / len(ans_words) if ans_words else 0
    recall = len(overlap) / len(gt_words) if gt_words else 0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return AnswerMetrics(
        faithfulness=0.0,  # can't judge without context
        relevance=min(1.0, recall * 1.2),  # how much of ground truth is covered
        correctness=f1,
        completeness=recall,
    )


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------


def run_eval(
    arbiter_url: str,
    hybrid_url: str | None = None,
    judge_url: str | None = None,
    judge_model: str = "gpt-4o-mini",
    limit: int | None = None,
    corpus_id: str = "ciroos-docs",
) -> list[EvalResult]:
    """Run evaluation on the full dataset."""
    # Load dataset
    records = []
    with open(DATASET_FILE) as f:
        for line in f:
            records.append(json.loads(line))

    if limit:
        records = records[:limit]

    print(f"Running eval on {len(records)} questions")
    print(f"  Arbiter: {arbiter_url}")
    if hybrid_url:
        print(f"  Hybrid:  {hybrid_url}")
    if judge_url:
        print(f"  Judge:   {judge_url} ({judge_model})")
    print()

    results: list[EvalResult] = []
    now = datetime.now(timezone.utc).isoformat()

    for i, rec in enumerate(records):
        q = rec["question"]
        gt = rec["ground_truth_answer"]
        gt_ids = rec.get("chunk_ids", [])
        gt_contexts = rec.get("ground_truth_contexts", [])

        result = EvalResult(
            id=rec["id"],
            question=q,
            ground_truth_answer=gt,
            question_type=rec["question_type"],
            difficulty=rec["difficulty"],
            doc_types=rec.get("doc_types", []),
            timestamp=now,
        )

        # --- Arbiter ---
        print(f"[{i+1}/{len(records)}] {q[:80]}...", flush=True)
        try:
            start = time.monotonic()
            arbiter_resp = call_arbiter(arbiter_url, q, corpus_id=corpus_id)
            elapsed = (time.monotonic() - start) * 1000

            result.arbiter_answer = arbiter_resp.get("answer", "")
            result.arbiter_engine = arbiter_resp.get("chosen_engine", "")
            result.arbiter_latency_ms = arbiter_resp.get("total_latency_ms", elapsed)
            result.arbiter_citations = arbiter_resp.get("citations", [])

            # Routing info
            routing = arbiter_resp.get("routing_decision", {})
            result.arbiter_routing_mode = routing.get("mode", "")
            qs = routing.get("query_signals", {})
            result.arbiter_routing_intent = qs.get("intent", "")

            # Per-engine metrics
            for er in arbiter_resp.get("engine_responses", []):
                resp = er.get("response", {})
                scores = resp.get("scores", {})
                usage = resp.get("usage", {})
                result.arbiter_engine_metrics.append(
                    EngineMetrics(
                        engine_name=resp.get("engine", ""),
                        arbiter_score=er.get("arbiter_score", 0),
                        confidence=scores.get("confidence", 0),
                        groundedness=scores.get("groundedness", 0),
                        relevance=scores.get("relevance", 0),
                        coverage=scores.get("coverage", 0),
                        latency_ms=usage.get("latency_ms", 0),
                    )
                )

            # Retrieval scoring from citations
            citation_ids = [c.get("doc_id", "") for c in result.arbiter_citations]
            citation_scores = [c.get("score", 0) for c in result.arbiter_citations]
            result.arbiter_retrieval = score_retrieval(
                citation_ids, citation_scores, gt_ids, gt_contexts, result.arbiter_answer
            )

            # Answer quality
            if judge_url:
                result.arbiter_answer_metrics = judge_answer(
                    q, gt, result.arbiter_answer, result.arbiter_citations,
                    judge_url, judge_model,
                )
            else:
                result.arbiter_answer_metrics = score_answer_overlap(gt, result.arbiter_answer)

            print(f"  arbiter: engine={result.arbiter_engine} "
                  f"latency={result.arbiter_latency_ms:.0f}ms "
                  f"correctness={result.arbiter_answer_metrics.correctness:.2f}")

        except Exception as e:
            result.arbiter_error = str(e)
            print(f"  arbiter: ERROR {e}")

        # --- Direct hybrid (optional) ---
        if hybrid_url:
            try:
                start = time.monotonic()
                hybrid_resp = call_hybrid_query(hybrid_url, q)
                elapsed = (time.monotonic() - start) * 1000

                result.hybrid_answer = hybrid_resp.get("answer", "")
                result.hybrid_latency_ms = hybrid_resp.get("usage", {}).get("latency_ms", elapsed)

                # Retrieval from trace
                citations = hybrid_resp.get("citations", [])
                cit_ids = [c.get("doc_id", "") for c in citations]
                cit_scores = [c.get("score", 0) for c in citations]
                result.hybrid_retrieval = score_retrieval(
                    cit_ids, cit_scores, gt_ids, gt_contexts, result.hybrid_answer
                )

                if judge_url:
                    result.hybrid_answer_metrics = judge_answer(
                        q, gt, result.hybrid_answer, citations, judge_url, judge_model,
                    )
                else:
                    result.hybrid_answer_metrics = score_answer_overlap(gt, result.hybrid_answer)

                print(f"  hybrid:  latency={result.hybrid_latency_ms:.0f}ms "
                      f"correctness={result.hybrid_answer_metrics.correctness:.2f}")

            except Exception as e:
                result.hybrid_error = str(e)
                print(f"  hybrid: ERROR {e}")

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def compute_aggregates(results: list[EvalResult]) -> dict:
    """Compute aggregate metrics from individual results."""
    valid = [r for r in results if not r.arbiter_error]
    if not valid:
        return {"error": "no valid results"}

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    # Overall
    agg: dict = {
        "total_questions": len(results),
        "successful": len(valid),
        "errors": len(results) - len(valid),
    }

    # Arbiter retrieval
    agg["arbiter"] = {
        "retrieval": {
            "hit_rate": avg([1.0 if r.arbiter_retrieval.hit else 0.0 for r in valid]),
            "mrr": avg([r.arbiter_retrieval.reciprocal_rank for r in valid]),
            "avg_precision": avg([r.arbiter_retrieval.precision_at_k for r in valid]),
            "avg_recall": avg([r.arbiter_retrieval.recall for r in valid]),
        },
        "answer": {
            "avg_faithfulness": avg([r.arbiter_answer_metrics.faithfulness for r in valid]),
            "avg_relevance": avg([r.arbiter_answer_metrics.relevance for r in valid]),
            "avg_correctness": avg([r.arbiter_answer_metrics.correctness for r in valid]),
            "avg_completeness": avg([r.arbiter_answer_metrics.completeness for r in valid]),
        },
        "latency": {
            "avg_ms": avg([r.arbiter_latency_ms for r in valid]),
            "p50_ms": sorted([r.arbiter_latency_ms for r in valid])[len(valid) // 2],
            "p95_ms": sorted([r.arbiter_latency_ms for r in valid])[int(len(valid) * 0.95)],
            "max_ms": max(r.arbiter_latency_ms for r in valid),
        },
        "engine_distribution": {},
        "routing_intent_distribution": {},
    }

    # Engine distribution
    for r in valid:
        eng = r.arbiter_engine or "none"
        agg["arbiter"]["engine_distribution"][eng] = (
            agg["arbiter"]["engine_distribution"].get(eng, 0) + 1
        )
        intent = r.arbiter_routing_intent or "unknown"
        agg["arbiter"]["routing_intent_distribution"][intent] = (
            agg["arbiter"]["routing_intent_distribution"].get(intent, 0) + 1
        )

    # By question type
    agg["by_question_type"] = {}
    for qt in {"factual", "procedural", "conceptual", "comparison", "multi-hop"}:
        subset = [r for r in valid if r.question_type == qt]
        if subset:
            agg["by_question_type"][qt] = {
                "count": len(subset),
                "avg_correctness": avg([r.arbiter_answer_metrics.correctness for r in subset]),
                "avg_relevance": avg([r.arbiter_answer_metrics.relevance for r in subset]),
                "avg_latency_ms": avg([r.arbiter_latency_ms for r in subset]),
                "hit_rate": avg([1.0 if r.arbiter_retrieval.hit else 0.0 for r in subset]),
            }

    # By difficulty
    agg["by_difficulty"] = {}
    for diff in {"easy", "medium", "hard"}:
        subset = [r for r in valid if r.difficulty == diff]
        if subset:
            agg["by_difficulty"][diff] = {
                "count": len(subset),
                "avg_correctness": avg([r.arbiter_answer_metrics.correctness for r in subset]),
                "avg_latency_ms": avg([r.arbiter_latency_ms for r in subset]),
            }

    # Hybrid comparison (if available)
    hybrid_valid = [r for r in valid if r.hybrid_answer and not r.hybrid_error]
    if hybrid_valid:
        agg["hybrid"] = {
            "retrieval": {
                "hit_rate": avg([1.0 if r.hybrid_retrieval.hit else 0.0 for r in hybrid_valid]),
                "mrr": avg([r.hybrid_retrieval.reciprocal_rank for r in hybrid_valid]),
            },
            "answer": {
                "avg_correctness": avg(
                    [r.hybrid_answer_metrics.correctness for r in hybrid_valid]
                ),
                "avg_relevance": avg(
                    [r.hybrid_answer_metrics.relevance for r in hybrid_valid]
                ),
            },
            "latency": {
                "avg_ms": avg([r.hybrid_latency_ms for r in hybrid_valid]),
            },
        }

    return agg


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def save_results(results: list[EvalResult], aggregates: dict, run_name: str) -> Path:
    """Save eval results and aggregates to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Individual results
    results_file = run_dir / "results.jsonl"
    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r), default=str) + "\n")

    # Aggregates
    agg_file = run_dir / "aggregates.json"
    with open(agg_file, "w") as f:
        json.dump(aggregates, f, indent=2)

    # Summary (flat CSV-friendly)
    summary_file = run_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "run_name": run_name,
                "timestamp": timestamp,
                "total": aggregates.get("total_questions", 0),
                "successful": aggregates.get("successful", 0),
                **{f"arbiter_{k}": v for k, v in aggregates.get("arbiter", {}).get("retrieval", {}).items()},
                **{f"arbiter_{k}": v for k, v in aggregates.get("arbiter", {}).get("answer", {}).items()},
                **{f"arbiter_latency_{k}": v for k, v in aggregates.get("arbiter", {}).get("latency", {}).items()},
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {run_dir}/")
    print(f"  {results_file.name}: {len(results)} individual results")
    print(f"  {agg_file.name}: aggregate metrics")
    print(f"  {summary_file.name}: flat summary")

    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--arbiter-url", required=True, help="Arbiter service URL")
    parser.add_argument("--hybrid-url", default=None, help="Direct hybrid engine URL (comparison)")
    parser.add_argument("--judge-url", default=None, help="LLM judge URL for answer scoring")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="Judge model name")
    parser.add_argument("--corpus-id", default="ciroos-docs", help="Corpus ID")
    parser.add_argument("--limit", type=int, default=None, help="Limit to N questions")
    parser.add_argument("--run-name", default="eval", help="Name for this eval run")
    parser.add_argument("--dataset", default=None, help="Path to eval dataset JSONL")

    args = parser.parse_args()

    if args.dataset:
        global DATASET_FILE
        DATASET_FILE = Path(args.dataset)

    results = run_eval(
        arbiter_url=args.arbiter_url,
        hybrid_url=args.hybrid_url,
        judge_url=args.judge_url,
        judge_model=args.judge_model,
        limit=args.limit,
        corpus_id=args.corpus_id,
    )

    aggregates = compute_aggregates(results)

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(json.dumps(aggregates, indent=2))

    save_results(results, aggregates, args.run_name)


if __name__ == "__main__":
    main()
