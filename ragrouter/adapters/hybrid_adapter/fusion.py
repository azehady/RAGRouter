"""Reciprocal Rank Fusion (RRF) for merging multiple ranked result lists.

Merges results from multiple queries (original + expanded variants) or
multiple retrieval strategies (BM25 + dense) into a single ranked list.

Formula: score(d) = sum(1 / (k + rank_i)) for each list containing d
         + top_rank_bonus if d appears in top-3 of any list

Reference: Cormack, Clarke, Buettcher (2009) - "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods"
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()

# Configurable via env vars
RRF_K = int(os.environ.get("HYBRID_RRF_K", "60"))
RRF_TOP_RANK_BONUS = float(os.environ.get("HYBRID_RRF_TOP_RANK_BONUS", "0.1"))
RRF_TOP_RANK_THRESHOLD = int(os.environ.get("HYBRID_RRF_TOP_RANK_THRESHOLD", "3"))


@dataclass
class FusedResult:
    """A single result after RRF fusion."""

    id: str
    text: str = ""
    score: float = 0.0
    rrf_score: float = 0.0
    appearances: int = 0
    payload: dict = field(default_factory=dict)


@dataclass
class FusionStats:
    """Telemetry for the fusion step."""

    input_lists: int = 0
    total_candidates: int = 0
    unique_candidates: int = 0
    output_count: int = 0
    latency_ms: float = 0.0


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int | None = None,
    top_rank_bonus: float | None = None,
    top_rank_threshold: int | None = None,
    top_k: int = 15,
) -> tuple[list[FusedResult], FusionStats]:
    """Merge multiple ranked result lists using RRF.

    Each result dict must have an "id" key. Additional keys (text, score, payload, etc.)
    are preserved from the highest-ranked occurrence.

    Args:
        result_lists: List of ranked result lists. Each inner list is ordered by relevance
            (best first). Each result is a dict with at least {"id": str}.
        k: RRF constant (default 60). Higher values reduce the impact of rank position.
        top_rank_bonus: Bonus score for items appearing in top positions.
        top_rank_threshold: Number of top positions eligible for bonus.
        top_k: Max results to return.

    Returns:
        Tuple of (sorted FusedResult list, FusionStats).
    """
    import time

    start = time.monotonic()
    _k = k if k is not None else RRF_K
    _bonus = top_rank_bonus if top_rank_bonus is not None else RRF_TOP_RANK_BONUS
    _threshold = top_rank_threshold if top_rank_threshold is not None else RRF_TOP_RANK_THRESHOLD

    stats = FusionStats(input_lists=len(result_lists))

    if not result_lists:
        stats.latency_ms = (time.monotonic() - start) * 1000
        return [], stats

    # Accumulate RRF scores per document
    scores: dict[str, float] = {}
    appearances: dict[str, int] = {}
    best_entry: dict[str, dict] = {}  # Keep the highest-ranked dict per id

    total_candidates = 0
    for list_idx, ranked_list in enumerate(result_lists):
        for rank, item in enumerate(ranked_list, start=1):
            doc_id = item.get("id", "")
            if not doc_id:
                continue

            total_candidates += 1
            rrf_contribution = 1.0 / (_k + rank)

            # Top-rank bonus
            if rank <= _threshold:
                rrf_contribution += _bonus

            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_contribution
            appearances[doc_id] = appearances.get(doc_id, 0) + 1

            # Keep the entry from the list where it ranked highest
            if doc_id not in best_entry or rank < _best_rank(best_entry[doc_id], result_lists):
                best_entry[doc_id] = item

    stats.total_candidates = total_candidates
    stats.unique_candidates = len(scores)

    # Build sorted results
    results = []
    for doc_id, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
        entry = best_entry.get(doc_id, {})
        results.append(
            FusedResult(
                id=doc_id,
                text=entry.get("text", ""),
                score=entry.get("score", 0.0),
                rrf_score=rrf_score,
                appearances=appearances.get(doc_id, 0),
                payload={
                    k: v
                    for k, v in entry.items()
                    if k not in ("id", "text", "score")
                },
            )
        )

    stats.output_count = len(results)
    stats.latency_ms = (time.monotonic() - start) * 1000

    logger.info(
        "rrf_fusion.done",
        input_lists=stats.input_lists,
        unique=stats.unique_candidates,
        output=stats.output_count,
        latency_ms=f"{stats.latency_ms:.2f}",
    )

    return results, stats


def _best_rank(entry: dict, result_lists: list[list[dict]]) -> int:
    """Find the best (lowest) rank of an entry across all lists."""
    doc_id = entry.get("id", "")
    best = float("inf")
    for ranked_list in result_lists:
        for rank, item in enumerate(ranked_list, start=1):
            if item.get("id") == doc_id:
                best = min(best, rank)
                break
    return int(best)
