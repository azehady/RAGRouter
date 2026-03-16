"""Tests for RRF fusion module."""

from __future__ import annotations

import pytest

from ragrouter.adapters.hybrid_adapter.fusion import (
    FusedResult,
    FusionStats,
    reciprocal_rank_fusion,
)


class TestReciprocalRankFusion:
    def test_empty_input(self):
        results, stats = reciprocal_rank_fusion([])
        assert results == []
        assert stats.input_lists == 0
        assert stats.output_count == 0

    def test_single_list(self):
        results, stats = reciprocal_rank_fusion([
            [
                {"id": "doc-1", "text": "first", "score": 0.9},
                {"id": "doc-2", "text": "second", "score": 0.8},
            ]
        ])
        assert len(results) == 2
        assert results[0].id == "doc-1"
        assert results[0].rrf_score > results[1].rrf_score
        assert stats.input_lists == 1
        assert stats.unique_candidates == 2

    def test_two_lists_shared_doc_ranked_higher(self):
        """A doc appearing in both lists should score higher than one in only one."""
        list_a = [
            {"id": "shared", "text": "shared doc", "score": 0.9},
            {"id": "a-only", "text": "only in A", "score": 0.8},
        ]
        list_b = [
            {"id": "shared", "text": "shared doc", "score": 0.85},
            {"id": "b-only", "text": "only in B", "score": 0.7},
        ]

        results, stats = reciprocal_rank_fusion([list_a, list_b])

        assert results[0].id == "shared"
        assert results[0].appearances == 2
        assert stats.unique_candidates == 3
        # shared should have higher RRF score than single-list docs
        assert results[0].rrf_score > results[1].rrf_score

    def test_top_k_limits_output(self):
        big_list = [{"id": f"doc-{i}", "text": f"text {i}", "score": 1.0 - i * 0.01} for i in range(20)]

        results, stats = reciprocal_rank_fusion([big_list], top_k=5)

        assert len(results) == 5
        assert stats.output_count == 5
        assert stats.unique_candidates == 20

    def test_rrf_formula_correctness(self):
        """Verify RRF score matches the formula: 1/(k + rank)."""
        k = 60
        results, _ = reciprocal_rank_fusion(
            [[{"id": "doc-1", "text": "t", "score": 0.5}]],
            k=k,
        )
        # rank=1, so score = 1/(60+1) = 0.01639... + top_rank_bonus 0.1
        expected = 1.0 / (k + 1) + 0.1  # top-3 bonus
        assert abs(results[0].rrf_score - expected) < 1e-6

    def test_no_top_rank_bonus_beyond_threshold(self):
        """Doc at rank 4 should NOT get top-rank bonus (threshold=3)."""
        items = [{"id": f"doc-{i}", "text": f"t{i}", "score": 0.5} for i in range(5)]

        results, _ = reciprocal_rank_fusion([items], k=60, top_rank_bonus=0.1, top_rank_threshold=3)

        # doc-0 (rank 1): 1/61 + 0.1
        # doc-3 (rank 4): 1/64 + 0 (no bonus)
        doc_0 = next(r for r in results if r.id == "doc-0")
        doc_3 = next(r for r in results if r.id == "doc-3")
        assert doc_0.rrf_score == pytest.approx(1.0 / 61 + 0.1, rel=1e-5)
        assert doc_3.rrf_score == pytest.approx(1.0 / 64, rel=1e-5)

    def test_multiple_query_variants_fusion(self):
        """Simulate 4 query variants (original + 3 expanded) being fused."""
        original_results = [
            {"id": "kb-001", "text": "reset password", "score": 0.92},
            {"id": "kb-002", "text": "password policy", "score": 0.88},
            {"id": "kb-003", "text": "MFA setup", "score": 0.85},
        ]
        lexical_results = [
            {"id": "kb-001", "text": "reset password", "score": 0.90},
            {"id": "kb-004", "text": "account lockout", "score": 0.82},
        ]
        semantic_results = [
            {"id": "kb-002", "text": "password policy", "score": 0.87},
            {"id": "kb-005", "text": "SSO login issues", "score": 0.80},
        ]
        hypothetical_results = [
            {"id": "kb-001", "text": "reset password", "score": 0.88},
            {"id": "kb-003", "text": "MFA setup", "score": 0.83},
            {"id": "kb-006", "text": "new employee onboarding", "score": 0.75},
        ]

        results, stats = reciprocal_rank_fusion(
            [original_results, lexical_results, semantic_results, hypothetical_results],
            top_k=5,
        )

        assert stats.input_lists == 4
        assert stats.unique_candidates == 6
        # kb-001 appears in 3 lists at rank 1 each time, should be #1
        assert results[0].id == "kb-001"
        assert results[0].appearances == 3

    def test_preserves_payload(self):
        results, _ = reciprocal_rank_fusion([
            [{"id": "doc-1", "text": "hello", "score": 0.9, "source_file": "readme.md", "section": "intro"}],
        ])
        assert results[0].payload["source_file"] == "readme.md"
        assert results[0].payload["section"] == "intro"

    def test_skips_items_without_id(self):
        results, stats = reciprocal_rank_fusion([
            [
                {"id": "doc-1", "text": "valid", "score": 0.9},
                {"text": "no id", "score": 0.8},  # Missing id
            ],
        ])
        assert len(results) == 1
        assert stats.total_candidates == 1

    def test_custom_k_parameter(self):
        """Lower k gives more weight to rank position."""
        items = [
            {"id": "doc-1", "text": "t1", "score": 0.5},
            {"id": "doc-2", "text": "t2", "score": 0.5},
        ]

        results_k20, _ = reciprocal_rank_fusion([items], k=20, top_rank_bonus=0.0)
        results_k100, _ = reciprocal_rank_fusion([items], k=100, top_rank_bonus=0.0)

        # With k=20: rank1 = 1/21 ≈ 0.0476, rank2 = 1/22 ≈ 0.0454, gap ≈ 0.002
        # With k=100: rank1 = 1/101 ≈ 0.0099, rank2 = 1/102 ≈ 0.0098, gap ≈ 0.0001
        gap_k20 = results_k20[0].rrf_score - results_k20[1].rrf_score
        gap_k100 = results_k100[0].rrf_score - results_k100[1].rrf_score
        assert gap_k20 > gap_k100  # Lower k = more rank sensitivity

    def test_stats_latency_populated(self):
        _, stats = reciprocal_rank_fusion([
            [{"id": "d1", "text": "t", "score": 0.5}],
        ])
        assert stats.latency_ms >= 0
