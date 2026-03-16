"""Shared fixtures for RAGRouter tests."""

from __future__ import annotations

import os

import pytest

# Disable external service connections during tests
os.environ.setdefault("HYBRID_QUERY_EXPANSION", "true")
os.environ.setdefault("HYBRID_LLM_RERANKING", "true")
os.environ.setdefault("HYBRID_LITELLM_API_BASE", "http://test-litellm:4000")
