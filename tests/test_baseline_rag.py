import hashlib
import re

import pytest

from memarena.errors import ProviderError
from memarena.providers.baseline_rag import BaselineRAGProvider, chunk_text

DIM = 32


def _bow_vector(text: str) -> list[float]:
    """Deterministic hashed bag-of-words vector — a test double standing in
    for a real embedding model so cosine ranking has something meaningful
    to rank, without hitting the network."""
    vector = [0.0] * DIM
    for word in re.findall(r"\w+", text.lower()):
        idx = int(hashlib.sha256(word.encode()).hexdigest(), 16) % DIM
        vector[idx] += 1.0
    return vector


class TestChunkText:
    def test_short_text_is_a_single_chunk(self):
        assert chunk_text("hello world", chunk_size=200) == ["hello world"]

    def test_long_text_is_split_into_fixed_size_chunks(self):
        text = "a" * 450
        chunks = chunk_text(text, chunk_size=200)
        assert len(chunks) == 3
        assert chunks[0] == "a" * 200
        assert chunks[1] == "a" * 200
        assert chunks[2] == "a" * 50

    def test_empty_text_yields_no_chunks(self):
        assert chunk_text("", chunk_size=200) == []


class TestBaselineRAGProvider:
    def setup_method(self):
        self.embed_calls: list[list[str]] = []

        def embed_fn(texts: list[str]) -> list[list[float]]:
            self.embed_calls.append(list(texts))
            return [_bow_vector(t) for t in texts]

        self.embed_fn = embed_fn
        self.provider = BaselineRAGProvider(
            {"embedding_model": "text-embedding-3-small", "chunk_size": 200},
            embed_fn=embed_fn,
        )

    def test_info(self):
        info = self.provider.info()
        assert info.name == "baseline_rag"
        assert info.pricing_model == "per_token"
        assert len(info.config_digest) == 64

    def test_reset_is_idempotent(self):
        self.provider.reset("ns1")
        self.provider.reset("ns1")
        assert self.provider.search("ns1", "anything") == []
        assert self.embed_calls == []  # empty bucket short-circuits before embedding

    def test_add_then_search_returns_relevant_record_first(self):
        self.provider.reset("ns1")
        self.provider.add("ns1", [{"role": "user", "content": "My dog's name is Biscuit."}],
                           session_id="s1", timestamp="2026-01-01T00:00:00Z")
        self.provider.add("ns1", [{"role": "user", "content": "I like hiking on weekends."}],
                           session_id="s2", timestamp="2026-01-01T00:00:00Z")

        results = self.provider.search("ns1", "What is my dog's name?", top_k=5)

        assert len(results) == 2
        assert "Biscuit" in results[0].content
        assert results[0].score >= results[1].score

    def test_search_respects_top_k(self):
        self.provider.reset("ns1")
        for i in range(10):
            self.provider.add("ns1", [{"role": "user", "content": f"fact number {i}"}],
                               session_id=f"s{i}", timestamp="2026-01-01T00:00:00Z")
        results = self.provider.search("ns1", "fact", top_k=3)
        assert len(results) == 3

    def test_namespaces_are_isolated(self):
        self.provider.reset("ns1")
        self.provider.reset("ns2")
        self.provider.add("ns1", [{"role": "user", "content": "secret A"}],
                           session_id="s1", timestamp="2026-01-01T00:00:00Z")
        assert self.provider.search("ns2", "secret", top_k=5) == []

    def test_records_have_stable_ids_and_created_at(self):
        self.provider.reset("ns1")
        self.provider.add("ns1", [{"role": "user", "content": "hi"}],
                           session_id="s1", timestamp="2026-01-01T00:00:00Z")
        [record] = self.provider.search("ns1", "hi", top_k=5)
        assert record.created_at == "2026-01-01T00:00:00Z"
        assert record.id

    def test_embed_fn_failure_raises_provider_error(self):
        def bad_embed(texts: list[str]) -> list[list[float]]:
            raise RuntimeError("network down")

        provider = BaselineRAGProvider({"embedding_model": "m", "chunk_size": 200}, embed_fn=bad_embed)
        provider.reset("ns1")
        with pytest.raises(ProviderError):
            provider.add("ns1", [{"role": "user", "content": "hi"}],
                          session_id="s1", timestamp="2026-01-01T00:00:00Z")

    def test_missing_api_key_raises_provider_error_without_stub(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = BaselineRAGProvider({"embedding_model": "text-embedding-3-small", "chunk_size": 200})
        provider.reset("ns1")
        with pytest.raises(ProviderError):
            provider.add("ns1", [{"role": "user", "content": "hi"}],
                          session_id="s1", timestamp="2026-01-01T00:00:00Z")
