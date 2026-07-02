from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable

import httpx
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from memarena.errors import ProviderError
from memarena.providers.base import MemoryProvider, MemoryRecord, ProviderInfo

EmbedFn = Callable[[list[str]], list[list[float]]]


def chunk_text(text: str, *, chunk_size: int) -> list[str]:
    if not text:
        return []
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4))
def _openai_embed_batch(texts: list[str], *, model: str, api_key: str) -> list[list[float]]:
    response = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "input": texts},
        timeout=30.0,
    )
    response.raise_for_status()
    payload = response.json()
    return [item["embedding"] for item in payload["data"]]


def _default_embed_fn(model: str) -> EmbedFn:
    def embed(texts: list[str]) -> list[list[float]]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError(
                "OPENAI_API_KEY is not set; baseline_rag needs it to call the "
                "embeddings API (set it in .env, never hardcode it)."
            )
        return _openai_embed_batch(texts, model=model, api_key=api_key)

    return embed


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


class BaselineRAGProvider(MemoryProvider):
    """Deliberately naive floor (§5.5): fixed-size chunking, one pinned
    embedding model, cosine top-k, no summarization, no graph. The number
    every funded vendor must beat for their score to mean anything.
    """

    supports_temporal = False
    supports_update_resolution = False
    memory_representation = "extractive"  # returns raw chunks of ingested text

    def __init__(self, config: dict, *, embed_fn: EmbedFn | None = None):
        self._config = config
        self._chunk_size = config.get("chunk_size", 200)
        self._embedding_model = config.get("embedding_model", "text-embedding-3-small")
        self._embed_fn = embed_fn or _default_embed_fn(self._embedding_model)
        self._store: dict[str, list[tuple[MemoryRecord, list[float]]]] = {}

    def info(self) -> ProviderInfo:
        digest = hashlib.sha256(json.dumps(self._config, sort_keys=True).encode()).hexdigest()
        return ProviderInfo(
            name="baseline_rag",
            client_version="0.1.0",
            config_digest=digest,
            pricing_model="per_token",
        )

    def reset(self, namespace: str) -> None:
        self._store[namespace] = []

    def add(self, namespace: str, messages: list[dict[str, str]], *,
            session_id: str, timestamp: str) -> None:
        chunks: list[str] = []
        for message in messages:
            chunks.extend(chunk_text(message["content"], chunk_size=self._chunk_size))
        if not chunks:
            return

        embeddings = self._embed(chunks)
        bucket = self._store.setdefault(namespace, [])
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            record = MemoryRecord(id=f"{namespace}:{session_id}:{i}", content=chunk, created_at=timestamp)
            bucket.append((record, embedding))

    def search(self, namespace: str, query: str, *, top_k: int = 5) -> list[MemoryRecord]:
        bucket = self._store.get(namespace, [])
        if not bucket:
            return []

        [query_embedding] = self._embed([query])
        scored = [
            MemoryRecord(
                id=record.id,
                content=record.content,
                metadata=record.metadata,
                score=_cosine_similarity(query_embedding, embedding),
                created_at=record.created_at,
            )
            for record, embedding in bucket
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._embed_fn(texts)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"baseline_rag embedding failed: {exc}") from exc
