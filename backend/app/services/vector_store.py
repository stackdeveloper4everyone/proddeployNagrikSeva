from __future__ import annotations

import hashlib
import logging
import math
import re
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models

from backend.app.core.config import Settings

logger = logging.getLogger(__name__)


class PolicyVectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.collection_name = settings.qdrant_collection
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=30
        )
        self.vector_name = "dense"
        self.hash_dim = 256
        self.use_hash_fallback = False
        self.embedding_model = self._select_supported_model(settings.qdrant_embedding_model)
        try:
            self.client.set_model(self.embedding_model)
        except Exception:
            logger.exception(
                "fastembed_model_init_failed model=%s; switching_to_hash_fallback",
                self.embedding_model,
            )
            self.use_hash_fallback = True

    def add_search_results(self, results: list[dict]) -> int:
        documents: list[str] = []
        metadata: list[dict] = []
        ids: list[str] = []

        for item in results:
            title = item.get("title", "")
            url = item.get("url", "")
            raw_content = item.get("raw_content") or item.get("content") or item.get("answer") or ""
            if not raw_content.strip():
                raw_content = self._build_fallback_document(item)
            if not raw_content.strip():
                continue
            for index, chunk in enumerate(self._chunk_text(raw_content)[:4]):
                documents.append(chunk)
                metadata.append(
                    {
                        "title": title,
                        "url": url,
                        "content": chunk,
                    }
                )
                ids.append(self._stable_id(url=url, title=title, chunk=chunk, index=index))

        if not documents:
            return 0

        if not self.use_hash_fallback:
            self._ensure_fastembed_collection_compatible()
            try:
                self.client.add(
                    collection_name=self.collection_name,
                    documents=documents,
                    metadata=metadata,
                    ids=ids,
                )
                return len(documents)
            except AssertionError as exc:
                # Collection vector schema can become incompatible after embedding-model changes.
                if self.client.collection_exists(self.collection_name):
                    logger.warning(
                        "qdrant_collection_incompatible_recreating collection=%s model=%s reason=%s",
                        self.collection_name,
                        self.embedding_model,
                        str(exc),
                    )
                    self.client.delete_collection(self.collection_name)
                    self.client.add(
                        collection_name=self.collection_name,
                        documents=documents,
                        metadata=metadata,
                        ids=ids,
                    )
                    return len(documents)
                logger.exception("qdrant_add_assertion_failed")
            except Exception as exc:
                if self._is_vector_name_mismatch_error(exc):
                    self._recreate_fastembed_collection()
                    self.client.add(
                        collection_name=self.collection_name,
                        documents=documents,
                        metadata=metadata,
                        ids=ids,
                    )
                    return len(documents)
                logger.exception(
                    "qdrant_add_failed collection=%s model=%s docs=%s; switching_to_hash_fallback",
                    self.collection_name,
                    self.embedding_model,
                    len(documents),
                )
            self.use_hash_fallback = True

        self._upsert_with_hash_embeddings(documents=documents, metadata=metadata, ids=ids)
        return len(documents)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        if not self.client.collection_exists(self.collection_name):
            return []
        if not self.use_hash_fallback:
            self._ensure_fastembed_collection_compatible()
            try:
                results = self.client.query(
                    collection_name=self.collection_name,
                    query_text=query,
                    limit=limit,
                )
                return self._normalize_query_results(results)
            except Exception as exc:
                if self._is_vector_name_mismatch_error(exc):
                    self._recreate_fastembed_collection()
                    return []
                logger.exception("qdrant_query_failed; switching_to_hash_fallback")
                self.use_hash_fallback = True

        vector = self._hash_embed(query)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            using=self.vector_name,
            limit=limit,
            with_payload=True,
        )
        return [
            {
                "title": (point.payload or {}).get("title", "Untitled source"),
                "url": (point.payload or {}).get("url", ""),
                "content": (point.payload or {}).get("content", ""),
                "score": point.score,
            }
            for point in response.points
        ]

    def _chunk_text(self, text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
        chunks: list[str] = []
        cursor = 0
        while cursor < len(text):
            chunk = text[cursor : cursor + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            cursor += max(chunk_size - overlap, 1)
        return chunks

    def _stable_id(self, *, url: str, title: str, chunk: str, index: int) -> str:
        # Qdrant PointStruct IDs must be UUID or unsigned integer.
        # Use deterministic UUID5:
        # - same source chunk re-ingestion updates existing point
        # - new chunk content from same URL creates a new point (no accidental overwrite)
        base = (url or title or "unknown-source").strip().lower()
        chunk_fingerprint = hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:16]
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{base}-{index}-{chunk_fingerprint}"))

    def _normalize_query_results(self, results) -> list[dict]:
        normalized: list[dict] = []
        for item in results:
            normalized.append(
                {
                    "title": (item.metadata or {}).get("title", "Untitled source"),
                    "url": (item.metadata or {}).get("url", ""),
                    "content": getattr(item, "document", "") or (item.metadata or {}).get("content", ""),
                    "score": getattr(item, "score", None),
                }
            )
        return normalized

    def _build_fallback_document(self, item: dict) -> str:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        snippet = (item.get("content") or item.get("answer") or "").strip()
        parts = [part for part in [title, snippet, url] if part]
        return "\n".join(parts)

    def get_collection_stats(self) -> dict[str, int | bool]:
        if not self.client.collection_exists(self.collection_name):
            return {"collection_exists": False, "points_count": 0}
        info = self.client.get_collection(self.collection_name)
        points_count = getattr(info, "points_count", 0) or 0
        try:
            exact = self.client.count(collection_name=self.collection_name, count_filter=None, exact=True)
            exact_count = getattr(exact, "count", None)
            if isinstance(exact_count, int):
                points_count = exact_count
        except Exception:
            # Keep compatibility across client versions and continue with collection metadata count.
            pass
        return {
            "collection_exists": True,
            "points_count": int(points_count),
            "hash_fallback_active": self.use_hash_fallback,
        }

    def _select_supported_model(self, requested_model: str) -> str:
        supported = self.client.list_text_models()
        supported_names = set(supported.keys())
        if requested_model in supported_names:
            return requested_model

        fallback_priority = [
            "BAAI/bge-base-en-v1.5",
            "BAAI/bge-base-en",
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-small-en",
            "sentence-transformers/all-MiniLM-L6-v2",
        ]
        for model_name in fallback_priority:
            if model_name in supported_names:
                logger.warning(
                    "embedding_model_fallback requested=%s selected=%s",
                    requested_model,
                    model_name,
                )
                return model_name

        # Last-resort deterministic fallback to first supported model.
        first_supported = sorted(supported_names)[0]
        logger.warning(
            "embedding_model_fallback requested=%s selected=%s via_first_supported",
            requested_model,
            first_supported,
        )
        return first_supported

    def _upsert_with_hash_embeddings(self, *, documents: list[str], metadata: list[dict], ids: list[str]) -> None:
        self._ensure_hash_collection()
        points: list[models.PointStruct] = []
        for idx, doc, meta in zip(ids, documents, metadata):
            payload = {"document": doc, **meta}
            points.append(
                models.PointStruct(
                    id=idx,
                    vector={self.vector_name: self._hash_embed(doc)},
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def _ensure_hash_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            try:
                info = self.client.get_collection(self.collection_name)
                vectors = info.config.params.vectors
                if isinstance(vectors, dict) and self.vector_name in vectors:
                    return
            except Exception:
                pass
            # Existing collection schema is incompatible with hash fallback.
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.vector_name: models.VectorParams(size=self.hash_dim, distance=models.Distance.COSINE)
            },
        )

    def _hash_embed(self, text: str) -> list[float]:
        vector = [0.0] * self.hash_dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            tokens = [text.lower().strip()] if text.strip() else ["empty"]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.hash_dim
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _ensure_fastembed_collection_compatible(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            return
        expected_vector_name = self.client.get_vector_field_name()
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict) and expected_vector_name in vectors:
            return
        logger.warning(
            "qdrant_vector_schema_mismatch collection=%s expected=%s; recreating",
            self.collection_name,
            expected_vector_name,
        )
        self._recreate_fastembed_collection()

    def _recreate_fastembed_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

    def _is_vector_name_mismatch_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return "not existing vector name" in text or "wrong input" in text and "vector name" in text
