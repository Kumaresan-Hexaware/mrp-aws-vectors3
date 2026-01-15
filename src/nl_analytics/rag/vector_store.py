from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Protocol
from pathlib import Path
import json

from nl_analytics.exceptions.errors import RetrievalError
from nl_analytics.logging.logger import get_logger

log = get_logger("rag.vector_store")

@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    metadata: Dict[str, Any]
    score: float

class VectorStore(Protocol):
    def upsert(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]) -> None: ...
    def query(self, query_text: str, top_k: int) -> List[RetrievedChunk]: ...

class ChromaVectorStore:
    def __init__(self, persist_dir: str, embedding_fn):
        self.persist_dir = persist_dir
        import chromadb
        from chromadb.config import Settings
        settings = Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory=str(self.persist_dir),  # or your chroma_dir
            chroma_api_impl="chromadb.api.segment.SegmentAPI",  # ✅ critical for Windows stability
        )

        self.client = chromadb.Client(settings=settings)
        # Create / open a persistent collection.
        # Note: embedding_fn must match Chroma's interface; orchestrator provides a compatible callable.
        self.collection = self.client.get_or_create_collection(
            name="nl-analytics",
            embedding_function=embedding_fn,
        )

    def upsert(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        try:
            self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
            log.info("Upserted chunks", extra={"count": len(ids)})
        except Exception as e:
            log.exception("Chroma upsert failed")
            raise RetrievalError("Vector upsert failed") from e

    def query(self, query_text: str, top_k: int) -> List[RetrievedChunk]:
        try:
            res = self.collection.query(query_texts=[query_text], n_results=top_k, include=["documents", "metadatas", "distances"])
            docs = res["documents"][0]
            metas = res["metadatas"][0]
            dists = res["distances"][0]
            out = []
            for t, m, d in zip(docs, metas, dists):
                score = float(max(0.0, 1.0 - d))
                out.append(RetrievedChunk(text=t, metadata=m, score=score))
            return out
        except Exception as e:
            log.exception("Chroma query failed")
            raise RetrievalError("Vector query failed") from e
    def has_ids(self, ids: List[str]) -> List[bool]:
        """Best-effort existence check without embedding any text."""
        try:
            res = self.collection.get(ids=ids, include=[])
            present = set(res.get("ids") or [])
            return [i in present for i in ids]
        except Exception:
            return [False for _ in ids]


class PgVectorStore:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("PGVector backend not implemented in this prototype.")

class S3VectorStore:
    """Simple S3-backed vector store.

    This backend stores vectors as a single JSONL object in S3:
      s3://{bucket}/{prefix}/records.jsonl

    Each line is a JSON object:
      {"id": "...", "text": "...", "metadata": {...}, "embedding": [..floats..]}

    Retrieval is performed via in-memory cosine similarity over all stored embeddings.
    This is suitable for prototypes and small-to-medium schema indexes (hundreds/thousands of records).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        cache_dir: str,
        refresh_seconds: int,
        embedding_fn,
    ):
        if not bucket:
            raise RetrievalError("S3 vector backend requires S3_VECTOR_BUCKET to be set.")
        import boto3
        self.s3 = boto3.client("s3")
        self.bucket = bucket
        self.prefix = (prefix or "").strip().strip("/")
        self.refresh_seconds = int(refresh_seconds)
        self.embedding_fn = embedding_fn

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.local_file = self.cache_dir / "records.jsonl"
        self._last_loaded_at = 0.0

        self._records: Dict[str, Dict[str, Any]] = {}
        self._load(force=True)

    def _s3_key(self) -> str:
        return f"{self.prefix}/records.jsonl" if self.prefix else "records.jsonl"

    def _download_if_exists(self) -> bool:
        key = self._s3_key()
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
        except Exception:
            return False

        try:
            self.s3.download_file(self.bucket, key, str(self.local_file))
            log.info("Downloaded S3 vector index", extra={"bucket": self.bucket, "key": key})
            return True
        except Exception as e:
            log.exception("Failed downloading S3 vector index")
            raise RetrievalError("Failed to download S3 vector index") from e

    def _upload(self) -> None:
        key = self._s3_key()
        log.info(f"Key **********:::{key}")
        try:
            self.s3.upload_file(str(self.local_file), self.bucket, key)
            log.info("Uploaded S3 vector index", extra={"bucket": self.bucket, "key": key})
        except Exception as e:
            log.exception("Failed uploading S3 vector index")
            raise RetrievalError("Failed to upload S3 vector index") from e

    def _load(self, force: bool = False) -> None:
        import time
        if not force and (time.time() - self._last_loaded_at) < self.refresh_seconds:
            return

        # Prefer local cache if exists; otherwise try S3
        if not self.local_file.exists():
            self._download_if_exists()

        self._records = {}
        if self.local_file.exists():
            try:
                for line in self.local_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    rid = obj["id"]
                    self._records[rid] = obj
                log.info("Loaded S3 vector records", extra={"count": len(self._records)})
            except Exception as e:
                log.exception("Failed parsing local S3 vector cache")
                raise RetrievalError("Failed to parse S3 vector cache") from e

        self._last_loaded_at = time.time()

    def _persist_local(self) -> None:
        try:
            lines = [json.dumps(self._records[k], ensure_ascii=False) for k in sorted(self._records.keys())]
            self.local_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        except Exception as e:
            log.exception("Failed writing local S3 vector cache")
            raise RetrievalError("Failed writing local S3 vector cache") from e

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        # returns [-1,1]
        if not a or not b or len(a) != len(b):
            return -1.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na <= 0.0 or nb <= 0.0:
            return -1.0
        return dot / ((na ** 0.5) * (nb ** 0.5))

    def upsert(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        if not (len(ids) == len(texts) == len(metadatas)):
            raise RetrievalError("upsert requires ids, texts, metadatas of same length")

        self._load(force=False)

        try:
            embeddings = self.embedding_fn(texts)
        except Exception as e:
            log.exception("Embedding generation failed during upsert")
            raise RetrievalError("Embedding generation failed") from e

        for rid, text, meta, emb in zip(ids, texts, metadatas, embeddings):
            self._records[rid] = {
                "id": rid,
                "text": text,
                "metadata": meta or {},
                "embedding": list(map(float, emb)),
            }

        self._persist_local()
        self._upload()
        log.info("Upserted S3 vector records", extra={"count": len(ids)})

    def has_ids(self, ids: List[str]) -> List[bool]:
        self._load(force=False)
        return [rid in self._records for rid in ids]


    def query(self, query_text: str, top_k: int) -> List[RetrievedChunk]:
        self._load(force=False)

        if not self._records:
            return []

        try:
            q_emb = self.embedding_fn([query_text])[0]
            # Validate query embedding dimension if we know the index dimension.
            if getattr(self, "index_dimension", None) is not None:
                expected = int(self.index_dimension)
                if len(q_emb) != expected:
                    raise RetrievalError(
                        f"S3 Vectors index expects dimension={expected}, but query embedding has dimension={len(q_emb)}."
                    )

            if getattr(self, "index_dimension", None) is not None:
                expected = int(self.index_dimension)
                if len(q_emb) != expected:
                    raise RetrievalError(
                        f"S3 Vectors index expects dimension={expected}, but query embedding has dimension={len(q_emb)}."
                    )
        except Exception as e:
            log.exception("Embedding generation failed during query")
            raise RetrievalError("Embedding generation failed") from e

        scored = []
        for obj in self._records.values():
            sim = self._cosine(q_emb, obj.get("embedding") or [])
            # map [-1,1] -> [0,1] to align with our confidence usage
            score = max(0.0, (sim + 1.0) / 2.0)
            scored.append((score, obj))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[RetrievedChunk] = []
        for score, obj in scored[: max(1, int(top_k))]:
            out.append(RetrievedChunk(text=obj["text"], metadata=obj.get("metadata") or {}, score=float(score)))
        return out

class S3VectorsVectorStore:
    """Native Amazon S3 Vectors backend (vector buckets + vector indexes).

    Uses boto3 client: boto3.client("s3vectors")
    Writes vectors using PutVectors and queries using QueryVectors.

    Notes:
    - Vector payload must be float32. S3 Vectors will coerce, but we still cast to float.
    - `returnMetadata=True` requires `s3vectors:GetVectors` permission in addition to `s3vectors:QueryVectors`.
    - This backend is recommended when your bucket is created via `aws s3vectors ...` (vector bucket).
    """

    def __init__(
        self,
        bucket: str,
        index: str,
        namespace: str,
        refresh_seconds: int,
        embedding_fn,
        region_name: str | None = None,
    ):
        if not bucket:
            raise RetrievalError("S3 Vectors backend requires S3VECTORS_BUCKET to be set.")
        if not index:
            raise RetrievalError("S3 Vectors backend requires S3VECTORS_INDEX to be set.")

        import boto3

        self.bucket = bucket
        self.index = index
        self.namespace = namespace or "default"
        self.refresh_seconds = int(refresh_seconds)
        self.embedding_fn = embedding_fn
        self.client = boto3.client("s3vectors", region_name=region_name)

        # Best-effort: fetch index attributes (dimension, dataType) so we can:
        # 1) validate embedding dimension before PutVectors/QueryVectors
        # 2) (for models that support it) request the correct embedding dimension from Bedrock.
        self.index_dimension = None
        self.index_data_type = None
        try:
            info = self.client.get_index(vectorBucketName=self.bucket, indexName=self.index)
            index_obj = (info or {}).get("index") or {}
            self.index_dimension = index_obj.get("dimension")
            self.index_data_type = index_obj.get("dataType")
            log.info(
                "S3 Vectors index loaded",
                extra={
                    "bucket": self.bucket,
                    "index": self.index,
                    "dimension": self.index_dimension,
                    "data_type": self.index_data_type,
                },
            )
            if self.index_dimension is not None and hasattr(self.embedding_fn, "desired_dimensions"):
                # Allows Titan Text Embeddings V2 to return vectors that match index dimension.
                setattr(self.embedding_fn, "desired_dimensions", int(self.index_dimension))
        except Exception as e:
            log.warning(
                "Could not fetch S3 Vectors index attributes (GetIndex). Continuing without dimension validation.",
                extra={"bucket": self.bucket, "index": self.index, "error": str(e)},
            )

    @staticmethod
    def _to_float32_payload(vec: List[float]) -> Dict[str, Any]:
        return {"float32": [float(x) for x in vec]}

    def upsert(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        if not (len(ids) == len(texts) == len(metadatas)):
            raise RetrievalError("upsert requires ids, texts, metadatas of same length")

        try:
            embs = self.embedding_fn(texts)
        except Exception as e:
            log.exception("Embedding generation failed during S3Vectors upsert")
            raise RetrievalError("Embedding generation failed") from e

        # Validate embedding dimensions if we know the index dimension.
        if getattr(self, "index_dimension", None) is not None:
            expected = int(self.index_dimension)
            for j, emb in enumerate(embs):
                if len(emb) != expected:
                    raise RetrievalError(
                        f"S3 Vectors index expects dimension={expected}, but embedding[{j}] has dimension={len(emb)}. "
                        f"Fix by aligning the embedding model output dimension with the index dimension."
                    )


        # API allows up to 500 vectors per request (keep smaller for safety)
        BATCH = 200
        for i in range(0, len(ids), BATCH):
            batch_vectors = []
            for rid, text, meta, emb in zip(
                ids[i:i+BATCH],
                texts[i:i+BATCH],
                metadatas[i:i+BATCH],
                embs[i:i+BATCH],
            ):
                meta = dict(meta or {})
                meta.setdefault("source_text", text)
                meta.setdefault("namespace", self.namespace)

                batch_vectors.append(
                    {
                        "key": str(rid),
                        "data": self._to_float32_payload(list(emb)),
                        "metadata": meta,
                    }
                )

            try:
                self.client.put_vectors(
                    vectorBucketName=self.bucket,
                    indexName=self.index,
                    vectors=batch_vectors,
                )
            except Exception as e:
                log.exception("S3Vectors put_vectors failed")
                raise RetrievalError("Failed to write vectors to S3 Vectors index") from e

        log.info("Upserted vectors into S3 Vectors", extra={"count": len(ids), "bucket": self.bucket, "index": self.index})

    def query(self, query_text: str, top_k: int) -> List[RetrievedChunk]:
        try:
            q_emb = self.embedding_fn([query_text])[0]
        except Exception as e:
            log.exception("Embedding generation failed during S3Vectors query")
            raise RetrievalError("Embedding generation failed") from e

        try:
            resp = self.client.query_vectors(
                vectorBucketName=self.bucket,
                indexName=self.index,
                topK=int(top_k),
                queryVector=self._to_float32_payload(list(q_emb)),
                returnMetadata=True,
                returnDistance=True,
            )
        except Exception as e:
            log.exception("S3Vectors query_vectors failed")
            raise RetrievalError("Failed to query vectors from S3 Vectors index") from e

        vectors = resp.get("vectors") or []
        out: List[RetrievedChunk] = []
        for v in vectors:
            dist = float(v.get("distance", 0.0))
            # distance metric depends on index config; map to a monotonic score in (0,1]
            score = 1.0 / (1.0 + max(0.0, dist))
            meta = v.get("metadata") or {}
            text = meta.get("source_text", "")
            out.append(RetrievedChunk(text=text, metadata=meta, score=score))

        return out
