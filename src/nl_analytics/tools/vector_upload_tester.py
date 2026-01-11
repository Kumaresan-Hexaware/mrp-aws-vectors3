from __future__ import annotations

"""
VectorUploadTester
------------------

A tiny, self-contained, end-to-end validator for:
1) reading a local file,
2) generating an embedding via Amazon Bedrock,
3) inserting that embedding into an Amazon S3 Vectors index via PutVectors,
4) optionally verifying the insert via GetVectors.

Usage (from repo root):
    python -m nl_analytics.tools.vector_upload_tester --file path/to/document.txt

If you need to override config at runtime, prefer environment variables:
    VECTOR_BACKEND=s3vectors
    AWS_REGION=us-east-1
    BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0
    S3VECTORS_BUCKET=your-vector-bucket-name
    S3VECTORS_INDEX=your-index-name   (or set to an ARN; tester will detect it)
    S3VECTORS_NAMESPACE=default

Optionally override S3 Vectors region if your vector bucket is in a different region
than your Bedrock region:
    python -m nl_analytics.tools.vector_upload_tester --file ... --s3vectors-region us-west-2
"""

import argparse
import datetime as _dt
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nl_analytics.config.settings import load_settings
from nl_analytics.logging.logger import init_logging, get_logger
from nl_analytics.bedrock.client import BedrockClient, BedrockConfig

log = get_logger("tools.vector_upload_tester")


@dataclass
class VectorUploadResult:
    ok: bool
    key: str
    embedding_dim: int
    vector_bucket: str
    index_name_or_arn: str
    error: Optional[str] = None
    index_dimension: Optional[int] = None
    index_data_type: Optional[str] = None


class VectorUploadTester:
    """
    End-to-end test helper for S3 Vectors ingestion.

    Notes:
    - Uses the same BedrockClient.embed() implementation as the main app.
    - Uses the native s3vectors boto3 client.
    - Validates embedding dimension vs index dimension when possible (via GetIndex).
    """

    def __init__(
        self,
        *,
        aws_region: str,
        bedrock_embed_model_id: str,
        s3vectors_bucket: str,
        s3vectors_index_or_arn: str,
        namespace: str = "default",
        s3vectors_region: Optional[str] = None,
        use_mock_bedrock: bool = False,
    ) -> None:
        if not s3vectors_bucket:
            raise ValueError("Missing S3VECTORS_BUCKET (vector bucket name).")
        if not s3vectors_index_or_arn:
            raise ValueError("Missing S3VECTORS_INDEX (index name) or an index ARN.")
        if not aws_region:
            raise ValueError("Missing AWS_REGION.")

        self.aws_region = aws_region
        self.bedrock_embed_model_id = bedrock_embed_model_id
        self.bucket = s3vectors_bucket
        self.index_or_arn = s3vectors_index_or_arn
        self.namespace = namespace or "default"
        self.s3vectors_region = s3vectors_region or aws_region
        self.use_mock_bedrock = use_mock_bedrock

        # Bedrock
        self.bedrock = BedrockClient(
            BedrockConfig(
                region=self.aws_region,
                embed_model_id=self.bedrock_embed_model_id,
                chat_model_id=os.environ.get("BEDROCK_CHAT_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"),
                temperature=float(os.environ.get("LLM_TEMPERATURE", "0.1")),
                max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "1200")),
                use_mock=self.use_mock_bedrock,
            )
        )

        # S3 Vectors
        import boto3

        # Important: s3vectors is regional. If bucket/index were created in a different region,
        # you must point the client at that region.
        self.s3v = boto3.client("s3vectors", region_name=self.s3vectors_region)

    @staticmethod
    def _read_text_for_embedding(
        file_path: str,
        *,
        max_bytes: int = 2_000_000,
        max_chars: int = 25_000,
        fallback_encodings: Optional[List[str]] = None,
    ) -> Tuple[str, str]:
        """
        Read a local file as text for embedding.

        Returns: (text, encoding_used)
        """
        p = Path(file_path)
        raw = p.read_bytes()[:max_bytes]

        encodings = ["utf-8"]
        if fallback_encodings:
            encodings.extend([e for e in fallback_encodings if e and e.lower() != "utf-8"])
        encodings.extend(["latin-1", "cp1252"])

        last_err: Optional[Exception] = None
        for enc in encodings:
            try:
                text = raw.decode(enc)
                # trim to keep within typical model limits
                text = text[:max_chars]
                return text, enc
            except Exception as e:
                last_err = e
                continue

        # last resort: replace errors
        text = raw.decode("utf-8", errors="replace")[:max_chars]
        return text, "utf-8(errors=replace)"

    @staticmethod
    def _to_float32_payload(vec: List[float]) -> Dict[str, Any]:
        # S3 Vectors expects a tagged union: {"float32": [ ... ]}
        return {"float32": [float(x) for x in vec]}

    @staticmethod
    def _is_arn(value: str) -> bool:
        return value.strip().lower().startswith("arn:")

    def _get_index_attributes(self) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """
        Best-effort: fetch index dimension/dataType.
        Returns: (dimension, dataType, error_str)
        """
        try:
            if self._is_arn(self.index_or_arn):
                resp = self.s3v.get_index(indexArn=self.index_or_arn)
            else:
                resp = self.s3v.get_index(vectorBucketName=self.bucket, indexName=self.index_or_arn)

            idx = (resp or {}).get("index") or {}
            dim = idx.get("dimension")
            dtype = idx.get("dataType")
            return (int(dim) if dim is not None else None, str(dtype) if dtype else None, None)
        except Exception as e:
            return (None, None, f"{type(e).__name__}: {e}")

    def put_single_vector_from_file(
        self,
        file_path: str,
        *,
        key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VectorUploadResult:
        """
        Embeds the file content and uploads a single vector to S3 Vectors.
        """
        # 1) Read file
        settings = load_settings()
        text, enc = self._read_text_for_embedding(
            file_path,
            fallback_encodings=getattr(settings, "fallback_encodings", None),
        )
        if not text.strip():
            return VectorUploadResult(
                ok=False,
                key=key or "",
                embedding_dim=0,
                vector_bucket=self.bucket,
                index_name_or_arn=self.index_or_arn,
                error="File text was empty after decoding; nothing to embed.",
            )

        # 2) Get index attributes (best-effort)
        idx_dim, idx_dtype, idx_err = self._get_index_attributes()
        if idx_err:
            log.warning("Could not fetch index attributes via GetIndex", extra={"error": idx_err})

        # 3) Generate embedding (request index dimension when possible)
        if idx_dim is not None:
            emb = self.bedrock.embed([text], dimensions=int(idx_dim))[0]
        else:
            emb = self.bedrock.embed([text])[0]
        emb_dim = len(emb)

        # 4) Validate dimension match if we know index dimension
        if idx_dim is not None and emb_dim != idx_dim:
            return VectorUploadResult(
                ok=False,
                key=key or "",
                embedding_dim=emb_dim,
                vector_bucket=self.bucket,
                index_name_or_arn=self.index_or_arn,
                error=f"Embedding dimension mismatch: embedding_dim={emb_dim}, index_dim={idx_dim}.",
                index_dimension=idx_dim,
                index_data_type=idx_dtype,
            )

        # 5) PutVectors
        key = key or f"tester::{Path(file_path).name}::{uuid.uuid4().hex}"
        meta = dict(metadata or {})
        meta.setdefault("kind", "vector_upload_tester")
        meta.setdefault("source_file", str(Path(file_path).name))
        meta.setdefault("encoding_used", enc)
        meta.setdefault("namespace", self.namespace)
        meta.setdefault("created_at", _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z")

        put_req: Dict[str, Any] = {
            "vectors": [
                {
                    "key": str(key),
                    "data": self._to_float32_payload(list(emb)),
                    "metadata": meta,
                }
            ]
        }

        try:
            if self._is_arn(self.index_or_arn):
                put_req["indexArn"] = self.index_or_arn
            else:
                put_req["vectorBucketName"] = self.bucket
                put_req["indexName"] = self.index_or_arn

            self.s3v.put_vectors(**put_req)
        except Exception as e:
            return VectorUploadResult(
                ok=False,
                key=str(key),
                embedding_dim=emb_dim,
                vector_bucket=self.bucket,
                index_name_or_arn=self.index_or_arn,
                error=f"{type(e).__name__}: {e}",
                index_dimension=idx_dim,
                index_data_type=idx_dtype,
            )

        # 6) Verify (best-effort) with GetVectors
        try:
            if self._is_arn(self.index_or_arn):
                _ = self.s3v.get_vectors(indexArn=self.index_or_arn, keys=[str(key)], returnData=False, returnMetadata=True)
            else:
                _ = self.s3v.get_vectors(
                    vectorBucketName=self.bucket,
                    indexName=self.index_or_arn,
                    keys=[str(key)],
                    returnData=False,
                    returnMetadata=True,
                )
        except Exception as e:
            # Upload succeeded, but we couldn't verify. Still return ok=True.
            log.warning("PutVectors succeeded but GetVectors verification failed", extra={"error": str(e), "key": str(key)})

        return VectorUploadResult(
            ok=True,
            key=str(key),
            embedding_dim=emb_dim,
            vector_bucket=self.bucket,
            index_name_or_arn=self.index_or_arn,
            error=None,
            index_dimension=idx_dim,
            index_data_type=idx_dtype,
        )


def _build_from_settings(args: argparse.Namespace) -> VectorUploadTester:
    settings = load_settings()
    init_logging(getattr(settings, "log_level", "INFO"))

    aws_region = args.aws_region or getattr(settings, "aws_region", None) or os.environ.get("AWS_REGION", "")
    embed_model_id = args.embed_model_id or getattr(settings, "bedrock_embed_model_id", None) or os.environ.get("BEDROCK_EMBED_MODEL_ID", "")
    bucket = args.bucket or getattr(settings, "s3vectors_bucket", None) or os.environ.get("S3VECTORS_BUCKET", "")
    index_or_arn = args.index or getattr(settings, "s3vectors_index", None) or os.environ.get("S3VECTORS_INDEX", "")
    namespace = args.namespace or getattr(settings, "s3vectors_namespace", None) or os.environ.get("S3VECTORS_NAMESPACE", "default")

    return VectorUploadTester(
        aws_region=aws_region,
        bedrock_embed_model_id=embed_model_id,
        s3vectors_bucket=bucket,
        s3vectors_index_or_arn=index_or_arn,
        namespace=namespace,
        s3vectors_region=args.s3vectors_region,
        use_mock_bedrock=bool(args.use_mock_bedrock),
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end validator for Bedrock embeddings -> S3 Vectors PutVectors.")
    parser.add_argument("--file", required=True, help="Path to a local file to embed and upload.")
    parser.add_argument("--key", default=None, help="Optional explicit vector key. If omitted, a unique key is generated.")
    parser.add_argument("--aws-region", dest="aws_region", default=None, help="AWS region for Bedrock (defaults from config/AWS_REGION).")
    parser.add_argument("--s3vectors-region", default=None, help="AWS region for S3 Vectors (defaults to aws-region).")
    parser.add_argument("--embed-model-id", dest="embed_model_id", default=None, help="Bedrock embedding model id (defaults from config/BEDROCK_EMBED_MODEL_ID).")
    parser.add_argument("--bucket", default=None, help="S3 Vectors bucket name (defaults from config/S3VECTORS_BUCKET).")
    parser.add_argument("--index", default=None, help="S3 Vectors index name OR index ARN (defaults from config/S3VECTORS_INDEX).")
    parser.add_argument("--namespace", default=None, help="Namespace tag to add in vector metadata.")
    parser.add_argument("--use-mock-bedrock", action="store_true", help="Use mock embeddings instead of calling Bedrock (for local dry runs).")
    args = parser.parse_args(argv)

    tester = _build_from_settings(args)

    res = tester.put_single_vector_from_file(args.file, key=args.key)
    if res.ok:
        print("✅ Vector upload SUCCESS")
        print(f"   key: {res.key}")
        print(f"   embedding_dim: {res.embedding_dim}")
        print(f"   vector_bucket: {res.vector_bucket}")
        print(f"   index: {res.index_name_or_arn}")
        if res.index_dimension is not None:
            print(f"   index_dimension: {res.index_dimension}  dataType: {res.index_data_type}")
        return 0

    print("❌ Vector upload FAILED")
    print(f"   error: {res.error}")
    print(f"   embedding_dim: {res.embedding_dim}")
    if res.index_dimension is not None:
        print(f"   index_dimension: {res.index_dimension}  dataType: {res.index_data_type}")
    print(f"   vector_bucket: {res.vector_bucket}")
    print(f"   index: {res.index_name_or_arn}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
