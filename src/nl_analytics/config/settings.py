from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import yaml
from dotenv import load_dotenv

load_dotenv()

def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(key, default)

def _env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")

def _env_list(key: str, default: List[str]) -> List[str]:
    val = os.environ.get(key)
    if not val:
        return default
    return [x.strip() for x in val.split(",") if x.strip()]

@dataclass(frozen=True)
class Settings:
    env: str
    log_level: str

    delimiter: str
    fallback_encodings: List[str]
    skip_bad_lines: bool
    max_rows_preview: int

    vector_backend: str
    chroma_dir: str
    rag_top_k: int
    rag_min_score: float
    chunk_size: int
    chunk_overlap: int

    # Retrieval quality (wide schemas)
    rag_oversample_factor: int
    rag_enable_rerank: bool
    rag_schema_only_for_planning: bool
    rag_schema_kinds: List[str]

    # Enforce candidate-column allowlist during planning (prevents picking arbitrary columns)
    rag_enforce_candidate_allowlist: bool

    # Column resolver (shortlist)
    col_resolver_max_candidates: int

    # S3 vector backend
    s3_vector_bucket: str
    s3_vector_prefix: str
    s3_vector_cache_dir: str
    s3_vector_refresh_seconds: int

    # Native S3 Vectors backend
    s3vectors_bucket: str
    s3vectors_index: str
    s3vectors_namespace: str
    s3vectors_refresh_seconds: int

    agent_min_confidence: float
    refuse_below_confidence: bool

    use_mock_bedrock: bool
    aws_region: str
    bedrock_chat_model_id: str
    bedrock_embed_model_id: str
    llm_temperature: float
    llm_max_tokens: int

    export_dir: str
    saved_query_dir: str

    # ------------------------------------------------------------------
    # Query/compute backend (duckdb / athena / redshift)
    # ------------------------------------------------------------------
    db_type: str

    # Athena (Query results are read from the configured output S3 location)
    athena_database: str
    athena_workgroup: str
    athena_output_location: str
    athena_catalog: str

    # Redshift Data API (provisioned or serverless)
    redshift_database: str
    redshift_cluster_id: str
    redshift_workgroup_name: str
    redshift_db_user: str
    redshift_secret_arn: str

    # ------------------------------------------------------------------
    # Query persistence (optional)
    # ------------------------------------------------------------------
    query_store_backend: str
    query_store_dir: str


    query_store_mode: str
    # DynamoDB query store (when query_store_backend = dynamodb)
    dynamodb_table_name: str
    dynamodb_pk_name: str
    dynamodb_sk_name: str
    dynamodb_region: str
    dynamodb_endpoint_url: str
    dynamodb_ttl_attribute: str
    dynamodb_ttl_seconds: int

def load_settings() -> Settings:
    app_env = _env("APP_ENV", "dev")
    cfg_path = Path("config") / f"{app_env}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    delimiter = _env("DEFAULT_DELIMITER", cfg["ingestion"]["delimiter"])
    fallback_encodings = _env_list("FALLBACK_ENCODINGS", cfg["ingestion"]["fallback_encodings"])
    skip_bad_lines = _env_bool("SKIP_BAD_LINES", cfg["ingestion"]["skip_bad_lines"])
    max_rows_preview = int(_env("MAX_ROWS_PREVIEW", str(cfg["ingestion"]["max_rows_preview"])))

    vector_backend = _env("VECTOR_BACKEND", cfg["rag"]["backend"])
    chroma_dir = _env("CHROMA_DIR", cfg["rag"]["chroma_dir"])
    rag_top_k = int(_env("RAG_TOP_K", str(cfg["rag"]["top_k"])))
    rag_min_score = float(_env("RAG_MIN_SCORE", str(cfg["rag"]["min_score"])))
    chunk_size = int(_env("RAG_CHUNK_SIZE", str(cfg["rag"]["chunk_size"])))
    chunk_overlap = int(_env("RAG_CHUNK_OVERLAP", str(cfg["rag"]["chunk_overlap"])))

    rag_oversample_factor = int(_env("RAG_OVERSAMPLE_FACTOR", str(cfg["rag"].get("oversample_factor", 5))))
    rag_enable_rerank = _env_bool("RAG_ENABLE_RERANK", bool(cfg["rag"].get("enable_rerank", True)))
    rag_schema_only_for_planning = _env_bool(
        "RAG_SCHEMA_ONLY_FOR_PLANNING", bool(cfg["rag"].get("schema_only_for_planning", True))
    )
    rag_schema_kinds = _env_list(
        "RAG_SCHEMA_KINDS",
        list(cfg["rag"].get("schema_kinds", ["table", "join", "column", "alias", "schema"]))
    )
    rag_enforce_candidate_allowlist = _env_bool(
        "RAG_ENFORCE_ALLOWLIST", bool(cfg["rag"].get("enforce_candidate_allowlist", True))
    )

    col_resolver_max_candidates = int(
        _env("COL_RESOLVER_MAX_CANDIDATES", str(cfg.get("column_resolver", {}).get("max_candidates", 40)))
    )

    s3_cfg = (cfg.get("rag") or {}).get("s3_vector") or {}
    s3_vector_bucket = _env("S3_VECTOR_BUCKET", str(s3_cfg.get("bucket", "")))
    s3_vector_prefix = _env("S3_VECTOR_PREFIX", str(s3_cfg.get("prefix", "nl-analytics/vectors")))
    s3_vector_cache_dir = _env("S3_VECTOR_CACHE_DIR", str(s3_cfg.get("cache_dir", ".s3_vector_cache")))
    s3_vector_refresh_seconds = int(_env("S3_VECTOR_REFRESH_SECONDS", str(s3_cfg.get("refresh_seconds", 300))))

    s3v_cfg = (cfg.get("rag") or {}).get("s3vectors") or {}
    s3vectors_bucket = _env("S3VECTORS_BUCKET", str(s3v_cfg.get("bucket", "")))
    s3vectors_index = _env("S3VECTORS_INDEX", str(s3v_cfg.get("index", "")))
    s3vectors_namespace = _env("S3VECTORS_NAMESPACE", str(s3v_cfg.get("namespace", "default")))
    s3vectors_refresh_seconds = int(_env("S3VECTORS_REFRESH_SECONDS", str(s3v_cfg.get("refresh_seconds", 300))))

    agent_min_confidence = float(_env("AGENT_MIN_CONFIDENCE", str(cfg["agent"]["min_confidence"])))
    refuse_below_confidence = bool(cfg["agent"].get("refuse_below_confidence", True))

    use_mock_bedrock = _env_bool("USE_MOCK_BEDROCK", False)
    aws_region = _env("AWS_REGION", cfg["models"]["region"])
    bedrock_chat_model_id = _env("BEDROCK_CHAT_MODEL_ID", cfg["models"]["chat_model_id"])
    bedrock_embed_model_id = _env("BEDROCK_EMBED_MODEL_ID", cfg["models"]["embed_model_id"])
    llm_temperature = float(_env("LLM_TEMPERATURE", str(cfg["models"]["temperature"])))
    llm_max_tokens = int(_env("LLM_MAX_TOKENS", str(cfg["models"]["max_tokens"])))

    export_dir = _env("EXPORT_DIR", cfg["export"]["export_dir"])
    saved_query_dir = _env("SAVED_QUERY_DIR", cfg["export"]["saved_query_dir"])

    # ------------------------------ Database ------------------------------
    # Keep everything optional and default to duckdb so existing projects
    # continue to run without any config changes.
    db_cfg = (cfg.get("database") or {})
    db_type = (_env("DB_TYPE", str(db_cfg.get("db_type", "duckdb"))) or "duckdb").strip().lower()

    ath_cfg = (db_cfg.get("athena") or {})
    athena_database = _env("ATHENA_DATABASE", str(ath_cfg.get("database", ""))) or ""
    athena_workgroup = _env("ATHENA_WORKGROUP", str(ath_cfg.get("workgroup", ""))) or ""
    athena_output_location = _env("ATHENA_OUTPUT_LOCATION", str(ath_cfg.get("output_location", ""))) or ""
    athena_catalog = _env("ATHENA_CATALOG", str(ath_cfg.get("catalog", "AwsDataCatalog"))) or "AwsDataCatalog"

    rs_cfg = (db_cfg.get("redshift") or {})
    redshift_database = _env("REDSHIFT_DATABASE", str(rs_cfg.get("database", ""))) or ""
    redshift_cluster_id = _env("REDSHIFT_CLUSTER_ID", str(rs_cfg.get("cluster_id", ""))) or ""
    redshift_workgroup_name = _env("REDSHIFT_WORKGROUP_NAME", str(rs_cfg.get("workgroup_name", ""))) or ""
    redshift_db_user = _env("REDSHIFT_DB_USER", str(rs_cfg.get("db_user", ""))) or ""
    redshift_secret_arn = _env("REDSHIFT_SECRET_ARN", str(rs_cfg.get("secret_arn", ""))) or ""

    # ------------------------------ Query persistence ------------------------------
    obs_cfg = (cfg.get("observability") or {})
    qs_cfg = (obs_cfg.get("query_store") or {})

    query_store_backend = _env("QUERY_STORE_BACKEND", str(qs_cfg.get("backend", "none"))) or "none"
    query_store_dir = _env("QUERY_STORE_DIR", str(qs_cfg.get("dir", "data/query_logs"))) or "data/query_logs"

    query_store_mode = _env("DYNAMODB_MODE", _env("QUERY_STORE_MODE", str(qs_cfg.get("mode", "detailed")))) or "detailed"

    ddb_cfg = (qs_cfg.get("dynamodb") or {})
    dynamodb_table_name = _env("DYNAMODB_TABLE_NAME", str(ddb_cfg.get("table_name", ""))) or ""
    dynamodb_pk_name = _env("DYNAMODB_PK_NAME", str(ddb_cfg.get("pk_name", "session_id"))) or "session_id"
    # Allow PK-only tables: if DYNAMODB_SK_NAME is set to an empty string, keep it empty.
    _sk_env = os.environ.get("DYNAMODB_SK_NAME")
    if _sk_env is not None:
        dynamodb_sk_name = _sk_env
    else:
        dynamodb_sk_name = _env("DYNAMODB_SK_NAME", str(ddb_cfg.get("sk_name", "sk"))) or "sk"
    dynamodb_region = _env("DYNAMODB_REGION", str(ddb_cfg.get("region", aws_region))) or aws_region
    dynamodb_endpoint_url = _env("DYNAMODB_ENDPOINT_URL", str(ddb_cfg.get("endpoint_url", ""))) or ""
    dynamodb_ttl_attribute = _env("DYNAMODB_TTL_ATTRIBUTE", str(ddb_cfg.get("ttl_attribute", "expires_at"))) or "expires_at"
    dynamodb_ttl_seconds = int(_env("DYNAMODB_TTL_SECONDS", str(ddb_cfg.get("ttl_seconds", 30 * 24 * 3600))))

    return Settings(
        env=app_env,
        log_level=_env("LOG_LEVEL", cfg["app"]["log_level"]),
        delimiter=delimiter,
        fallback_encodings=fallback_encodings,
        skip_bad_lines=skip_bad_lines,
        max_rows_preview=max_rows_preview,
        vector_backend=vector_backend,
        chroma_dir=chroma_dir,
        rag_top_k=rag_top_k,
        rag_min_score=rag_min_score,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,

        rag_oversample_factor=rag_oversample_factor,
        rag_enable_rerank=rag_enable_rerank,
        rag_schema_only_for_planning=rag_schema_only_for_planning,
        rag_schema_kinds=rag_schema_kinds,
        rag_enforce_candidate_allowlist=rag_enforce_candidate_allowlist,

        col_resolver_max_candidates=col_resolver_max_candidates,
        s3_vector_bucket=s3_vector_bucket,
        s3_vector_prefix=s3_vector_prefix,
        s3_vector_cache_dir=s3_vector_cache_dir,
        s3_vector_refresh_seconds=s3_vector_refresh_seconds,
        s3vectors_bucket=s3vectors_bucket,
        s3vectors_index=s3vectors_index,
        s3vectors_namespace=s3vectors_namespace,
        s3vectors_refresh_seconds=s3vectors_refresh_seconds,
        agent_min_confidence=agent_min_confidence,
        refuse_below_confidence=refuse_below_confidence,
        use_mock_bedrock=use_mock_bedrock,
        aws_region=aws_region,
        bedrock_chat_model_id=bedrock_chat_model_id,
        bedrock_embed_model_id=bedrock_embed_model_id,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        export_dir=export_dir,
        saved_query_dir=saved_query_dir,

        db_type=db_type,
        athena_database=athena_database,
        athena_workgroup=athena_workgroup,
        athena_output_location=athena_output_location,
        athena_catalog=athena_catalog,
        redshift_database=redshift_database,
        redshift_cluster_id=redshift_cluster_id,
        redshift_workgroup_name=redshift_workgroup_name,
        redshift_db_user=redshift_db_user,
        redshift_secret_arn=redshift_secret_arn,

        query_store_backend=query_store_backend,
        query_store_dir=query_store_dir,
        query_store_mode=query_store_mode,
        dynamodb_table_name=dynamodb_table_name,
        dynamodb_pk_name=dynamodb_pk_name,
        dynamodb_sk_name=dynamodb_sk_name,
        dynamodb_region=dynamodb_region,
        dynamodb_endpoint_url=dynamodb_endpoint_url,
        dynamodb_ttl_attribute=dynamodb_ttl_attribute,
        dynamodb_ttl_seconds=dynamodb_ttl_seconds,
    )