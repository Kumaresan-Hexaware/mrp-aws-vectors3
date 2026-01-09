# NL Analytics Agentic Prototype (RAG + ReAct) — Streamlit + Plotly

A **production-ready, reusable prototype** for:

- **Natural-language Analytics** → ad-hoc **tabular reports** from uploaded **Ç-delimited `.nzf`** files (single or multiple with **registry-defined joins**).
- **Natural-language Dashboards** → interactive **Plotly dashboards** (charts, KPIs, filters) generated from simple English questions.
- **Strict grounding** → outputs are strictly based on uploaded data + schema registry. If data is missing/insufficient, the app returns: **`Insufficient data.`**

This repo includes a **Mock Bedrock mode** for local development and a Bedrock wrapper with **retries + circuit breaker** for production usage.

---

## Features

### Reports (Tabular)
- Upload one or more `.nzf` files (Ç-delimited)
- Ask a business question in plain English
- System generates an ad-hoc tabular report
- Export formats:
  - **CSV**
  - **XML**
  - **PDF** (first 200 rows for safety)

### Dashboards (Plotly)
- Ask a business question to generate a dashboard
- Plotly interactivity (zoom, pan, hover tooltips)
- Save queries so dashboards are **regenerable**
- Saved plans are stored in: `data/saved_queries/*.json`

### Safety / No Hallucination
The system refuses to answer and returns **`Insufficient data.`** when:
- Required tables are not uploaded
- Retrieval confidence is below threshold
- The LLM proposes tables/columns/joins not present in schema registry
- Execution returns an empty result set

---

## Architecture (RAG + Tool-Augmented ReAct)

1) **Ingestion**
- Reads `.nzf` with configurable delimiter (default `Ç`)
- Fallback encodings (`utf-8` → `latin-1`)
- Skip malformed rows (configurable)

2) **Preprocessing**
- Standardize column names
- Type coercion using schema registry (logs coercions/warnings)

3) **Schema Registry (Hard Rules)**
- Logical table definitions
- Table-to-file pattern mapping
- Column types/descriptions
- Join rules (join keys, join types)
- **Joins are NEVER inferred by the LLM**

4) **RAG (Schema Grounding)**
- Index schema descriptions + join rules in **ChromaDB**
- Retrieve relevant schema context per question

5) **Query Understanding**
- LLM returns a strict **JSON query plan**
- Plan is validated against registry (tables, columns, join path, allowed aggregations)

6) **Execution**
- Executes safe SQL using **DuckDB**
- Filters are parsed conservatively
- Joins follow the registry join path only

7) **Visualization & Export**
- Plotly figure rendering (fallback to table if render fails)
- Report exports: CSV/XML/PDF

---

## Project Layout

```
nl_analytics_agentic_prototype/
  config/                    # env-specific configs (dev/test/accpt/prod)
  schemas/                   # schema registry YAML
  scripts/
    streamlit_app.py         # Streamlit UI entrypoint
  src/nl_analytics/
    agents/                  # orchestrator (ReAct-style tool flow)
    bedrock/                 # Bedrock wrapper (retries + circuit breaker)
    config/                  # settings loader
    data/                    # in-memory session + table store
    exceptions/              # custom exceptions
    export/                  # CSV/XML/PDF + saved query storage
    ingestion/               # file reader + file-to-table mapper
    preprocessing/           # cleaning + type coercion
    rag/                     # vector store abstraction (Chroma wired)
    schema/                  # registry loader + join path reasoning
    tools/                   # retrieval / planning / execution tools
    viz/                     # plotly figure factory
    logging/                 # logger init
  data/
    sample/                  # sample nzf
    exports/                 # exported files
    saved_queries/           # saved dashboard plans
  logs/
```

---

## Setup Guide (Local)

### 1) Create venv
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
# source venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Configure environment
```bash
cp .env.example .env
```

Recommended for local (no AWS calls):
```env
APP_ENV=dev
USE_MOCK_BEDROCK=1
```

### 4) Run the app
```bash
streamlit run scripts/streamlit_app.py
```

### 5) Quick test
Upload:
- `data/sample/sample_sales.nzf`

Then ask:
- **Reports:** `total sales by region`
- **Dashboards:** `dashboard of total sales over time`

---

## Configuration

- `APP_ENV=dev|test|accpt|prod` loads `config/<env>.yaml`
- Environment variables override YAML

### Common env vars
**Ingestion**
- `DEFAULT_DELIMITER` (default `Ç`)
- `FALLBACK_ENCODINGS` (e.g., `utf-8,latin-1`)
- `SKIP_BAD_LINES` (1/0)


**RAG**
- `VECTOR_BACKEND` (`chroma` or `s3`)
- `CHROMA_DIR` (default `.chroma`)
- `RAG_TOP_K` (default `8`)
- `RAG_MIN_SCORE` (default `0.25`)

**S3 Vector Bucket backend (optional)**
- `S3_VECTOR_BUCKET` (required when `VECTOR_BACKEND=s3`)
- `S3_VECTOR_PREFIX` (default `nl-analytics/vectors`)
- `S3_VECTOR_CACHE_DIR` (default `.s3_vector_cache`)
- `S3_VECTOR_REFRESH_SECONDS` (default `300`)

**Agent**
- `AGENT_MIN_CONFIDENCE` (default `0.45`)
- `LLM_TEMPERATURE` (default `0.1`)
- `LLM_MAX_TOKENS` (default `1200`)

**Export**
- `EXPORT_DIR` (default `data/exports`)
- `SAVED_QUERY_DIR` (default `data/saved_queries`)

---

## Schema Registry

File:
- `schemas/schema_registry.yaml`

Defines:
- tables
- file patterns used to map uploaded files → logical tables
- column types and descriptions
- join rules

**Important:** If a join is not defined in the registry, it cannot be used.

---

## AWS Bedrock Setup

### 1) Disable mock mode
In `.env`:
```env
USE_MOCK_BEDROCK=0
AWS_REGION=us-east-1
BEDROCK_CHAT_MODEL_ID=<your_chat_model_id>
BEDROCK_EMBED_MODEL_ID=<your_embed_model_id>
```

### 2) Provide AWS credentials
Any standard method:
- `AWS_PROFILE`
- env vars (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`)
- SSO
- instance/task role (EC2/ECS)

### 3) Reliability included
- 3 retries with exponential backoff
- simple circuit breaker (opens after repeated failures and auto-recovers)

> Note: Bedrock request/response schemas vary by model provider. This prototype includes a simple JSON-returning wrapper; you may adapt the payload format to your chosen Bedrock model.

---

## Exports

### Reports
Exports are written to `EXPORT_DIR` and also provided as download buttons in the UI.

### Dashboards
Plotly is rendered in the UI.  
To add file exports (HTML/PNG/SVG/PDF), extend `src/nl_analytics/viz/` using `kaleido`.

---

## Logging

Log file:
- `logs/app.log`

Logged operations include:
- upload / ingestion
- schema mapping
- retrieval confidence
- plan validation
- query execution
- chart render
- export operations
- exception traces

---

## Troubleshooting

### “Insufficient data.”
Typical causes:
- required data files weren’t uploaded / mapped to registry tables
- question references columns not present in registry
- confidence below `AGENT_MIN_CONFIDENCE`
- query executes but result is empty after filtering/grouping

### Delimiter shows as `Ã‡`
This is almost always encoding mismatch. Ensure:
- `DEFAULT_DELIMITER=Ç`
- `FALLBACK_ENCODINGS` includes `latin-1` if your files aren’t UTF-8

### Reset vector store
Delete `.chroma/` and restart:
```bash
rm -rf .chroma
```

---

## Extending / Production Hardening

- Implement `PgVectorStore` and `S3VectorStore` in `src/nl_analytics/rag/vector_store.py`
- Add streaming ingestion for very large files
- Persist standardized tables as Parquet to avoid reprocessing
- Add stronger plan grammar + richer chart spec mapping for 20+ chart types

---

## License
Internal prototype for extension and evaluation.


## If you see `ModuleNotFoundError: No module named 'nl_analytics'`

This repo uses a `src/` layout. The Streamlit app auto-adds `src/` to `sys.path`. If you run other scripts, run from the repo root or add `src/` to `PYTHONPATH`.


## Chroma error: `_EmbedFn` has no attribute `name`
If you see an error like:
`AttributeError: '_EmbedFn' object has no attribute 'name'`

You are using a newer Chroma version that expects `name()`/`get_config()` on the embedding function.
This prototype includes that adapter. If you still see conflicts due to an old persisted collection,
delete the local Chroma directory and restart:
```bash
rm -rf .chroma
```


## Using Amazon S3 Vector Buckets (native S3 Vectors)

If your bucket is a **vector bucket** (visible via `aws s3vectors list-vector-buckets`), you must use the **S3 Vectors API** (not normal S3 PutObject).

Set in `.env`:

```env
VECTOR_BACKEND=s3vectors
S3VECTORS_BUCKET=your-vector-bucket
S3VECTORS_INDEX=your-index-name
S3VECTORS_NAMESPACE=default
```

Permissions needed (typical):
- `s3vectors:PutVectors` (write vectors)
- `s3vectors:QueryVectors` (search)
- `s3vectors:GetVectors` (required because this prototype queries with `returnMetadata=True`)

This prototype stores the chunk text inside vector metadata key `source_text` (AWS docs use the same approach).
