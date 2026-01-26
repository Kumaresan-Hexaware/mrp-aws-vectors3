- 
- 
- 
- 
- 
- **Natural‑language Analytics** → dynamic **tabular reports** from uploaded **Ç‑delimited `.nzf`** (and other delimited text) files.
- **Natural‑language Dashboards** → interactive **Plotly dashboards** (KPIs, charts, filters) generated from plain‑English questions.
- **Strict grounding** → outputs are computed from uploaded data + schema registry only. If required data is missing/insufficient, the app responds with: **`Insufficient data.`**

This repository supports:
- **Amazon Bedrock** (chat + embeddings)
- **Amazon S3 Vectors** (native S3 Vector Buckets) as the vector store backend
- **Mock mode** for local development without AWS calls

---

## What you get

### Reports (Tabular)
- Upload one or more `.nzf` files (or delimited text)
- Ask a business question in plain English
- The system generates an ad‑hoc tabular report (grounded in your data)
- Export options:
  - **CSV**
  - **XML**
  - **PDF** (first 200 rows for safety)

### Dashboards (Plotly)
- Ask a question to generate a dashboard plan + figures
- Plotly interactivity (zoom, pan, hover tooltips)
- Save queries so dashboards are **regenerable**
- Saved plans stored at: `data/saved_queries/*.json`

### Safety / No Hallucination
The system returns **`Insufficient data.`** when:
- Required tables are not uploaded
- Retrieval confidence is below threshold
- A proposed plan references tables/columns/joins not present in the schema registry
- Execution returns an empty result set after filters/grouping

---

## Architecture

**Pipeline (RAG + Tool‑Augmented ReAct):**

1) **Ingestion**
- Reads `.nzf` using configurable delimiter (default `Ç`)
- Fallback encodings (e.g., `utf‑8` → `latin‑1`)
- Optional “skip bad lines” mode for malformed rows

2) **Schema Registry (Hard Rules)**
- Logical table definitions
- File pattern mapping: uploaded file → logical table
- Column types + descriptions
- Allowed join rules (keys, join types)
- **Joins are never inferred by the LLM**; only registry-defined join paths are allowed

3) **RAG (Schema Grounding)**
- Indexes schema descriptions + join rules into the selected vector backend
- Retrieves the most relevant schema context per user question

4) **Planning (LLM)**
- LLM produces a strict **JSON query plan** (tables, joins, filters, aggregations, chart spec)

5) **Validation**
- Plan is validated against the schema registry (tables/cols/joins/aggregations)
- If invalid → **Insufficient data** (with a helpful reason in logs)

6) **Execution**
- Executes safe SQL using **DuckDB**
- Registry-defined join path only

7) **Visualization & Export**
- Plotly figure rendering (fallback to table if figure render fails)
- Report exports: CSV/XML/PDF

---

## Project layout

```
.
├─ config/                       # env-specific configs (dev/test/acpt/prod)
├─ schemas/                      # schema registry YAML
├─ scripts/
│  └─ streamlit_app.py           # Streamlit UI entrypoint
├─ src/nl_analytics/
│  ├─ agents/                    # orchestrator (tool flow)
│  ├─ bedrock/                   # Bedrock wrapper
│  ├─ config/                    # settings loader
│  ├─ data/                      # in-memory session + table store
│  ├─ exceptions/                # custom exceptions
│  ├─ export/                    # CSV/XML/PDF + saved query storage
│  ├─ ingestion/                 # file reader + file-to-table mapper
│  ├─ preprocessing/             # cleaning + type coercion
│  ├─ rag/                       # vector store backends (s3vectors/chroma)
│  ├─ schema/                    # registry loader + join reasoning
│  ├─ tools/                     # retrieval / planning / execution tools
│  ├─ viz/                       # plotly figure factory
│  └─ logging/                   # logger init
│
├─ data/
│  ├─ sample/                    # sample nzf
│  ├─ exports/                   # exported files
│  └─ saved_queries/             # saved dashboard plans
└─ logs/
```

> **Note:** This repo uses a `src/` layout. Run commands from the repo root.

---

## Prerequisites

- Python **3.10+** (3.11 recommended)
- AWS access (only if using Bedrock / S3 Vectors)
- If using S3 Vectors:
  - A **Vector Bucket** and an **Index** created in the target region
  - Index dimension must match your embedding dimension (details below)

---

## Quick start (Local)

### 1) Create & activate a virtual environment
```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (cmd)
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Configure environment
Create `.env` (or copy from `.env.example` if present):

**Local dev without AWS calls**
```env
APP_ENV=dev
USE_MOCK_BEDROCK=1
VECTOR_BACKEND=chroma
```

### 4) Run the Streamlit app
```bash
streamlit run scripts/streamlit_app.py
```

### 5) Quick test
Upload:
- `data/sample/sample_sales.nzf`

Then ask:
- **Report:** `total sales by region`
- **Dashboard:** `dashboard of total sales over time`

---

## Configuration model

- `APP_ENV=dev|test|acpt|prod` loads `config/<env>.yaml`
- Environment variables override YAML values

### Common environment variables

#### Ingestion
- `DEFAULT_DELIMITER` (default `Ç`)
- `FALLBACK_ENCODINGS` (e.g., `utf-8,latin-1`)
- `SKIP_BAD_LINES` (`1`/`0`)

#### RAG / Retrieval
- `VECTOR_BACKEND` (`chroma` | `s3vectors`)
- `RAG_TOP_K` (default `8`)
- `RAG_MIN_SCORE` (default `0.25`)

#### Agent
- `AGENT_MIN_CONFIDENCE` (default `0.45`)
- `LLM_TEMPERATURE` (default `0.1`)
- `LLM_MAX_TOKENS` (default `1200`)

#### Export
- `EXPORT_DIR` (default `data/exports`)
- `SAVED_QUERY_DIR` (default `data/saved_queries`)

---

## Using Amazon Bedrock

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
- environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`)
- AWS SSO
- instance/task role (EC2/ECS)

---

## Using Amazon S3 Vectors (native vector buckets)

### 1) Enable S3 Vectors backend
Set in `.env`:

```env
VECTOR_BACKEND=s3vectors
S3VECTORS_BUCKET=your-vector-bucket
S3VECTORS_INDEX=your-index-name
S3VECTORS_NAMESPACE=default
AWS_REGION=us-east-1
USE_MOCK_BEDROCK=0
```

> You can also set `S3VECTORS_INDEX` to an **index ARN** instead of a name.

### 2) Dimension alignment (important)
S3 Vectors requires the vector length to match the index dimension exactly.

- If your index is **1024‑dim**, your embedding output must be **1024‑dim**
- If your index is **512‑dim**, your embedding output must be **512‑dim**, etc.

This repo’s S3 Vectors backend and CLI tester are designed to:
- fetch the index dimension (when permitted)
- request embeddings at that dimension when the embedding model supports it (e.g., Titan v2)

### 3) IAM permissions (typical)
- `bedrock:InvokeModel` (embedding + chat)
- `s3vectors:PutVectors` (write)
- `s3vectors:QueryVectors` (search)
- `s3vectors:GetVectors` / `s3vectors:GetIndex` (debug/verification)

---

## End‑to‑end vector ingestion test (CLI)

Use the helper to validate **Bedrock → S3 Vectors** independently of the GUI:

```bash
python -m nl_analytics.tools.vector_upload_tester --file path/to/yourfile.nzf
```

Optional region override (if the vector bucket/index is in a different region from Bedrock):
```bash
python -m nl_analytics.tools.vector_upload_tester --file path/to/yourfile.nzf --s3vectors-region us-west-2
```

The tester prints either a success message or the exact AWS exception (permissions, region, validation, etc.).

---

## Schema registry

File:
- `schemas/schema_registry.yaml`

Defines:
- tables
- file patterns used to map uploaded files → logical tables
- column types and descriptions
- join rules

**Important:** If a join is not defined in the registry, it cannot be used.

---

## Logging & observability

Log file:
- `logs/app.log`

Logged operations include:
- upload / ingestion
- schema mapping
- retrieval scores
- plan validation
- query execution
- chart rendering
- exports
- full exception traces (including S3 Vectors ingestion failures)

---

## Troubleshooting

### Vectors not appearing in S3 Vectors
1) Run the CLI tester:
   ```bash
   python -m nl_analytics.tools.vector_upload_tester --file path/to/yourfile.nzf
   ```
2) Check `logs/app.log` for the exact exception.
3) Most common causes:
   - index dimension mismatch
   - wrong region for the `s3vectors` client
   - missing IAM permissions (`s3vectors:PutVectors`)
   - outdated boto3 that doesn’t support `s3vectors`

### “Insufficient data.”
Typical causes:
- required data files weren’t uploaded / mapped to registry tables
- question references columns not present in registry
- confidence below `AGENT_MIN_CONFIDENCE`
- query executes but result is empty after filtering/grouping

### Delimiter shows as `Ã‡`
Usually encoding mismatch. Ensure:
- `DEFAULT_DELIMITER=Ç`
- `FALLBACK_ENCODINGS` includes `latin-1` if your files aren’t UTF‑8

---

## Athena execution engine (workable setup)

When `DB_TYPE=athena`, this app compiles SQL and executes it in **Amazon Athena**, then reads the CSV result back from the Athena output S3 location.

### AWS setup steps

1) **Create S3 buckets**
   - Data bucket: stores your table files (recommended: Parquet)
   - Results bucket/prefix: Athena query output (example: `s3://<RESULTS_BUCKET>/athena-results/`)

2) **Put table data in S3** (recommended layout)
   - `s3://<DATA_BUCKET>/mrp/pvr00600/ingest_dt=2026-01-25/part-000.parquet`
   - `s3://<DATA_BUCKET>/mrp/pvr00500/ingest_dt=2026-01-25/part-000.parquet`

   *Why Parquet:* much faster + cheaper than scanning big delimited text.

3) **Create Glue database + tables**
   - Create a Glue database (example: `mrp_db`)
   - Use a Glue crawler (or DDL) to create tables matching your registry table names (e.g., `pvr00600`, `pvr00500`)
   - Ensure column names match the registry (case-insensitive is usually OK in Athena)

4) **Grant IAM permissions** to the role/user running the app
   - `athena:StartQueryExecution`, `athena:GetQueryExecution`, `athena:GetWorkGroup`, `athena:GetQueryResults`
   - `glue:GetDatabase`, `glue:GetTables`, `glue:GetTable`, `glue:GetPartitions`
   - `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` on both data + results locations

5) **Configure the app**
   - In `config/<env>.yaml` set:
     - `database.db_type: "athena"`
     - `database.athena.database: "mrp_db"`
     - `database.athena.output_location: "s3://<RESULTS_BUCKET>/athena-results/"`
     - optionally `database.athena.workgroup`

   Or use env vars:
   - `DB_TYPE=athena`
   - `ATHENA_DATABASE=mrp_db`
   - `ATHENA_OUTPUT_LOCATION=s3://<RESULTS_BUCKET>/athena-results/`
   - `ATHENA_WORKGROUP=<optional>`

### Notes / gotchas

- The Streamlit "Workspace" upload is still useful for **previewing** data locally, but Athena execution requires the same tables to exist in Glue/Athena.
- If your source `.nzf` is large, convert to Parquet once (ETL) and query Parquet thereafter.
- Partitioning (e.g. `ingest_dt=YYYY-MM-DD`) can drastically reduce scan cost.

## Production hardening checklist (recommended)

- Persist standardized tables as Parquet to avoid reprocessing on every restart
- Stream ingestion for very large files (chunked reads + chunk embeddings)
- Add request tracing/correlation IDs (UI → ingestion → retrieval → execution)
- Add authn/authz if used beyond a trusted network
- Add structured logging + centralized log shipping (CloudWatch/ELK)
- Add rate-limits and cost guardrails for LLM calls

---

## License

Internal prototype for extension and evaluation.


### Wide-schema accuracy (500+ columns)
- Column-level schema chunks are indexed for precise matching.
- Candidate-column allowlist enforcement prevents the planner from selecting arbitrary columns.
  - Config: `rag.enforce_candidate_allowlist: true`
  - Env: `RAG_ENFORCE_ALLOWLIST=true`
