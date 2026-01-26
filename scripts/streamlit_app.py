from __future__ import annotations
import sys
from pathlib import Path as _Path

# -----------------------------------------------------------------------------
# Streamlit entrypoint (UI-focused).
# NOTE: This file is intentionally kept as a thin UI layer over existing
# nl_analytics modules. Business logic, services, and data behavior remain
# unchanged.
# -----------------------------------------------------------------------------

_ROOT = _Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import time
from pathlib import Path
import streamlit as st
import traceback
import io
import zipfile
from datetime import datetime
import pandas as pd
import plotly.io as pio

from nl_analytics.config.settings import load_settings
from nl_analytics.logging.logger import init_logging
from nl_analytics.schema.registry import SchemaRegistry
from nl_analytics.ingestion.reader import read_nzf
from nl_analytics.ingestion.mapper import map_file_to_table
from nl_analytics.preprocessing.cleaning import standardize_columns, coerce_types
from nl_analytics.data.session import DataSession
from nl_analytics.agents.orchestrator import AgentOrchestrator, INSUFFICIENT
from nl_analytics.export.exporter import export_report
from nl_analytics.export.saved_queries import save_query, list_queries, load_query


# ------------------------------ Page config ----------------------------------

st.set_page_config(
    page_title="NL Analytics Agentic Prototype",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_global_style() -> None:
    """Inject a modern, enterprise-style theme with consistent spacing.

    Streamlit doesn't ship with a full design system out-of-the-box, so we use a
    small amount of CSS to improve typography, hierarchy, and interaction states.
    """
    st.markdown(
        """
        <style>
          /* Import Inter + Material Icons for a modern SaaS look */
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
          @import url('https://fonts.googleapis.com/icon?family=Material+Icons');

          :root {
            --bg: #f6f8fc;
            --panel: #ffffff;
            --panel2: #f8fafc;
            --stroke: #e2e8f0;
            --text: #0f172a;
            --muted: #334155;
            --muted2: #64748b;

            --brand: #2563eb;
            --ok: #16a34a;
            --warn: #d97706;
            --bad: #dc2626;

            --r: 16px;
            --r2: 20px;
            --shadow: 0 10px 24px rgba(2,6,23,.08);
          }

          html, body, [class*="css"]  {
            font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
          }

          /* Main app background */
          .stApp {
            background: var(--bg);
            color: var(--text);
          }


          /* Sidebar */
          section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--stroke);
          }

          /* Reduce Streamlit top padding a bit */
          .block-container { padding-top: 1.15rem; padding-bottom: 2.5rem; }

          /* Headings */
          h1, h2, h3, h4 { letter-spacing: -0.02em; }
          h1 { font-size: 1.55rem; }
          h2 { font-size: 1.25rem; }
          h3 { font-size: 1.05rem; }
          p, label, span { color: var(--text); }

          /* Cards */
          .mrp-card {
            background: var(--panel);
            border: 1px solid var(--stroke);
            border-radius: var(--r2);
            padding: 1rem 1.05rem;
            box-shadow: var(--shadow);
          }
          .mrp-card.soft {
            background: var(--panel2);
            box-shadow: none;
          }

          .mrp-kpi {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: .75rem;
          }
          .mrp-kpi .label { color: var(--muted2); font-size: .82rem; }
          .mrp-kpi .value { font-weight: 700; font-size: 1.25rem; margin-top: .15rem; }
          .mrp-kpi .chip {
            display: inline-flex;
            align-items: center;
            gap: .35rem;
            padding: .22rem .55rem;
            border-radius: 999px;
            border: 1px solid var(--stroke);
            color: var(--muted);
            font-size: .75rem;
            background: rgba(0,0,0,.10);
            white-space: nowrap;
          }

          /* Buttons */
          .stButton button {
            border-radius: 12px;
            border: 1px solid var(--stroke);
            background: #ffffff;
            color: var(--text);
            padding: .55rem .85rem;
            font-weight: 600;
            transition: transform .08s ease, background .18s ease, border-color .18s ease;
          }
          .stButton button:hover {
            background: #f1f5f9;
            border-color: rgba(37,99,235,.45);
            transform: translateY(-1px);
          }
          .stButton button:active { transform: translateY(0px); }


          /* Inputs */
          .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
            border-radius: 12px !important;
            border: 1px solid var(--stroke) !important;
            background: #ffffff !important;
          }


          /* Tabs */
          button[data-baseweb="tab"] {
            font-weight: 600;
          }

          /* Dataframe container */
          .stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid var(--stroke); }

          /* Alerts - soften */
          div[data-testid="stAlert"] {
            border-radius: 14px;
            border: 1px solid var(--stroke);
            background: #ffffff;
          }


          /* Hide Streamlit watermark/footer */
          footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _material_icon(name: str) -> str:
    return f"<span class='material-icons' style='font-size: 18px; line-height: 1; opacity:.9'>{name}</span>"


def _card_kpi(label: str, value: str, chip: str | None = None) -> None:
    chip_html = f"<span class='chip'>{_material_icon('insights')} {chip}</span>" if chip else ""
    st.markdown(
        f"""
        <div class="mrp-card soft">
          <div class="mrp-kpi">
            <div>
              <div class="label">{label}</div>
              <div class="value">{value}</div>
            </div>
            {chip_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------- Existing ingestion logic --------------------------

def _auto_load_previous_uploads() -> int:
    """Load any existing files in data/uploads into workspace (in-memory).

    This behavior is intentionally preserved to keep the original UX intact.
    """
    uploads_dir = _ROOT / "data" / "uploads"
    if not uploads_dir.exists():
        return 0
    files = sorted(list(uploads_dir.glob("*.nzf")))
    if not files:
        return 0

    n = 0
    for p in files:
        try:
            _ingest_file_path(p, original_name=p.name)
            n += 1
        except Exception:
            # Don't fail the app if one file cannot be loaded.
            continue
    return n


def _ingest_file_path(path: Path, original_name: str | None = None) -> None:
    """Ingest a file already present on disk into the in-memory DataSession."""
    sess: DataSession = st.session_state["data_session"]
    name = original_name or path.name

    # Use configured delimiter/encodings from Settings (business behavior unchanged)
    ing = read_nzf(str(path), settings.delimiter, settings.fallback_encodings, skip_bad_lines=settings.skip_bad_lines)
    df, encoding = ing.df, ing.encoding_used
    table = map_file_to_table(registry, name)
    df = standardize_columns(df)
    # coerce_types expects a {column: type} mapping, not the registry itself.
    # Keep business behavior the same by deriving the mapping from the registry spec.
    column_types = {cname: cspec.type for cname, cspec in registry.get_table(table).columns.items()}
    df = coerce_types(df, column_types)

    sess.register_table(table, df, original_name or path.name)

    # Best-effort: upsert a capped preview into the vector store (idempotent IDs).
    try:
        max_rows = min(len(df), 2000)
        if max_rows > 0:
            ids, texts, metas = [], [], []
            preview_df = df.head(max_rows)
            for i, row in enumerate(preview_df.itertuples(index=False), start=1):
                row_dict = row._asdict() if hasattr(row, "_asdict") else dict(zip(preview_df.columns, row))
                parts = [f"{k}={row_dict.get(k)}" for k in list(preview_df.columns)[:50]]
                text = f"DATA ROW | table={table} | source={name} | " + " | ".join(parts)
                ids.append(f"data::{table}::{name}::{i}")
                texts.append(text)
                metas.append({"kind": "data", "table": table, "source_file": name, "row": i})
            orch.store.upsert(ids=ids, texts=texts, metadatas=metas)
    except Exception:
        # Don't block ingestion; vector store is optional.
        pass


def ingest_uploaded_files(files) -> None:
    """Ingest uploaded files: keep business behavior identical, improve UX (progress + messaging)."""
    sess: DataSession = st.session_state["data_session"]

    progress = st.progress(0, text="Preparing ingestion…")
    total = len(files)

    for idx, f in enumerate(files, start=1):
        try:
            # Persist to data/uploads for restart-friendly UX.
            uploads_dir = _ROOT / "data" / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            out_path = uploads_dir / f.name
            out_path.write_bytes(f.getvalue())

            _ingest_file_path(out_path, original_name=f.name)

            st.success(f"Ingested {f.name}")
        except Exception as e:
            st.error(f"Ingestion failed for {getattr(f, 'name', 'file')}: {e}")
            st.code(traceback.format_exc())

        progress.progress(int(idx / max(total, 1) * 100), text=f"Ingesting files… ({idx}/{total})")

    progress.empty()


# ------------------------------ Bootstrap ------------------------------------

@st.cache_resource
def bootstrap():
    settings = load_settings()
    init_logging(settings.log_level)
    registry = SchemaRegistry.load("schemas/schema_registry.yaml")
    orch = AgentOrchestrator(settings, registry)
    return settings, registry, orch


try:
    settings, registry, orch = bootstrap()
except Exception:
    st.error("Startup failed. See error below.")
    st.code(traceback.format_exc())
    st.stop()

if "data_session" not in st.session_state:
    st.session_state["data_session"] = DataSession(registry=registry, settings=settings)

if "last_plan" not in st.session_state:
    st.session_state["last_plan"] = None

_inject_global_style()


# ------------------------------ Sidebar --------------------------------------

with st.sidebar:
    st.markdown(
        f"""
        <div class="mrp-card">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:.75rem;">
            <div style="display:flex;align-items:center;gap:.6rem;">
              <div style="width:36px;height:36px;border-radius:12px;background:rgba(122,162,255,.18);display:flex;align-items:center;justify-content:center;border:1px solid rgba(255,255,255,.10);">
                {_material_icon('query_stats')}
              </div>
              <div>
                <div style="font-weight:800;font-size:1.02rem;line-height:1.05;">NL Analytics</div>
                <div style="color:var(--muted2);font-size:.78rem;margin-top:.12rem;">RAG + ReAct prototype</div>
              </div>
            </div>
            <div style="padding:.22rem .55rem;border-radius:999px;border:1px solid var(--stroke);background:rgba(0,0,0,.10);color:var(--muted);font-size:.75rem;">v1</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    nav = st.radio(
        "Navigation",
        ["Workspace", "Reports", "Dashboards"],
        index=0,
        help="Choose a workspace view (files), reports, or dashboards.",
    )

    st.write("")

    sess: DataSession = st.session_state["data_session"]
    _upl_dir = _ROOT / "data" / "uploads"
    ws_file_count = len(list(_upl_dir.glob("*"))) if _upl_dir.exists() else 0
    _card_kpi("Workspace files", str(ws_file_count), chip="Uploads")
    st.write("")
    loaded_tables = sess.available_tables()
    _card_kpi("Loaded tables", str(len(loaded_tables)), chip="Workspace")
    st.write("")

    # A lightweight status summary (purely UI).
    vector_status = "Connected" if getattr(orch, "store", None) is not None else "Unavailable"
    _card_kpi("Vector store", vector_status, chip="RAG Index")

    st.write("")

    with st.expander("Tables", expanded=False):
        if not loaded_tables:
            st.caption("No tables in workspace yet.")
        else:
            for t in loaded_tables:
                st.markdown(f"• **{t}**")

    with st.expander("Diagnostics", expanded=False):
        st.caption("UI diagnostics (does not affect business logic).")
        st.json(st.session_state.get("last_plan") or {})
        st.caption("Session state keys")
        st.write(list(st.session_state.keys()))

    st.write("")

    st.caption("Tip: Upload .nzf files in **Workspace**. Use **Reports**/**Dashboards** to run NL queries.")


# ------------------------------ Main views -----------------------------------

# One-time auto-load from disk on fresh UI session (preserved behavior).
if "auto_loaded_uploads" not in st.session_state:
    st.session_state["auto_loaded_uploads"] = True
    try:
        n = _auto_load_previous_uploads()
        if n:
            st.toast(f"Auto-loaded {n} .nzf file(s) from data/uploads", icon="✅")
    except Exception:
        st.warning("Auto-load of previous uploads failed. You can re-upload the file(s) to continue.")


def _page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div style="display:flex;align-items:flex-end;justify-content:space-between;gap:1rem;margin-bottom:.85rem;">
          <div>
            <div style="font-size:1.45rem;font-weight:800;letter-spacing:-0.02em;">{title}</div>
            <div style="color:var(--muted2);margin-top:.15rem;">{subtitle}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



# ------------------------------ Workspace helpers ------------------------------

def _format_bytes(n: int) -> str:
    if n is None:
        return "-"
    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for u in units:
        if size < step:
            return f"{size:.1f} {u}"
        size /= step
    return f"{size:.1f} PB"


def _list_workspace_files() -> list[dict]:
    """Return workspace files stored under data/uploads (UI helper)."""
    uploads_dir = _ROOT / "data" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(uploads_dir.glob("*")):
        if not p.is_file():
            continue
        stat = p.stat()
        out.append(
            {
                "name": p.name,
                "path": str(p),
                "size": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime),
                "suffix": p.suffix.lower().lstrip("."),
            }
        )
    return out


def _ensure_workspace_state() -> None:
    if "ws_selected_file" not in st.session_state:
        st.session_state["ws_selected_file"] = None
    if "last_report_result" not in st.session_state:
        st.session_state["last_report_result"] = None
    if "last_dashboard_result" not in st.session_state:
        st.session_state["last_dashboard_result"] = None


def _read_nzf_preview(path: str, nrows: int = 50):
    """Fast preview for .nzf without ingesting the full file (UI-only)."""
    p = Path(path)
    # Reuse configured delimiter/encodings for consistent previews.
    sep = getattr(settings, "delimiter", "Ç")
    encs = list(getattr(settings, "fallback_encodings", ["utf-8", "cp1252", "latin-1"]))
    # Keep preview snappy: try the first few encodings first.
    encs = encs[:5] if encs else ["utf-8", "cp1252", "latin-1"]

    last_err = None
    for enc in encs:
        try:
            df = pd.read_csv(
                p,
                sep=sep,
                encoding=enc,
                engine="python",
                on_bad_lines="skip",
                nrows=nrows,
            )
            return df, enc
        except Exception as e:
            last_err = e
    raise last_err


def _excel_bytes_from_df(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="Report")
    bio.seek(0)
    return bio.read()


def _zip_plotly_images(figs: list, fmt: str = "png") -> bytes:
    """Export one or many Plotly figures to PNG/PDF. Returns bytes."""
    if not figs:
        return b""
    if len(figs) == 1:
        return pio.to_image(figs[0], format=fmt, scale=2 if fmt == "png" else 1)
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, fig in enumerate(figs, start=1):
            try:
                data = pio.to_image(fig, format=fmt, scale=2 if fmt == "png" else 1)
                zf.writestr(f"panel_{i:02d}.{fmt}", data)
            except Exception:
                # Best-effort: skip panels that cannot be exported
                continue
    bio.seek(0)
    return bio.read()


def _dashboard_html_bytes(figs: list) -> bytes:
    parts = []
    for i, fig in enumerate(figs or [], start=1):
        try:
            parts.append(f"<h3>Panel {i}</h3>")
            parts.append(fig.to_html(include_plotlyjs="cdn", full_html=False))
            parts.append("<hr/>")
        except Exception:
            continue
    html = "<html><head><meta charset='utf-8'/><title>Dashboard</title></head><body>" + "\n".join(parts) + "</body></html>"
    return html.encode("utf-8")


# ------------------------------ Views ----------------------------------------

def _workspace_view() -> None:
    _ensure_workspace_state()
    _page_header(
        "Workspace",
        "Upload, manage, and preview files. Reports & Dashboards use only what’s loaded here.",
    )

    sess: DataSession = st.session_state["data_session"]
    files_on_disk = _list_workspace_files()

    # Top status
    st.markdown('<div class="mrp-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1], gap="medium")
    c1.metric("Workspace files", str(len(files_on_disk)))
    c2.metric("Loaded tables", str(len(sess.available_tables())))
    vector_status = "Connected" if getattr(orch, "store", None) is not None else "Unavailable"
    c3.metric("Vector store", vector_status)
    st.caption("✅ Files available for Reports & Dashboards once ingested into the Workspace.")
    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns([1.25, 1], gap="large")

    # Upload (single location for ALL uploads)
    with left:
        st.markdown('<div class="mrp-card">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload files")
        st.caption("Uploads are stored under `data/uploads` for restart-friendly sessions.")
        upl = st.file_uploader(
            "Upload files",
            type=["nzf", "png", "jpg", "jpeg", "webp", "pdf", "txt", "csv", "xlsx"],
            accept_multiple_files=True,
            key="ws_upl",
            help="Upload here only. Reports & Dashboards pull from Workspace files.",
        )
        if upl:
            with st.spinner("Ingesting uploads into Workspace…"):
                ingest_uploaded_files(upl)
            # Refresh file list after ingest
            files_on_disk = _list_workspace_files()

        st.markdown("---")
        st.markdown("### 📁 Workspace files")

        if not files_on_disk:
            st.info("No files uploaded yet. Add a file above to begin.")
        else:
            header = st.columns([2.2, 1, 1.2, 1.6], gap="small")
            header[0].markdown("**File name**")
            header[1].markdown("**Size**")
            header[2].markdown("**Upload date**")
            header[3].markdown("**Actions**")

            for f in files_on_disk:
                row = st.columns([2.2, 1, 1.2, 1.6], gap="small")
                row[0].write(f["name"])
                row[1].write(_format_bytes(f["size"]))
                row[2].write(f["mtime"].strftime("%Y-%m-%d %H:%M"))
                bprev, bdel = row[3].columns([1, 1], gap="small")
                if bprev.button("Preview", key=f"prev::{f['name']}", help="Show preview on the right"):
                    st.session_state["ws_selected_file"] = f["path"]
                if bdel.button("Delete", key=f"del::{f['name']}", help="Delete from Workspace storage (data/uploads)"):
                    try:
                        Path(f["path"]).unlink(missing_ok=True)
                        st.toast(f"Deleted {f['name']}", icon="🗑️")
                        if st.session_state.get("ws_selected_file") == f["path"]:
                            st.session_state["ws_selected_file"] = None
                        files_on_disk = _list_workspace_files()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Notes", expanded=False):
            st.caption("Deleting a file removes it from Workspace storage. Already-loaded tables remain in memory until restart or workspace reset (business logic unchanged).")

    # Preview pane
    with right:
        st.markdown('<div class="mrp-card">', unsafe_allow_html=True)
        st.markdown("### 👁️ Preview")
        selected = st.session_state.get("ws_selected_file")
        if not selected:
            st.info("Select **Preview** for a file to see it here.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        p = Path(selected)
        if not p.exists():
            st.warning("Selected file no longer exists.")
            st.session_state["ws_selected_file"] = None
            st.markdown("</div>", unsafe_allow_html=True)
            return

        st.caption(f"**{p.name}** • {_format_bytes(p.stat().st_size)} • {datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}")
        suffix = p.suffix.lower().lstrip(".")

        # Mapping + quick preview
        if suffix == "nzf":
            try:
                table = map_file_to_table(p.name, registry)
                st.caption(f"Mapped table: **{table}**")
            except Exception:
                table = None

            try:
                dfp, enc = _read_nzf_preview(str(p), nrows=60)
                st.caption(f"Preview encoding: `{enc}` • showing first {min(len(dfp), 60)} rows")
                st.dataframe(dfp, height=420)
            except Exception as e:
                st.error(f"Preview failed: {e}")
                st.code(traceback.format_exc())
        elif suffix in {"png", "jpg", "jpeg", "webp"}:
            st.image(str(p), use_container_width=True)
        else:
            # Generic text preview (best-effort)
            try:
                raw = p.read_bytes()
                st.code(raw[:20000].decode("utf-8", errors="replace"))
            except Exception as e:
                st.error(f"Preview failed: {e}")

        st.markdown("</div>", unsafe_allow_html=True)


def _reports_view() -> None:
    _ensure_workspace_state()

    _page_header(
        "Generate Reports from Workspace Files",
        "Ask a question and export results. Uploads happen only in Workspace.",
    )

    sess: DataSession = st.session_state["data_session"]
    files_on_disk = _list_workspace_files()

    st.info(f"Files loaded in Workspace: **{len(files_on_disk)}** file(s)")

    st.markdown('<div class="mrp-card">', unsafe_allow_html=True)
    q = st.text_area(
        "Enter your report query:",
        key="rep_q",
        height=90,
        placeholder="e.g., Show total CurrentUPBAmt grouped by PortfolioID, sort descending",
        help="Use natural language. The agent will plan, validate, and execute against Workspace tables.",
    )
    run = st.button("Generate Report", key="rep_run", type="primary")

    if run:
        if (settings.db_type or "duckdb").strip().lower() == "duckdb" and not sess.available_tables():
            st.error("No tables loaded. Please upload and ingest your .nzf file first in **Workspace**.")
            st.stop()

        with st.spinner("Generating report…"):
            res = orch.run(sess, q, mode_hint="report")
        st.session_state["last_plan"] = res.plan
        st.session_state["last_report_result"] = res

        if not res.ok:
            st.error(INSUFFICIENT)
        else:
            st.success("Report generated.")
            st.markdown(f"**Confidence:** `{res.confidence:.3f}`")
            st.markdown("#### Results")
            st.dataframe(res.df, width='stretch')

            st.markdown("---")
            st.markdown("#### Downloads")
            # Keep existing export logic unchanged (CSV/PDF via exporter)
            # NOTE: nl_analytics.export.exporter.export_report expects a DataFrame, out_dir, and base_name.
            from datetime import datetime
            report_dir = str(Path(settings.export_dir) / "reports")
            base_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            exp = export_report(res.df, out_dir=report_dir, base_name=base_name)

            c1, c2, c3 = st.columns([1, 1, 1], gap="medium")

            # CSV
            if exp.csv_path and Path(exp.csv_path).exists():
                c1.download_button(
                    "CSV",
                    data=Path(exp.csv_path).read_bytes(),
                    file_name=Path(exp.csv_path).name,
                    help="Download report as CSV",
                )
            else:
                c1.download_button(
                    "CSV",
                    data=res.df.to_csv(index=False).encode("utf-8"),
                    file_name="report.csv",
                    help="Download report as CSV",
                )

            # Excel (UI-only addition)
            try:
                xlsx = _excel_bytes_from_df(res.df)
                c2.download_button(
                    "Excel",
                    data=xlsx,
                    file_name="report.xlsx",
                    help="Download report as Excel (.xlsx)",
                )
            except Exception as e:
                c2.caption(f"Excel export unavailable: {e}")

            # PDF
            if exp.pdf_path and Path(exp.pdf_path).exists():
                c3.download_button(
                    "PDF",
                    data=Path(exp.pdf_path).read_bytes(),
                    file_name=Path(exp.pdf_path).name,
                    help="Download report as PDF",
                )
            else:
                c3.caption("PDF not available")

    st.markdown("</div>", unsafe_allow_html=True)


def _dashboards_view() -> None:
    _ensure_workspace_state()

    _page_header(
        "Dashboards from Workspace Files",
        "Generate charts/KPIs from Workspace tables and export.",
    )

    sess: DataSession = st.session_state["data_session"]
    files_on_disk = _list_workspace_files()

    st.info(f"Files loaded in Workspace: **{len(files_on_disk)}** file(s)")

    st.markdown('<div class="mrp-card">', unsafe_allow_html=True)

    q = st.text_area(
        "Enter your dashboard query:",
        key="dash_q",
        height=90,
        placeholder="e.g., Dashboard with Total UPB by PortfolioID (bar) and a KPI for total UPB",
        help="Use natural language. The agent will decide the best charts and KPIs.",
    )

    b1, b2 = st.columns([1, 1], gap="medium")
    run = b1.button("Generate Dashboard", key="dash_run", type="primary")
    save = b2.button("Save query", key="dash_save", help="Save the last successful dashboard plan for later")

    if save:
        plan = st.session_state.get("last_plan")
        if not q or not plan:
            st.warning("Generate a dashboard first (so there is a plan to save).")
        else:
            qid = save_query(settings.saved_query_dir, mode="dashboard", question=q, plan=plan)
            st.success(f"Saved query id: {qid}")

    if run:
        if (settings.db_type or "duckdb").strip().lower() == "duckdb" and not sess.available_tables():
            st.error("No tables loaded. Please upload and ingest your .nzf file first in **Workspace**.")
            st.stop()

        with st.spinner("Generating dashboard…"):
            res = orch.run(sess, q, mode_hint="dashboard")
        st.session_state["last_plan"] = res.plan
        st.session_state["last_dashboard_result"] = res

        if not res.ok:
            st.error(INSUFFICIENT)
        else:
            st.success("Dashboard generated.")
            st.markdown(f"**Confidence:** `{res.confidence:.3f}`")

            figs = getattr(res, "figures", None) or ([] if res.figure is None else [res.figure])

            if figs:
                # Classify figures into KPI-like vs others (re-uses existing heuristic)
                kpis, others = [], []
                for f in figs:
                    try:
                        if getattr(f, "data", None) and len(getattr(f, "data", [])) == 1:
                            trace = f.data[0]
                            if getattr(trace, "type", "") in ("indicator",):
                                kpis.append(f)
                            else:
                                others.append(f)
                        else:
                            others.append(f)
                    except Exception:
                        others.append(f)

                left, right = st.columns([0.45, 1.55], gap="large")
                with left:
                    if kpis:
                        st.markdown("#### Metrics")
                        for k in kpis:
                            st.plotly_chart(k, use_container_width=True)
                    else:
                        st.markdown("#### Metrics")
                        st.caption("No KPI panels detected.")
                with right:
                    st.markdown("#### Charts")
                    for f in others:
                        st.plotly_chart(f, use_container_width=True)

                st.markdown("---")
                st.markdown("#### Downloads")
                c1, c2, c3 = st.columns([1, 1, 1], gap="medium")

                # PNG/PDF: single figure => direct, multi => zip
                png_bytes = _zip_plotly_images(figs, fmt="png")
                pdf_bytes = _zip_plotly_images(figs, fmt="pdf")
                html_bytes = _dashboard_html_bytes(figs)

                c1.download_button(
                    "PNG",
                    data=png_bytes,
                    file_name="dashboard.png" if len(figs) == 1 else "dashboard_png_panels.zip",
                    help="Download PNG (single chart) or ZIP (multiple panels)",
                )
                c2.download_button(
                    "PDF",
                    data=pdf_bytes,
                    file_name="dashboard.pdf" if len(figs) == 1 else "dashboard_pdf_panels.zip",
                    help="Download PDF (single chart) or ZIP (multiple panels)",
                )
                c3.download_button(
                    "Interactive HTML",
                    data=html_bytes,
                    file_name="dashboard.html",
                    help="Download an interactive HTML dashboard",
                )

            # If the agent returned a backing dataframe, show it for transparency
            if getattr(res, "df", None) is not None and hasattr(res.df, "head"):
                with st.expander("Underlying data (preview)", expanded=False):
                    st.dataframe(res.df, width='stretch')

    # Saved dashboards (preserves existing functionality without adding a 4th nav tab)
    with st.expander("Saved dashboards", expanded=False):
        saved = list_queries(settings.saved_query_dir)
        if not saved:
            st.caption("No saved dashboards yet.")
        else:
            options = {f"{s.id} — {s.question[:80]}": s.id for s in saved}
            sel = st.selectbox("Select a saved query", list(options.keys()), key="saved_sel")
            regen = st.button("Regenerate selected dashboard", key="saved_regen")
            if regen:
                qid = options[sel]
                sq = load_query(settings.saved_query_dir, qid)
                if (settings.db_type or "duckdb").strip().lower() == "duckdb" and not sess.available_tables():
                    st.error("No tables loaded. Please upload and ingest your .nzf file first in **Workspace**.")
                else:
                    with st.spinner("Regenerating dashboard…"):
                        res = orch.run(sess, sq.question, mode_hint="dashboard")
                    st.session_state["last_plan"] = res.plan
                    st.session_state["last_dashboard_result"] = res
                    if not res.ok:
                        st.error(INSUFFICIENT)
                    else:
                        st.success("Regenerated dashboard.")
                        figs = getattr(res, "figures", None) or ([] if res.figure is None else [res.figure])
                        for f in figs:
                            st.plotly_chart(f, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
def _saved_view() -> None:
    _page_header(
        "Saved dashboards",
        "Regenerate previously saved dashboard queries.",
    )

    st.markdown('<div class="mrp-card">', unsafe_allow_html=True)

    saved = list_queries(settings.saved_query_dir)
    if not saved:
        st.info("No saved queries yet.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    options = {f"{s.id} — {s.question[:80]}": s.id for s in saved}
    sel = st.selectbox("Select a saved query", list(options.keys()), key="saved_sel")
    regen = st.button("Regenerate selected dashboard", key="saved_regen")

    if regen:
        qid = options[sel]
        sq = load_query(settings.saved_query_dir, qid)
        sess = st.session_state["data_session"]
        if (settings.db_type or "duckdb").strip().lower() == "duckdb" and not sess.available_tables():
            st.error("No tables loaded. Please upload and ingest your .nzf file first.")
            st.stop()

        res = orch.run(sess, sq.question, mode_hint="dashboard")
        st.session_state["last_plan"] = res.plan
        if not res.ok:
            st.error(INSUFFICIENT)
        else:
            st.markdown(f"**Confidence:** `{res.confidence:.3f}`")
            if res.figure is not None:
                st.plotly_chart(res.figure, use_container_width=True)
            else:
                st.dataframe(res.df, width='stretch')

    st.markdown('</div>', unsafe_allow_html=True)


def _debug_view() -> None:
    _page_header(
        "Debug",
        "Developer utilities (no impact to business logic).",
    )

    with st.expander("Last Plan (JSON)", expanded=True):
        st.json(st.session_state.get("last_plan") or {})

    with st.expander("Session state keys", expanded=False):
        st.write(list(st.session_state.keys()))


# ------------------------------ Router ---------------------------------------

if nav == "Workspace":
    _workspace_view()
elif nav == "Reports":
    _reports_view()
else:
    _dashboards_view()
