from __future__ import annotations
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import time
from pathlib import Path
import streamlit as st
import traceback

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

st.set_page_config(page_title="NL Analytics Agentic Prototype", layout="wide")

@st.cache_resource
def bootstrap():
    settings = load_settings()
    init_logging(settings.log_level)
    registry = SchemaRegistry.load("schemas/schema_registry.yaml")
    orch = AgentOrchestrator(settings, registry)
    return settings, registry, orch

try:
    settings, registry, orch = bootstrap()
except Exception as e:
    import traceback
    st.error("Startup failed. See error below.")
    st.code(traceback.format_exc())
    raise

if "data_session" not in st.session_state:
    st.session_state["data_session"] = DataSession(registry=registry)

if "last_plan" not in st.session_state:
    st.session_state["last_plan"] = None

def _ingest_file_path(path: Path, original_name: str | None = None) -> None:
    """Ingest a file already present on disk into the in-memory DataSession.

    This makes restarts user-friendly: data can be re-hydrated from data/uploads without re-upload.
    """
    sess: DataSession = st.session_state["data_session"]
    name = original_name or path.name
    table = map_file_to_table(registry, name)
    if not table:
        st.warning(f"File '{name}' did not match any registry table patterns.")
        return

    res = read_nzf(str(path), delimiter=settings.delimiter, fallback_encodings=settings.fallback_encodings, skip_bad_lines=settings.skip_bad_lines)
    df = standardize_columns(res.df)

    spec = registry.get_table(table)
    col_types = {c: spec.columns[c].type for c in spec.columns}
    df = coerce_types(df, col_types)

    sess.register_table(table, df, name)

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

def _auto_load_previous_uploads() -> int:
    """Load any .nzf files found under data/uploads into the current session."""
    upload_dir = Path("data/uploads")
    if not upload_dir.exists():
        return 0
    count = 0
    for p in sorted(upload_dir.glob("*.nzf")):
        try:
            _ingest_file_path(p)
            count += 1
        except Exception:
            # Keep going; we'll show a single warning outside.
            continue
    return count

def ingest_uploaded_files(files):
    sess: DataSession = st.session_state["data_session"]
    for f in files:
        tmp_dir = Path("data/uploads")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f.name
        tmp_path.write_bytes(f.getbuffer())

        try:
            _ingest_file_path(tmp_path, original_name=f.name)
            st.success(f"Ingested {f.name} (from upload cache)")
        except Exception as e:
            st.error(f"Ingestion failed for {f.name}: {e}")
            st.code(traceback.format_exc())


# One-time auto-load from disk on fresh UI session.
if "auto_loaded_uploads" not in st.session_state:
    st.session_state["auto_loaded_uploads"] = True
    try:
        n = _auto_load_previous_uploads()
        if n:
            st.caption(f"Auto-loaded {n} previously uploaded .nzf file(s) from data/uploads")
    except Exception:
        st.warning("Auto-load of previous uploads failed. You can re-upload the file(s) to continue.")

st.title("Natural-language Analytics + Dashboards (RAG + ReAct)")

with st.expander("Loaded Tables", expanded=True):
    sess: DataSession = st.session_state["data_session"]
    st.write("**Available tables:**", sess.available_tables() or "None")
    if sess.available_tables():
        t = sess.available_tables()[0]
        st.write(f"**Preview:** {t}")
        st.dataframe(sess.get_table(t).head(20), use_container_width=True)

tabs = st.tabs(["Reports", "Dashboards"])

with tabs[0]:
    st.subheader("Reports")
    files = st.file_uploader("Upload Ç-delimited .nzf files (multiple allowed)", type=["nzf"], accept_multiple_files=True, key="rep_upl")
    if files:
        ingest_uploaded_files(files)

    q = st.text_input("Ask a business question (report):", key="rep_q", placeholder="e.g., total sales by region for 2025")
    if st.button("Generate report", key="rep_run"):
        sess = st.session_state["data_session"]
        if not sess.available_tables():
            st.error("No tables loaded. Please upload and ingest your .nzf file first.")
            st.stop()
        res = orch.run(sess, q, mode_hint="report")
        st.session_state["last_plan"] = res.plan
        if not res.ok:
            st.error(INSUFFICIENT)
        else:
            st.info(f"Confidence: {res.confidence:.3f}")
            df = res.df

            try:
                from st_aggrid import AgGrid, GridOptionsBuilder
                gb = GridOptionsBuilder.from_dataframe(df)
                gb.configure_pagination(paginationAutoPageSize=True)
                AgGrid(df, gridOptions=gb.build(), height=420)
            except Exception:
                st.dataframe(df, use_container_width=True, height=420)

            base_name = f"report_{int(time.time())}"
            exp = export_report(df, settings.export_dir, base_name=base_name)

            c1, c2, c3 = st.columns(3)
            with c1:
                if exp.csv_path and Path(exp.csv_path).exists():
                    st.download_button("Download CSV", data=Path(exp.csv_path).read_bytes(), file_name=Path(exp.csv_path).name)
            with c2:
                if exp.xml_path and Path(exp.xml_path).exists():
                    st.download_button("Download XML", data=Path(exp.xml_path).read_bytes(), file_name=Path(exp.xml_path).name)
            with c3:
                if exp.pdf_path and Path(exp.pdf_path).exists():
                    st.download_button("Download PDF", data=Path(exp.pdf_path).read_bytes(), file_name=Path(exp.pdf_path).name)

with tabs[1]:
    st.subheader("Dashboards")
    files = st.file_uploader("Upload Ç-delimited .nzf files (multiple allowed)", type=["nzf"], accept_multiple_files=True, key="dash_upl")
    if files:
        ingest_uploaded_files(files)

    q = st.text_input("Ask a business question (dashboard):", key="dash_q", placeholder="e.g., dashboard of monthly sales trend with region filter and KPI for total sales")
    b1, b2 = st.columns([1,1])
    run = b1.button("Generate dashboard", key="dash_run")
    save = b2.button("Save query", key="dash_save")

    if run:
        sess = st.session_state["data_session"]
        if not sess.available_tables():
            st.error("No tables loaded. Please upload and ingest your .nzf file first.")
            st.stop()
        res = orch.run(sess, q, mode_hint="dashboard")
        st.session_state["last_plan"] = res.plan
        if not res.ok:
            st.error(INSUFFICIENT)
        else:
            st.info(f"Confidence: {res.confidence:.3f}")
            # Support multi-panel dashboards (e.g., KPI + chart).
            figs = getattr(res, "figures", None) or ([] if res.figure is None else [res.figure])
            if figs:
                if len(figs) == 1:
                    st.plotly_chart(figs[0], use_container_width=True)
                else:
                    # If there's at least one KPI, show KPIs in a narrow left column.
                    kpis = []
                    others = []
                    for f in figs:
                        try:
                            if getattr(f, "data", None) and len(f.data) and getattr(f.data[0], "type", "") == "indicator":
                                kpis.append(f)
                            else:
                                others.append(f)
                        except Exception:
                            others.append(f)

                    if kpis and others:
                        left, right = st.columns([1, 3])
                        with left:
                            for k in kpis:
                                st.plotly_chart(k, use_container_width=True)
                        with right:
                            for o in others:
                                st.plotly_chart(o, use_container_width=True)
                    else:
                        cols = st.columns(min(3, len(figs)))
                        for i, f in enumerate(figs):
                            with cols[i % len(cols)]:
                                st.plotly_chart(f, use_container_width=True)
            else:
                st.warning("Plotly render failed; showing table fallback.")
                st.dataframe(res.df, use_container_width=True)

    if save:
        plan = st.session_state.get("last_plan")
        if not q or not plan:
            st.warning("Generate a dashboard first (so there is a plan to save).")
        else:
            qid = save_query(settings.saved_query_dir, mode="dashboard", question=q, plan=plan)
            st.success(f"Saved query id: {qid}")

    st.divider()
    st.markdown("### Saved dashboards (regenerable)")
    saved = list_queries(settings.saved_query_dir)
    if not saved:
        st.caption("No saved queries yet.")
    else:
        options = {f"{s.id} — {s.question[:60]}": s.id for s in saved}
        sel = st.selectbox("Select a saved query", list(options.keys()))
        if st.button("Regenerate selected dashboard"):
            qid = options[sel]
            sq = load_query(settings.saved_query_dir, qid)
            sess = st.session_state["data_session"]
            if not sess.available_tables():
                st.error("No tables loaded. Please upload and ingest your .nzf file first.")
                st.stop()
            res = orch.run(sess, sq.question, mode_hint="dashboard")
            if not res.ok:
                st.error(INSUFFICIENT)
            else:
                st.info(f"Confidence: {res.confidence:.3f}")
                if res.figure is not None:
                    st.plotly_chart(res.figure, use_container_width=True)
                else:
                    st.dataframe(res.df, use_container_width=True)

with st.expander("Debug: Last Plan (JSON)", expanded=False):
    st.json(st.session_state.get("last_plan") or {})