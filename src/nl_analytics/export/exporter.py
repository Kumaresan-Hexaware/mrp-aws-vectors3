from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

from nl_analytics.exceptions.errors import ExportError
from nl_analytics.logging.logger import get_logger

log = get_logger("export.exporter")

@dataclass(frozen=True)
class ExportPaths:
    csv_path: Optional[str] = None
    xml_path: Optional[str] = None
    pdf_path: Optional[str] = None

def export_report(df: pd.DataFrame, out_dir: str, base_name: str) -> ExportPaths:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = str(Path(out_dir) / f"{base_name}.csv")
    xml_path = str(Path(out_dir) / f"{base_name}.xml")
    pdf_path = str(Path(out_dir) / f"{base_name}.pdf")

    try:
        df.to_csv(csv_path, index=False, encoding="utf-8")
        log.info("Exported CSV", extra={"path": csv_path})
    except Exception as e:
        log.exception("CSV export failed")
        raise ExportError("CSV export failed") from e

    try:
        df.to_xml(xml_path, index=False, root_name="Report", row_name="Row")
        log.info("Exported XML", extra={"path": xml_path})
    except Exception:
        log.exception("XML export failed")
        xml_path = None

    try:
        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        elements = [Paragraph(f"Report: {base_name}", styles["Title"]), Spacer(1, 12)]
        max_rows = min(len(df), 200)
        table_data = [list(df.columns)] + df.head(max_rows).astype(str).values.tolist()
        t = Table(table_data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ]))
        elements.append(t)
        doc.build(elements)
        log.info("Exported PDF", extra={"path": pdf_path})
    except Exception:
        log.exception("PDF export failed")
        pdf_path = None

    return ExportPaths(csv_path=csv_path, xml_path=xml_path, pdf_path=pdf_path)
