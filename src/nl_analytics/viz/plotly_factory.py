from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from nl_analytics.exceptions.errors import PlotlyRenderError
from nl_analytics.logging.logger import get_logger

log = get_logger("viz.plotly_factory")

SUPPORTED_TYPES = {
    "bar", "line", "pie", "scatter", "area", "histogram", "box", "violin", "heatmap", "kpi", "table",
    "3d_scatter", "sankey"
}

def build_figure(df: pd.DataFrame, spec: Dict[str, Any]) -> go.Figure:
    try:
        chart_type = (spec.get("type") or "bar").lower()
        if chart_type not in SUPPORTED_TYPES:
            raise PlotlyRenderError(f"Unsupported chart type: {chart_type}")

        if chart_type == "kpi":
            value_col = spec.get("value")
            title = spec.get("title", value_col or "KPI")
            if not value_col or value_col not in df.columns:
                raise PlotlyRenderError("KPI requires a valid 'value' column.")
            val = df[value_col].iloc[0] if len(df) else None
            return go.Figure(go.Indicator(mode="number", value=val, title={"text": str(title)}))

        if chart_type == "table":
            return go.Figure(
                data=[go.Table(
                    header=dict(values=list(df.columns)),
                    cells=dict(values=[df[c].tolist() for c in df.columns])
                )]
            )

        if chart_type == "sankey":
            src = spec.get("source", "source")
            tgt = spec.get("target", "target")
            val = spec.get("value", "value")
            for c in (src, tgt, val):
                if c not in df.columns:
                    raise PlotlyRenderError(f"Sankey requires column: {c}")
            labels = pd.Index(pd.concat([df[src].astype(str), df[tgt].astype(str)]).unique()).tolist()
            idx = {k:i for i,k in enumerate(labels)}
            return go.Figure(data=[go.Sankey(
                node=dict(label=labels),
                link=dict(source=df[src].astype(str).map(idx), target=df[tgt].astype(str).map(idx), value=df[val])
            )])

        x = spec.get("x")
        y = spec.get("y")
        color = spec.get("color")

        if chart_type in {"bar", "line", "scatter", "area"}:
            if not x or not y:
                raise PlotlyRenderError(f"{chart_type} requires x and y.")
            if x not in df.columns or y not in df.columns:
                raise PlotlyRenderError("Spec references missing columns.")
            fn = {"bar": px.bar, "line": px.line, "scatter": px.scatter, "area": px.area}[chart_type]
            return fn(df, x=x, y=y, color=color if (color in df.columns) else None)

        if chart_type == "pie":
            names = spec.get("names")
            values = spec.get("values")
            if not names or not values:
                raise PlotlyRenderError("pie requires names and values.")
            return px.pie(df, names=names, values=values)

        if chart_type == "heatmap":
            x = spec.get("x")
            y = spec.get("y")
            z = spec.get("z")
            if not x or not y or not z:
                raise PlotlyRenderError("heatmap requires x, y, z.")
            return px.density_heatmap(df, x=x, y=y, z=z)

        if chart_type == "histogram":
            x = spec.get("x")
            if not x:
                raise PlotlyRenderError("histogram requires x.")
            return px.histogram(df, x=x, color=color if (color in df.columns) else None)

        if chart_type in {"box", "violin"}:
            x = spec.get("x")
            y = spec.get("y")
            if not y:
                raise PlotlyRenderError(f"{chart_type} requires y.")
            fn = px.box if chart_type == "box" else px.violin
            return fn(df, x=x if (x in df.columns) else None, y=y, color=color if (color in df.columns) else None)

        if chart_type == "3d_scatter":
            x = spec.get("x")
            y = spec.get("y")
            z = spec.get("z")
            if not x or not y or not z:
                raise PlotlyRenderError("3d_scatter requires x, y, z.")
            return px.scatter_3d(df, x=x, y=y, z=z, color=color if (color in df.columns) else None)

        raise PlotlyRenderError(f"Unhandled chart type: {chart_type}")
    except Exception:
        log.exception("Plotly render error")
        raise
