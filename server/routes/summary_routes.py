"""Summary statistics routes."""

from fastapi import APIRouter, HTTPException
from server.services import cache_service

router = APIRouter(prefix="/api", tags=["summary"])


@router.get("/summary/{name}")
async def get_summary(name: str):
    """Return summary statistics JSON for a table."""
    cached = cache_service.get(name)
    if not cached or not cached.get("summary"):
        raise HTTPException(404, f"No summary for {name}")
    return cached["summary"]


@router.get("/summary/{name}/charts/{chart_type}")
async def get_chart(name: str, chart_type: str):
    """Return Plotly JSON for specific chart types."""
    cached = cache_service.get(name)
    if not cached or not cached.get("summary"):
        raise HTTPException(404, f"No summary for {name}")

    summary = cached["summary"]

    if chart_type == "missingness":
        missingness = summary.get("missingness", {})
        columns_with_missing = missingness.get("columns_with_missing", [])
        if not columns_with_missing:
            return {"data": [], "layout": {}}

        import pandas as pd
        df = pd.DataFrame(columns_with_missing).head(10)

        return {
            "data": [{
                "type": "bar",
                "x": df["column"].tolist(),
                "y": df["missing_percent"].tolist(),
                "marker": {
                    "color": df["missing_percent"].tolist(),
                    "colorscale": "Reds",
                },
                "text": [f"{v:.1f}%" for v in df["missing_percent"]],
                "textposition": "outside",
            }],
            "layout": {
                "title": "Top 10 Columns by Missing Percentage",
                "xaxis": {"title": "Column"},
                "yaxis": {"title": "Missing %"},
                "height": 400,
                "template": "plotly_dark",
            },
        }

    if chart_type == "distribution":
        distributions = summary.get("distributions", {})
        charts = {}
        for key, dist_data in distributions.items():
            if isinstance(dist_data, dict) and "values" in dist_data and "counts" in dist_data:
                charts[key] = {
                    "data": [{
                        "type": "pie",
                        "labels": dist_data["values"],
                        "values": dist_data["counts"],
                        "textinfo": "percent+label",
                    }],
                    "layout": {
                        "title": f"{key.replace('_', ' ').title()} Distribution",
                        "height": 400,
                        "template": "plotly_dark",
                    },
                }
        return charts

    raise HTTPException(404, f"Unknown chart type: {chart_type}")
