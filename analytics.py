"""
analytics.py
-------------
Everything that turns the active dataset into business numbers:

    calculate_kpis            top-level KPI cards on the Executive Dashboard
    revenue_trend              monthly revenue series (for the trend chart)
    breakdown_by                generic "sum by category" used for several charts
    summary_statistics          numeric summary table for the Analytics page
    correlation_matrix          numeric correlation table for the Analytics page
    category_distribution       simple value-count breakdown
    generate_insights           rule-based one-line insights (no AI needed)
    compute_health_score        0-100 composite "Business Health" score
    generate_recommendations    turns opportunities/risks into short action items
    simulate                    what-if price/demand/cost calculator
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Core KPIs, trends and breakdowns
# ---------------------------------------------------------------------------
def calculate_kpis(df: pd.DataFrame, roles: dict) -> dict:
    """Top-level KPI cards shown on the Executive Dashboard."""
    rev_col = roles.get("revenue_column")
    cost_col = roles.get("cost_column")
    cust_col = roles.get("customer_column")
    date_col = roles.get("date_column")

    total_revenue = float(df[rev_col].sum()) if rev_col else 0.0
    total_orders = int(len(df))
    total_customers = int(df[cust_col].nunique()) if cust_col else 0
    avg_order_value = round(total_revenue / total_orders, 2) if total_orders else 0.0

    total_cost = float(df[cost_col].sum()) if cost_col else None
    total_profit = (total_revenue - total_cost) if total_cost is not None else None

    growth_pct = 0.0
    if rev_col and date_col and date_col in df.columns:
        monthly = df.dropna(subset=[date_col]).copy()
        monthly["month"] = pd.to_datetime(monthly[date_col]).dt.to_period("M")
        by_month = monthly.groupby("month")[rev_col].sum().sort_index()
        if len(by_month) >= 2:
            prev, last = by_month.iloc[-2], by_month.iloc[-1]
            if prev > 0:
                growth_pct = round(((last - prev) / prev) * 100, 1)

    return {
        "total_revenue": round(total_revenue, 2),
        "total_orders": total_orders,
        "total_customers": total_customers,
        "avg_order_value": avg_order_value,
        "growth_pct": growth_pct,
        "total_cost": round(total_cost, 2) if total_cost is not None else None,
        "total_profit": round(total_profit, 2) if total_profit is not None else None,
    }


def revenue_trend(df: pd.DataFrame, roles: dict) -> list:
    """Monthly revenue totals, used to draw the Revenue Trend SVG chart."""
    rev_col, date_col = roles.get("revenue_column"), roles.get("date_column")
    if not (rev_col and date_col):
        return []
    data = df.dropna(subset=[date_col]).copy()
    data["month"] = pd.to_datetime(data[date_col]).dt.strftime("%Y-%m")
    trend = data.groupby("month")[rev_col].sum().round(2).reset_index()
    return [{"label": row["month"], "value": float(row[rev_col])} for _, row in trend.iterrows()]


def breakdown_by(df: pd.DataFrame, group_col: str, value_col: str, top_n: int = None) -> list:
    """Generic 'sum value_col grouped by group_col' used for category/region/product charts."""
    if not group_col or not value_col or group_col not in df.columns:
        return []
    result = df.groupby(group_col)[value_col].sum().round(2).sort_values(ascending=False)
    if top_n:
        result = result.head(top_n)
    return [{"name": str(k), "value": float(v)} for k, v in result.items()]


def summary_statistics(df: pd.DataFrame) -> list:
    """Numeric summary stats table for the Analytics page."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return []
    desc = numeric.describe().round(2)
    rows = []
    for col in desc.columns:
        rows.append({
            "column": col,
            "count": int(desc[col]["count"]),
            "mean": float(desc[col]["mean"]),
            "std": float(desc[col]["std"]),
            "min": float(desc[col]["min"]),
            "max": float(desc[col]["max"]),
        })
    return rows


def correlation_matrix(df: pd.DataFrame):
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr().round(2)
    columns = list(corr.columns)
    rows = [{"label": idx, "cells": [float(v) for v in corr.loc[idx]]} for idx in columns]
    return {"columns": columns, "rows": rows}


def category_distribution(df: pd.DataFrame, col: str) -> list:
    if not col or col not in df.columns:
        return []
    counts = df[col].value_counts().head(10)
    return [{"name": str(k), "value": int(v)} for k, v in counts.items()]


def generate_insights(df: pd.DataFrame, roles: dict, kpis: dict) -> list:
    """Small set of rule-based 'insights' shown on the Dashboard (no AI needed)."""
    insights = []
    rev_col = roles.get("revenue_column")
    cat_col = roles.get("category_column")
    region_col = roles.get("region_column")
    product_col = roles.get("product_column")

    if kpis["growth_pct"] > 0:
        insights.append(f"Revenue grew {kpis['growth_pct']}% versus the previous period.")
    elif kpis["growth_pct"] < 0:
        insights.append(f"Revenue dropped {abs(kpis['growth_pct'])}% versus the previous period.")

    if product_col and rev_col:
        top_product = df.groupby(product_col)[rev_col].sum().idxmax()
        insights.append(f"'{top_product}' is the top revenue-generating product.")

    if region_col and rev_col:
        top_region = df.groupby(region_col)[rev_col].sum().idxmax()
        insights.append(f"The '{top_region}' region contributes the most revenue.")

    if cat_col and rev_col:
        top_cat = df.groupby(cat_col)[rev_col].sum().idxmax()
        insights.append(f"'{top_cat}' is the strongest performing category.")

    if rev_col:
        insights.append(f"Average order value is ${kpis['avg_order_value']:,.2f}.")

    return insights


# ---------------------------------------------------------------------------
# Business Health score
# ---------------------------------------------------------------------------
def compute_health_score(kpis: dict, anomaly_result: dict, churn_result: dict, forecast_result: dict, total_rows: int) -> dict:
    """A 0-100 composite "Business Health" score, built ONLY from signals
    that are actually available for the current dataset. If a signal can't
    be computed (e.g. churn needs more data than is available), its weight
    is simply left out of the average instead of being faked with a neutral
    default - the score should never claim more confidence than the data
    supports."""
    signals = []  # (name, score_0_100, weight, explanation)

    # 1. Revenue growth
    growth = kpis.get("growth_pct", 0)
    growth_score = max(0, min(100, 50 + growth * 4))  # +12.5% growth -> 100, -12.5% -> 0
    signals.append(("Revenue Growth", growth_score, 1.5,
                     f"Revenue {'grew' if growth >= 0 else 'declined'} {abs(growth)}% period-over-period."))

    # 2. Data / anomaly stability
    if anomaly_result and total_rows:
        anomaly_rate = anomaly_result.get("anomaly_count", 0) / max(1, total_rows)
        anomaly_score = max(0, 100 - anomaly_rate * 100 * 10)  # 10% anomaly rate -> 0
        signals.append(("Data Stability", anomaly_score, 1.0,
                         f"{anomaly_result.get('anomaly_count', 0)} anomalies detected out of {total_rows} records."))

    # 3. Customer retention (churn)
    if churn_result and churn_result.get("available"):
        total_scored = max(1, churn_result.get("customers_scored", 1))
        high_risk_rate = churn_result.get("high_risk_count", 0) / total_scored
        churn_score = max(0, 100 - high_risk_rate * 200)  # 50% high-risk -> 0
        signals.append(("Customer Retention", churn_score, 1.5,
                         f"{churn_result.get('high_risk_count', 0)} of {total_scored} customers "
                         f"({round(high_risk_rate * 100, 1)}%) are high churn risk."))

    # 4. Forecast outlook
    if forecast_result and forecast_result.get("available"):
        slope = forecast_result.get("trend_slope", 0) or 0
        forecast_score = max(0, min(100, 50 + slope * 5))
        signals.append(("Forecast Outlook", forecast_score, 1.0,
                         f"Forecast trend is {'rising' if slope >= 0 else 'declining'}."))

    if not signals:
        return {"available": False, "reason": "Not enough signals available to compute a health score yet."}

    total_weight = sum(w for _, _, w, _ in signals)
    weighted = sum(s * w for _, s, w, _ in signals) / total_weight
    score = round(weighted, 1)

    if score >= 80:
        status = "Healthy"
    elif score >= 60:
        status = "Watch"
    elif score >= 40:
        status = "At Risk"
    else:
        status = "Critical"

    return {
        "available": True,
        "score": score,
        "status": status,
        "signals": [
            {"name": name, "score": round(s, 1), "weight": w, "explanation": expl}
            for name, s, w, expl in signals
        ],
    }


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
def generate_recommendations(kpis: dict, opportunities: dict, risks: dict, top_products: list) -> list:
    """Turns opportunities + risks + insights into 3-5 short, actionable
    recommendations for the Dashboard. Every recommendation traces back to a
    real number that was already computed elsewhere. Where an impact is
    estimated (rather than directly measured), it is always labeled
    "Estimated impact" and explains its basis - never presented as a
    guarantee."""
    recommendations = []

    for opp in opportunities.get("opportunities", [])[:3]:
        rec = {
            "title": opp["recommended_action"],
            "reason": opp["evidence"],
            "estimated_impact": None,
        }
        if opp["type"] == "Product Growth" and top_products:
            rec["estimated_impact"] = (
                f"Estimated impact: continued growth at the current rate could add meaningfully to next "
                f"period's revenue. Basis: recent period-over-period growth rate for this product."
            )
        recommendations.append(rec)

    for risk in risks.get("risks", [])[:2]:
        recommendations.append({
            "title": risk["mitigation"],
            "reason": risk["evidence"],
            "estimated_impact": None,
        })

    if kpis.get("total_revenue") and top_products:
        top = top_products[0]
        share = round((top["value"] / kpis["total_revenue"]) * 100, 1) if kpis["total_revenue"] else 0
        if share > 25:
            recommendations.append({
                "title": f"Protect supply and marketing continuity for '{top['name']}'",
                "reason": f"'{top['name']}' contributes {share}% of total revenue, making it a concentration point.",
                "estimated_impact": None,
            })

    return recommendations[:5]


# ---------------------------------------------------------------------------
# What-If Simulator
# ---------------------------------------------------------------------------
def _baseline(df: pd.DataFrame, roles: dict) -> dict:
    rev_col, cost_col = roles.get("revenue_column"), roles.get("cost_column")
    revenue = float(df[rev_col].sum()) if rev_col else 0.0
    cost = float(df[cost_col].sum()) if cost_col else None
    profit = (revenue - cost) if cost is not None else None
    margin = (profit / revenue * 100) if (profit is not None and revenue) else None
    return {
        "revenue": round(revenue, 2), "cost": round(cost, 2) if cost is not None else None,
        "profit": round(profit, 2) if profit is not None else None,
        "margin_pct": round(margin, 1) if margin is not None else None,
    }


def _apply_scenario(baseline: dict, price_pct: float, demand_pct: float, cost_pct: float) -> dict:
    revenue_mult = (1 + price_pct / 100) * (1 + demand_pct / 100)
    revenue_new = baseline["revenue"] * revenue_mult

    cost_new = profit_new = margin_new = None
    if baseline["cost"] is not None:
        cost_mult = (1 + demand_pct / 100) * (1 + cost_pct / 100)
        cost_new = baseline["cost"] * cost_mult
        profit_new = revenue_new - cost_new
        margin_new = (profit_new / revenue_new * 100) if revenue_new else None

    return {
        "revenue": round(revenue_new, 2),
        "cost": round(cost_new, 2) if cost_new is not None else None,
        "profit": round(profit_new, 2) if profit_new is not None else None,
        "margin_pct": round(margin_new, 1) if margin_new is not None else None,
    }


def simulate(df: pd.DataFrame, roles: dict, price_pct: float = 0, demand_pct: float = 0, cost_pct: float = 0) -> dict:
    """A transparent what-if calculator, NOT a causal forecasting model.
    Applies user-specified percentage changes to price, demand, and cost to
    project revenue/profit impact using simple, clearly-stated multiplicative
    assumptions."""
    if not roles.get("revenue_column"):
        return {"available": False, "reason": "Simulation needs a revenue column, which wasn't detected in this dataset."}

    baseline = _baseline(df, roles)
    custom = _apply_scenario(baseline, price_pct, demand_pct, cost_pct)

    return {
        "available": True,
        "baseline": baseline,
        "inputs": {"price_change_pct": price_pct, "demand_change_pct": demand_pct, "cost_change_pct": cost_pct},
        "custom_scenario": custom,
        "presets": {
            "best_case": _apply_scenario(baseline, max(price_pct, 5), max(demand_pct, 10), min(cost_pct, -3)),
            "expected_case": custom,
            "worst_case": _apply_scenario(baseline, min(price_pct, -5), min(demand_pct, -10), max(cost_pct, 5)),
        },
        "assumptions": [
            "Revenue scales multiplicatively with price change and demand/quantity change - a direct 'what if' calculation, not a prediction.",
            "Cost (if available) scales with demand/quantity change plus any direct cost change you specify.",
            "This is a planning tool for directional impact, not a causal forecast of what will actually happen.",
        ],
    }
