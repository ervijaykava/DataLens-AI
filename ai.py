"""
ai.py
------
Everything AI-flavored in the app:

    build_business_context / ask_gemini / ask_copilot
        The Copilot chat. Builds a compact business-data summary in Python,
        sends ONE request to Gemini with that summary + the user's
        question, returns the text answer. We never send the raw dataset
        to Gemini - only pre-computed statistics (totals, top products,
        growth %, etc.). If GEMINI_API_KEY isn't set, or the request fails
        for any reason, `ask_copilot` falls back to a simple rule-based
        answer so the app still works end-to-end without any external
        credentials.

    ask_gemini_for_sql / is_safe_select / run_safe_select
        Lets the Copilot optionally answer questions by generating SQL and
        running it - but ONLY read-only SELECT queries against the user's
        own dataset table. Gemini output is never trusted or executed
        blindly; every query is validated first.

    run_full_analysis
        "Analyze My Business" - runs every analysis module already in this
        app back-to-back and assembles the results into one report, plus a
        short rule-based executive summary. No new modeling here - this is
        pure orchestration on top of data.py / analytics.py / customers.py /
        prediction.py.
"""

import json
import re

import requests
from sqlalchemy import text

import config
from database import engine
import analytics
import customers
import data
import prediction

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent"

FORBIDDEN_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "truncate",
    "create", "replace", "grant", "revoke", "attach", "into outfile",
]


# ---------------------------------------------------------------------------
# AI Copilot (Gemini + rule-based fallback)
# ---------------------------------------------------------------------------
def build_business_context(kpis: dict, insights: list, top_products: list, top_regions: list,
                            top_categories: list, forecast: dict, anomaly_count: int) -> str:
    """Turn pre-computed statistics into a compact text block for Gemini -
    never the raw dataset itself."""
    lines = [
        f"Total revenue: ${kpis.get('total_revenue', 0):,.2f}",
        f"Total orders: {kpis.get('total_orders', 0):,}",
        f"Total customers: {kpis.get('total_customers', 0):,}",
        f"Average order value: ${kpis.get('avg_order_value', 0):,.2f}",
        f"Revenue growth (latest period vs previous): {kpis.get('growth_pct', 0)}%",
    ]
    if top_products:
        lines.append("Top products by revenue: " + ", ".join(f"{p['name']} (${p['value']:,.0f})" for p in top_products[:5]))
    if top_regions:
        lines.append("Top regions by revenue: " + ", ".join(f"{r['name']} (${r['value']:,.0f})" for r in top_regions[:5]))
    if top_categories:
        lines.append("Top categories by revenue: " + ", ".join(f"{c['name']} (${c['value']:,.0f})" for c in top_categories[:5]))
    if forecast and forecast.get("available"):
        lines.append(f"Forecast trend: {forecast['trend_direction']} (best model: {forecast['best_model']})")
    if anomaly_count:
        lines.append(f"Anomalous records detected: {anomaly_count}")
    if insights:
        lines.append("Existing rule-based insights: " + " | ".join(insights[:5]))
    return "\n".join(lines)


def ask_gemini(prompt: str) -> str:
    """Send one prompt to the Gemini API and return the text response.
    Raises an exception on failure - callers decide how to handle that."""
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": config.GEMINI_API_KEY}

    response = requests.post(GEMINI_URL, headers=headers, data=json.dumps(payload), timeout=30)
    response.raise_for_status()
    body = response.json()
    return body["candidates"][0]["content"]["parts"][0]["text"]


def _fallback_answer(question: str, kpis: dict, top_products: list, top_regions: list) -> str:
    """Rule-based fallback so the Copilot still answers something useful
    without a Gemini API key or if the API call fails."""
    q = question.lower()

    if "region" in q and top_regions:
        best = top_regions[0]
        return (f"The best-performing region is {best['name']} with ${best['value']:,.2f} in revenue. "
                f"Consider allocating more marketing spend there while investigating lower-performing regions.")

    if "product" in q and top_products:
        best = top_products[0]
        return (f"{best['name']} generated the highest revenue at ${best['value']:,.2f}. "
                f"Make sure inventory stays sufficient, and consider bundling it with slower-moving products.")

    if "declin" in q or "drop" in q or "why" in q:
        growth = kpis.get("growth_pct", 0)
        if growth < 0:
            return (f"Revenue declined {abs(growth)}% versus the prior period. "
                     f"Check the Root Cause Analysis page for exactly which segment is driving this.")
        return (f"Revenue actually grew {growth}% versus the prior period, so there's no decline to explain "
                 f"right now - but check Root Cause Analysis any time that changes.")

    return (f"Total revenue is ${kpis.get('total_revenue', 0):,.2f} across {kpis.get('total_orders', 0):,} orders, "
            f"with period-over-period growth of {kpis.get('growth_pct', 0)}%. "
            f"Explore the Analytics and Autonomous Analyst pages for a deeper breakdown.")


def ask_gemini_for_sql(question: str, table_name: str, column_names: list) -> str:
    """Ask Gemini to translate a plain-English question into ONE safe
    read-only SQL SELECT query against the user's dataset table. The query
    is validated (see is_safe_select below) before it is ever run - Gemini
    output is never trusted or executed blindly."""
    prompt = (
        "Write exactly one MySQL SELECT query (no explanation, no markdown, no semicolon) "
        f"that answers this question using ONLY table `{table_name}` with these columns: "
        f"{', '.join(column_names)}.\n\nQuestion: {question}\n\nSQL query:"
    )
    raw = ask_gemini(prompt)
    # Strip markdown code fences if Gemini added them anyway.
    return raw.strip().strip("`").replace("sql\n", "", 1).strip()


def ask_copilot(question: str, context: str, kpis: dict, top_products: list, top_regions: list) -> dict:
    """Main entry point used by the /copilot route."""
    prompt = (
        "You are a business intelligence analyst. Answer the user's question using ONLY "
        "the business data summary below. Be concise, cite concrete numbers from the "
        "summary, and end with one actionable recommendation. If the summary doesn't "
        "contain enough information to answer, say so honestly.\n\n"
        f"BUSINESS DATA SUMMARY:\n{context}\n\nQUESTION: {question}"
    )

    if config.GEMINI_API_KEY:
        try:
            answer = ask_gemini(prompt)
            return {"answer": answer, "source": "gemini"}
        except Exception:
            pass  # fall through to the rule-based fallback below

    fallback = _fallback_answer(question, kpis, top_products, top_regions)
    return {"answer": fallback, "source": "fallback"}


# ---------------------------------------------------------------------------
# Safe, read-only SQL for the Copilot
# ---------------------------------------------------------------------------
def is_safe_select(sql: str, allowed_table: str) -> tuple[bool, str]:
    """Safety rules, checked before anything is executed:
      1. The query must start with SELECT.
      2. None of the destructive keywords may appear anywhere in the query.
      3. Only a single statement is allowed (no ";" stacking).
      4. The query must reference only the caller's own table name."""
    cleaned = sql.strip().rstrip(";").strip()

    if not cleaned.lower().startswith("select"):
        return False, "Only SELECT queries are allowed."

    if ";" in cleaned:
        return False, "Only a single statement is allowed."

    lowered = cleaned.lower()
    for word in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return False, f"The keyword '{word}' is not allowed in Copilot queries."

    if allowed_table.lower() not in lowered:
        return False, f"The query must reference your dataset table ({allowed_table})."

    return True, ""


def run_safe_select(sql: str, allowed_table: str, row_limit: int = 200):
    """A LIMIT is always added if the query doesn't already have one, so a
    runaway query can't return an enormous result set."""
    ok, reason = is_safe_select(sql, allowed_table)
    if not ok:
        return {"ok": False, "error": reason, "rows": [], "columns": []}

    cleaned = sql.strip().rstrip(";").strip()
    if "limit" not in cleaned.lower():
        cleaned = f"{cleaned} LIMIT {row_limit}"

    try:
        with engine.connect() as conn:
            result = conn.execute(text(cleaned))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchall()]
        return {"ok": True, "error": None, "rows": rows, "columns": columns}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "rows": [], "columns": []}


# ---------------------------------------------------------------------------
# Autonomous Analyst (orchestrates every module above into one report)
# ---------------------------------------------------------------------------
def run_full_analysis(df, roles):
    quality = data.compute_quality_score(df)
    kpis = analytics.calculate_kpis(df, roles)
    insights = analytics.generate_insights(df, roles, kpis)

    root_cause = prediction.analyze_kpi_change(df, roles)
    rfm = customers.rfm_analysis(df, roles)
    churn = customers.predict_churn(df, roles)
    anomalies = prediction.detect_anomalies(df, roles)
    forecast = prediction.forecast_metric(df, roles)
    health = analytics.compute_health_score(kpis, anomalies, churn, forecast, len(df))
    opportunities = customers.get_opportunities(df, roles, rfm, churn)
    risks = customers.get_risks(df, roles, kpis, churn, anomalies, forecast)

    findings = []

    if kpis["growth_pct"] < -5:
        findings.append({
            "priority": "HIGH", "title": f"Revenue declined {abs(kpis['growth_pct'])}% period-over-period.",
            "detail": root_cause.get("strongest_contributor_note") or "Review the Root Cause page for a full breakdown.",
            "action": "Investigate the declining segment and consider a targeted promotion.",
        })
    elif kpis["growth_pct"] > 5:
        findings.append({
            "priority": "MEDIUM", "title": f"Revenue grew {kpis['growth_pct']}% period-over-period.",
            "detail": "Growth is currently trending positive.",
            "action": "Double down on whichever product/region is driving the growth.",
        })

    if anomalies.get("anomaly_count", 0) > 0:
        findings.append({
            "priority": "MEDIUM", "title": f"{anomalies['anomaly_count']} anomalous records detected.",
            "detail": "These records fall well outside the normal range for this dataset.",
            "action": "Review the Anomaly Detection page, starting with Critical-risk records.",
        })

    if churn.get("available") and churn.get("high_risk_count", 0) > 0:
        findings.append({
            "priority": "HIGH", "title": f"{churn['high_risk_count']} customers are at high risk of churning.",
            "detail": f"Out of {churn['customers_scored']} customers scored.",
            "action": "Prioritize retention outreach for the At-Risk Customers list.",
        })

    if opportunities.get("opportunities"):
        top = opportunities["opportunities"][0]
        findings.append({
            "priority": "MEDIUM", "title": top["title"],
            "detail": top["evidence"],
            "action": top["recommended_action"],
        })

    priority_weight = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    findings.sort(key=lambda f: -priority_weight.get(f["priority"], 0))

    summary_lines = [
        f"Analyzed {len(df):,} records across {len(df.columns)} columns (data quality: {quality['score']}/100, {quality['status']}).",
    ]
    if health.get("available"):
        summary_lines.append(f"Overall business health is {health['score']}/100 ({health['status']}).")
    summary_lines.append(
        f"Total revenue is ${kpis['total_revenue']:,.0f} across {kpis['total_orders']:,} orders, "
        f"{'up' if kpis['growth_pct'] >= 0 else 'down'} {abs(kpis['growth_pct'])}% period-over-period."
    )
    if risks.get("risks"):
        summary_lines.append(f"{len(risks['risks'])} risk(s) identified, overall risk level: {risks['overall_risk_level']}.")
    if opportunities.get("opportunities"):
        summary_lines.append(f"{len(opportunities['opportunities'])} opportunity(ies) identified.")

    return {
        "executive_summary": summary_lines,
        "findings": findings,
        "data_quality": quality,
        "kpis": kpis,
        "insights": insights,
        "root_cause": root_cause,
        "customer_rfm": rfm,
        "customer_churn": churn,
        "anomalies": anomalies,
        "forecast": forecast,
        "business_health": health,
        "opportunities": opportunities,
        "risks": risks,
    }
