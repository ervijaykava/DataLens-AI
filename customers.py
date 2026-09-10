"""
customers.py
-------------
Everything about customers and the opportunities/risks derived from them:

    rfm_analysis         Recency / Frequency / Monetary segmentation
    predict_churn        Random Forest churn classifier
    get_opportunities    ranks growth/retention signals into an opportunity list
    get_risks            ranks decline/churn/anomaly signals into a risk list

Opportunities and risks reuse the RFM/churn results above plus a couple of
genuinely new but simple calculations (product/region growth, revenue
concentration) - every opportunity or risk traces back to a real number in
the dataset, nothing fabricated.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ---------------------------------------------------------------------------
# RFM segmentation + churn prediction
# ---------------------------------------------------------------------------
def _has_required_columns(roles, *keys):
    return all(roles.get(k) for k in keys)


def rfm_analysis(df: pd.DataFrame, roles: dict) -> dict:
    """RFM segmentation - fully deterministic, no model needed. Needs a
    customer, date, and revenue column."""
    cust_col, date_col, rev_col = roles.get("customer_column"), roles.get("date_column"), roles.get("revenue_column")
    if not _has_required_columns(roles, "customer_column", "date_column", "revenue_column"):
        return {"available": False, "reason": "RFM analysis needs a customer, date, and revenue column."}

    data = df.dropna(subset=[date_col, cust_col]).copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])
    if data[cust_col].nunique() < 5:
        return {"available": False, "reason": "Need at least 5 distinct customers for meaningful segmentation."}

    snapshot = data[date_col].max() + pd.Timedelta(days=1)
    rfm = data.groupby(cust_col).agg(
        recency=(date_col, lambda s: (snapshot - s.max()).days),
        frequency=(cust_col, "count"),
        monetary=(rev_col, "sum"),
    ).reset_index()

    def qscore(series, ascending):
        try:
            ranks = pd.qcut(series.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        except ValueError:
            ranks = pd.Series([3] * len(series), index=series.index)
        return ranks if ascending else (6 - ranks)

    rfm["r_score"] = qscore(rfm["recency"], ascending=False)
    rfm["f_score"] = qscore(rfm["frequency"], ascending=True)
    rfm["m_score"] = qscore(rfm["monetary"], ascending=True)

    def segment(row):
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "VIP"
        if r <= 2 and f >= 4 and m >= 4:
            return "At Risk"
        if r <= 2 and f <= 2:
            return "Churned"
        if r >= 4 and f <= 2:
            return "New"
        if f >= 4 and m <= 2:
            return "Loyal"
        if r >= 4 and m >= 4:
            return "Potential VIP"
        return "Low Value"

    rfm["segment"] = rfm.apply(segment, axis=1)
    seg_counts = rfm["segment"].value_counts().to_dict()
    top_customers = rfm.sort_values("monetary", ascending=False).head(15)

    return {
        "available": True,
        "customer_count": int(len(rfm)),
        "segment_counts": {k: int(v) for k, v in seg_counts.items()},
        "top_customers": [
            {
                "customer": str(r[cust_col]), "recency_days": int(r["recency"]),
                "frequency": int(r["frequency"]), "monetary": round(float(r["monetary"]), 2),
                "segment": r["segment"],
            }
            for _, r in top_customers.iterrows()
        ],
    }


def predict_churn(df: pd.DataFrame, roles: dict) -> dict:
    """Churn prediction - a real classifier (Random Forest), not a heuristic.
    The dataset's date range is split into a "history" window and a "recent"
    window. A customer is labeled churned (1) if they purchased during
    history but not during the recent window. Features are computed only
    from the history window, so the model never sees the future it's
    predicting - the standard setup for churn modeling on transaction logs."""
    cust_col, date_col, rev_col = roles.get("customer_column"), roles.get("date_column"), roles.get("revenue_column")
    if not _has_required_columns(roles, "customer_column", "date_column", "revenue_column"):
        return {"available": False, "reason": "Churn prediction needs a customer, date, and revenue column."}

    data = df.dropna(subset=[date_col, cust_col]).copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])

    date_min, date_max = data[date_col].min(), data[date_col].max()
    span_days = (date_max - date_min).days
    if span_days < 14 or data[cust_col].nunique() < 20:
        return {"available": False, "reason": "Churn prediction needs a longer time span and at least ~20 customers to train reliably."}

    cutoff = date_min + pd.Timedelta(days=int(span_days * 0.75))
    history = data[data[date_col] <= cutoff]
    recent = data[data[date_col] > cutoff]

    if history[cust_col].nunique() < 10:
        return {"available": False, "reason": "Not enough transaction history before the evaluation cutoff to train a churn model."}

    snapshot = cutoff + pd.Timedelta(days=1)
    feats = history.groupby(cust_col).agg(
        recency=(date_col, lambda s: (snapshot - s.max()).days),
        frequency=(cust_col, "count"),
        monetary=(rev_col, "sum"),
        avg_order_value=(rev_col, "mean"),
    ).reset_index()

    active_recent = set(recent[cust_col].unique())
    feats["churned"] = (~feats[cust_col].isin(active_recent)).astype(int)

    X = feats[["recency", "frequency", "monetary", "avg_order_value"]]
    y = feats["churned"]

    if y.nunique() < 2 or len(feats) < 15:
        return {"available": False, "reason": "The churn/retained split is too imbalanced or too small to train a reliable model."}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 3),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 3),
    }

    # Score every customer using the full dataset ("today" risk).
    snapshot_full = date_max + pd.Timedelta(days=1)
    full_feats = data.groupby(cust_col).agg(
        recency=(date_col, lambda s: (snapshot_full - s.max()).days),
        frequency=(cust_col, "count"),
        monetary=(rev_col, "sum"),
        avg_order_value=(rev_col, "mean"),
    ).reset_index()
    full_feats["churn_probability"] = model.predict_proba(
        full_feats[["recency", "frequency", "monetary", "avg_order_value"]]
    )[:, 1]

    def risk_level(p):
        if p >= 0.7:
            return "HIGH"
        if p >= 0.4:
            return "MEDIUM"
        return "LOW"

    def reasons(row):
        r = []
        if row["recency"] > full_feats["recency"].median() * 1.5:
            r.append("long inactivity period")
        if row["frequency"] < full_feats["frequency"].median():
            r.append("reduced order frequency")
        if row["avg_order_value"] < full_feats["avg_order_value"].median():
            r.append("lower average order value")
        return r or ["no single dominant factor - overall pattern is atypical for retained customers"]

    full_feats["risk_level"] = full_feats["churn_probability"].apply(risk_level)
    at_risk = full_feats[full_feats["risk_level"] == "HIGH"].sort_values("churn_probability", ascending=False).head(25)

    return {
        "available": True,
        "model": "RandomForestClassifier",
        "metrics": metrics,
        "customers_scored": int(len(full_feats)),
        "high_risk_count": int((full_feats["risk_level"] == "HIGH").sum()),
        "medium_risk_count": int((full_feats["risk_level"] == "MEDIUM").sum()),
        "low_risk_count": int((full_feats["risk_level"] == "LOW").sum()),
        "at_risk_customers": [
            {
                "customer": str(r[cust_col]),
                "churn_probability": round(float(r["churn_probability"]) * 100, 1),
                "risk_level": r["risk_level"],
                "recency_days": int(r["recency"]),
                "monetary": round(float(r["monetary"]), 2),
                "reasons": reasons(r),
            }
            for _, r in at_risk.iterrows()
        ],
    }


# ---------------------------------------------------------------------------
# Opportunity Center and Risk Center
# ---------------------------------------------------------------------------
def _period_growth_by_dimension(df: pd.DataFrame, roles: dict, dim_col: str):
    date_col, rev_col = roles.get("date_column"), roles.get("revenue_column")
    if not date_col or not rev_col or not dim_col or dim_col not in df.columns:
        return []
    data = df.dropna(subset=[date_col]).copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])
    data["_period"] = data[date_col].dt.to_period("M")
    periods = sorted(data["_period"].unique())
    if len(periods) < 2:
        return []
    prev_p, last_p = periods[-2], periods[-1]
    prev = data[data["_period"] == prev_p].groupby(dim_col)[rev_col].sum()
    last = data[data["_period"] == last_p].groupby(dim_col)[rev_col].sum()
    combined = pd.concat([prev, last], axis=1, keys=["prev", "last"]).fillna(0)
    combined = combined[combined["prev"] > 0]
    combined["growth_pct"] = ((combined["last"] - combined["prev"]) / combined["prev"] * 100).round(1)
    return [
        {"segment": str(idx), "prev": round(float(r["prev"]), 2), "last": round(float(r["last"]), 2), "growth_pct": float(r["growth_pct"])}
        for idx, r in combined.iterrows()
    ]


def get_opportunities(df: pd.DataFrame, roles: dict, rfm_result: dict, churn_result: dict) -> dict:
    opportunities = []

    product_growth = _period_growth_by_dimension(df, roles, roles.get("product_column"))
    for p in sorted(product_growth, key=lambda x: -x["growth_pct"])[:3]:
        if p["growth_pct"] > 10:
            opportunities.append({
                "type": "Product Growth",
                "title": f"'{p['segment']}' is growing fast",
                "evidence": f"Revenue grew {p['growth_pct']}% period-over-period (${p['prev']:,.0f} -> ${p['last']:,.0f}).",
                "impact": "High" if p["growth_pct"] > 40 else "Medium",
                "confidence": "Medium",
                "effort": "Low",
                "recommended_action": f"Increase inventory and marketing focus on '{p['segment']}' while demand is accelerating.",
            })

    region_growth = _period_growth_by_dimension(df, roles, roles.get("region_column"))
    for r in sorted(region_growth, key=lambda x: x["growth_pct"])[:2]:
        if r["growth_pct"] < -10:
            opportunities.append({
                "type": "Regional Recovery",
                "title": f"'{r['segment']}' region has room to recover",
                "evidence": f"Revenue fell {abs(r['growth_pct'])}% period-over-period (${r['prev']:,.0f} -> ${r['last']:,.0f}).",
                "impact": "Medium",
                "confidence": "Medium",
                "effort": "Medium",
                "recommended_action": f"Investigate what changed in '{r['segment']}' and consider a targeted regional promotion.",
            })

    if rfm_result.get("available") and churn_result.get("available"):
        high_value_customers = {c["customer"] for c in rfm_result.get("top_customers", [])}
        overlap = [c for c in churn_result.get("at_risk_customers", []) if c["customer"] in high_value_customers]
        for c in overlap[:5]:
            opportunities.append({
                "type": "Retention",
                "title": f"Retain high-value customer '{c['customer']}'",
                "evidence": f"{c['churn_probability']}% churn risk, but ${c['monetary']:,.0f} lifetime value.",
                "impact": "High",
                "confidence": "Medium",
                "effort": "Low",
                "recommended_action": "Proactive outreach or a retention offer before this high-value customer fully disengages.",
            })

    weight = {"High": 3, "Medium": 2, "Low": 1}
    for o in opportunities:
        o["priority_score"] = weight[o["impact"]] * weight[o["confidence"]] - weight[o["effort"]]
    opportunities.sort(key=lambda x: -x["priority_score"])

    return {
        "available": len(opportunities) > 0,
        "opportunities": opportunities,
        "reason": None if opportunities else "No strong opportunities detected with the current data - this is a status, not an error.",
    }


def get_risks(df: pd.DataFrame, roles: dict, kpis: dict, churn_result: dict, anomaly_result: dict, forecast_result: dict) -> dict:
    risks = []

    if kpis.get("growth_pct", 0) < -5:
        risks.append({
            "type": "Revenue Risk", "severity": "High" if kpis["growth_pct"] < -15 else "Medium",
            "evidence": f"Revenue declined {abs(kpis['growth_pct'])}% period-over-period.",
            "impact": "Direct revenue loss if the trend continues.",
            "mitigation": "Run Root Cause Analysis to isolate which segment is driving the decline.",
        })

    if churn_result.get("available"):
        total = max(1, churn_result.get("customers_scored", 1))
        rate = churn_result.get("high_risk_count", 0) / total
        if rate > 0.1:
            risks.append({
                "type": "Customer Churn Risk", "severity": "Critical" if rate > 0.3 else ("High" if rate > 0.2 else "Medium"),
                "evidence": f"{churn_result['high_risk_count']} of {total} customers ({round(rate * 100, 1)}%) are high churn risk.",
                "impact": "Lost recurring revenue from disengaged customers.",
                "mitigation": "Review the At-Risk Customers list in Customer Intelligence and prioritize retention outreach.",
            })

    if anomaly_result and anomaly_result.get("anomaly_count", 0) > 0:
        total_scored = max(1, anomaly_result["anomaly_count"] + anomaly_result.get("normal_count", 0))
        rate = anomaly_result["anomaly_count"] / total_scored
        critical = anomaly_result.get("risk_counts", {}).get("Critical", 0)
        if critical > 0 or rate > 0.05:
            risks.append({
                "type": "Anomaly / Data Integrity Risk", "severity": "High" if critical > 5 else "Medium",
                "evidence": f"{anomaly_result['anomaly_count']} anomalous records detected, {critical} flagged Critical.",
                "impact": "Possible data entry errors, fraud, or genuinely unusual transactions worth reviewing.",
                "mitigation": "Review flagged records in Anomaly Detection, starting with Critical-risk ones.",
            })

    if forecast_result.get("available") and forecast_result.get("trend_direction") == "declining":
        risks.append({
            "type": "Forecast Risk", "severity": "Medium",
            "evidence": f"The best-performing forecast model projects a declining trend (slope {forecast_result.get('trend_slope')}).",
            "impact": "Revenue is projected to continue declining without intervention.",
            "mitigation": "Combine with Root Cause Analysis and the What-If Simulator to test recovery scenarios.",
        })

    rev_col, cust_col = roles.get("revenue_column"), roles.get("customer_column")
    if rev_col and cust_col:
        by_cust = df.groupby(cust_col)[rev_col].sum().sort_values(ascending=False)
        total_rev = by_cust.sum()
        if total_rev > 0 and len(by_cust) > 1:
            top_share = by_cust.iloc[0] / total_rev * 100
            if top_share > 15:
                risks.append({
                    "type": "Customer Concentration Risk", "severity": "High" if top_share > 30 else "Medium",
                    "evidence": f"Your single largest customer ('{by_cust.index[0]}') accounts for {round(top_share, 1)}% of total revenue.",
                    "impact": "Losing this one customer would materially impact total revenue.",
                    "mitigation": "Diversify the customer base; consider dedicated account management for this customer.",
                })

    severity_weight = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
    risks.sort(key=lambda r: -severity_weight.get(r["severity"], 0))

    overall_score = min(100, sum(severity_weight.get(r["severity"], 0) for r in risks) * 8) if risks else 0
    overall_level = "Critical" if overall_score > 80 else "High" if overall_score > 60 else "Medium" if overall_score > 30 else "Low"

    return {"available": True, "risks": risks, "overall_risk_score": overall_score, "overall_risk_level": overall_level}
