"""
prediction.py
--------------
All the scikit-learn-based predictive analytics in the app:

    forecast_metric          monthly time-series forecast (Forecast page)
    detect_anomalies          Isolation Forest outlier detection (Anomalies page)
    analyze_kpi_change        period-over-period root cause breakdown (Root Cause page)
    get_predictable_columns   which columns can be predicted
    train_and_predict         generic regression on a user-chosen target column (Predictions page)

Deliberately sticks to naive / moving-average / linear-regression /
random-forest / isolation-forest models - fast, dependency-light, and easy
to explain - rather than ARIMA/Prophet/LSTM, which need far more data than
a typical uploaded spreadsheet has.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
def _monthly_series(df: pd.DataFrame, date_col: str, metric_col: str) -> pd.Series:
    data = df.dropna(subset=[date_col]).copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])
    data["_period"] = data[date_col].dt.to_period("M")
    return data.groupby("_period")[metric_col].sum().sort_index()


def _backtest(y: np.ndarray, holdout: int = 3):
    """Check how well each simple model would have predicted the last
    `holdout` known points, using only earlier data to train."""
    holdout = min(holdout, max(1, len(y) - 3))
    train, test = y[:-holdout], y[-holdout:]
    if len(train) < 2:
        return None

    x_train = np.arange(len(train)).reshape(-1, 1)
    x_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)

    predictions = {
        "naive": np.full(len(test), train[-1]),
        "moving_average": np.full(len(test), np.mean(train[-min(3, len(train)):])),
    }

    lr = LinearRegression().fit(x_train, train)
    predictions["linear_regression"] = lr.predict(x_test)

    if len(train) >= 6:
        rf = RandomForestRegressor(n_estimators=150, max_depth=4, random_state=42)
        rf.fit(x_train, train)
        predictions["random_forest"] = rf.predict(x_test)

    metrics = {}
    for name, pred in predictions.items():
        mae = float(mean_absolute_error(test, pred))
        rmse = float(np.sqrt(mean_squared_error(test, pred)))
        metrics[name] = {"mae": round(mae, 2), "rmse": round(rmse, 2)}
    return metrics


def forecast_metric(df: pd.DataFrame, roles: dict, metric_col: str = None, periods: int = 6) -> dict:
    """Aggregates a metric to a monthly time series, backtests a few simple
    forecasting approaches on the most recent known points, and forecasts
    forward using whichever approach backtested best."""
    date_col = roles.get("date_column")
    metric_col = metric_col or roles.get("revenue_column")

    if not date_col or not metric_col or metric_col not in df.columns:
        return {"available": False, "reason": "Forecasting needs a date column and a numeric metric column (defaults to revenue)."}

    series = _monthly_series(df, date_col, metric_col)
    if len(series) < 4:
        return {"available": False, "reason": "Need at least 4 monthly data points to forecast reliably."}

    y = series.values.astype(float)
    metrics = _backtest(y)
    if not metrics:
        return {"available": False, "reason": "Not enough history to backtest forecasting models."}

    best_model = min(metrics, key=lambda m: metrics[m]["mae"])

    x_full = np.arange(len(y)).reshape(-1, 1)
    x_future = np.arange(len(y), len(y) + periods).reshape(-1, 1)

    if best_model == "naive":
        forecast = np.full(periods, y[-1])
    elif best_model == "moving_average":
        forecast = np.full(periods, np.mean(y[-min(3, len(y)):]))
    elif best_model == "linear_regression":
        forecast = LinearRegression().fit(x_full, y).predict(x_future)
    else:  # random_forest
        rf = RandomForestRegressor(n_estimators=150, max_depth=4, random_state=42)
        rf.fit(x_full, y)
        forecast = rf.predict(x_future)

    trend_model = LinearRegression().fit(x_full, y)
    slope = float(trend_model.coef_[0])
    resid_std = float(np.std(y - trend_model.predict(x_full)))
    band = resid_std * 1.28  # roughly an 80% interval

    future_labels = [str(series.index[-1] + i) for i in range(1, periods + 1)]

    return {
        "available": True,
        "metric": metric_col,
        "history": [{"period": str(p), "value": round(float(v), 2)} for p, v in series.items()],
        "forecast": [
            {"period": lbl, "value": round(float(v), 2), "lower": round(float(v - band), 2), "upper": round(float(v + band), 2)}
            for lbl, v in zip(future_labels, forecast)
        ],
        "model_comparison": metrics,
        "best_model": best_model,
        "trend_slope": round(slope, 2),
        "trend_direction": "rising" if slope > 0 else ("declining" if slope < 0 else "flat"),
    }


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------
def detect_anomalies(df: pd.DataFrame, roles: dict, contamination: float = 0.03) -> dict:
    """Flags unusual rows (e.g. abnormally large orders) using Isolation
    Forest - a well-known, easily explainable model for this kind of task -
    then describes each anomaly in a plain-language sentence."""
    numeric_cols = [c for c in roles.get("numeric_columns", []) if "id" not in c.lower()]

    if len(numeric_cols) < 1 or len(df) < 10:
        return {"anomaly_count": 0, "normal_count": len(df), "anomalies": [], "chart_data": [], "risk_counts": {}}

    X = df[numeric_cols].fillna(0)

    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(X)  # -1 = anomaly, 1 = normal
    scores = model.decision_function(X)

    score_min, score_max = float(scores.min()), float(scores.max())
    score_range = (score_max - score_min) or 1.0

    def risk_score(s):
        normalized = (score_max - s) / score_range  # lower decision_function -> higher risk
        return round(float(normalized) * 100, 1)

    def risk_bucket(score):
        if score <= 30:
            return "Low"
        if score <= 60:
            return "Medium"
        if score <= 80:
            return "High"
        return "Critical"

    result_df = df.copy()
    result_df["_is_anomaly"] = preds == -1
    result_df["_anomaly_score"] = scores
    result_df["_risk_score"] = [risk_score(s) for s in scores]
    result_df["_risk_level"] = result_df["_risk_score"].apply(risk_bucket)

    anomalies = result_df[result_df["_is_anomaly"]].sort_values("_anomaly_score").head(25)

    rev_col = roles.get("revenue_column")
    qty_col = roles.get("quantity_column")
    product_col = roles.get("product_column")

    explained = []
    for _, row in anomalies.iterrows():
        reasons = []
        if qty_col and row[qty_col] > df[qty_col].mean() + 2 * df[qty_col].std():
            reasons.append("unusually high quantity")
        if rev_col and row[rev_col] > df[rev_col].mean() + 2 * df[rev_col].std():
            reasons.append("unusually high revenue")
        if not reasons:
            reasons.append("statistically unusual combination of values")

        explained.append({
            "product": str(row[product_col]) if product_col else None,
            "revenue": round(float(row[rev_col]), 2) if rev_col else None,
            "reason": "Flagged due to " + " and ".join(reasons) + ".",
            "risk_score": row["_risk_score"],
            "risk_level": row["_risk_level"],
        })

    risk_counts = anomalies["_risk_level"].value_counts().to_dict() if len(anomalies) else {}

    chart_sample = result_df.sample(min(300, len(result_df)), random_state=1)
    chart_data = [
        {
            "x": float(row[numeric_cols[0]]),
            "y": float(row[numeric_cols[1]]) if len(numeric_cols) > 1 else float(row[numeric_cols[0]]),
            "is_anomaly": bool(row["_is_anomaly"]),
        }
        for _, row in chart_sample.iterrows()
    ]

    return {
        "anomaly_count": int(result_df["_is_anomaly"].sum()),
        "normal_count": int((~result_df["_is_anomaly"]).sum()),
        "anomalies": explained,
        "risk_counts": {k: int(v) for k, v in risk_counts.items()},
        "chart_data": chart_data,
    }


# ---------------------------------------------------------------------------
# Root cause analysis
# ---------------------------------------------------------------------------
def analyze_kpi_change(df: pd.DataFrame, roles: dict, metric_col: str = None) -> dict:
    """Answers "why did <metric> change?" by comparing the two most recent
    periods and breaking the net change down across whichever categorical
    dimensions exist (product, region, category, customer). Uses associative
    language ("contributor") rather than causal language - a period-over-period
    breakdown shows WHERE a change concentrated, not a proven cause of it."""
    date_col = roles.get("date_column")
    metric_col = metric_col or roles.get("revenue_column")

    if not date_col or not metric_col or metric_col not in df.columns:
        return {"available": False, "reason": "Root cause analysis needs a date column and a numeric metric column (defaults to revenue)."}

    data = df.dropna(subset=[date_col]).copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])
    data["_period"] = data[date_col].dt.to_period("M")

    periods = sorted(data["_period"].unique())
    if len(periods) < 2:
        return {"available": False, "reason": "Need at least two time periods (months) of data to compare."}

    prev_period, last_period = periods[-2], periods[-1]
    prev_df = data[data["_period"] == prev_period]
    last_df = data[data["_period"] == last_period]

    prev_total = float(prev_df[metric_col].sum())
    last_total = float(last_df[metric_col].sum())
    net_change = round(last_total - prev_total, 2)
    net_change_pct = round((net_change / prev_total) * 100, 1) if prev_total else 0.0

    dimensions = {}
    for label, col in [
        ("product", roles.get("product_column")),
        ("region", roles.get("region_column")),
        ("category", roles.get("category_column")),
        ("customer", roles.get("customer_column")),
    ]:
        if not col or col not in df.columns:
            continue
        prev_by = prev_df.groupby(col)[metric_col].sum()
        last_by = last_df.groupby(col)[metric_col].sum()
        combined = pd.concat([prev_by, last_by], axis=1, keys=["prev", "last"]).fillna(0)
        combined["change"] = combined["last"] - combined["prev"]

        top_negative = combined.sort_values("change").head(3)
        top_positive = combined.sort_values("change", ascending=False).head(3)

        dimensions[label] = {
            "top_negative_contributors": [
                {"segment": str(idx), "prev": round(float(r["prev"]), 2), "last": round(float(r["last"]), 2), "change": round(float(r["change"]), 2)}
                for idx, r in top_negative.iterrows() if r["change"] < 0
            ],
            "top_positive_contributors": [
                {"segment": str(idx), "prev": round(float(r["prev"]), 2), "last": round(float(r["last"]), 2), "change": round(float(r["change"]), 2)}
                for idx, r in top_positive.iterrows() if r["change"] > 0
            ],
        }

    strongest = None
    if net_change < 0:
        candidates = []
        for label, d in dimensions.items():
            if d["top_negative_contributors"]:
                candidates.append((label, d["top_negative_contributors"][0]))
        if candidates:
            candidates.sort(key=lambda t: t[1]["change"])
            label, worst = candidates[0]
            contribution_pct = round((worst["change"] / net_change) * 100, 1) if net_change else 0
            strongest = (f"The strongest single contributor is {label} = '{worst['segment']}', accounting for an "
                         f"estimated {contribution_pct}% of the net change (associative, not proven causal).")

    return {
        "available": True,
        "metric": metric_col,
        "period_from": str(prev_period),
        "period_to": str(last_period),
        "prev_total": round(prev_total, 2),
        "last_total": round(last_total, 2),
        "net_change": net_change,
        "net_change_pct": net_change_pct,
        "direction": "increase" if net_change > 0 else ("decrease" if net_change < 0 else "flat"),
        "dimensions": dimensions,
        "strongest_contributor_note": strongest,
    }


# ---------------------------------------------------------------------------
# Generic predictions (Predictions page)
# ---------------------------------------------------------------------------
MODELS = {
    "linear_regression": lambda: LinearRegression(),
    "random_forest": lambda: RandomForestRegressor(n_estimators=200, random_state=42),
}


def get_predictable_columns(roles: dict) -> list:
    """Numeric columns that make sense as a prediction target."""
    return [c for c in roles.get("numeric_columns", []) if "id" not in c.lower()]


def _prepare_features(df: pd.DataFrame, target: str):
    drop_like = [c for c in df.columns if "id" in c.lower() or "date" in c.lower()]
    feature_cols = [c for c in df.columns if c != target and c not in drop_like]

    X = df[feature_cols].copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(0)

    return X, df[target], feature_cols


def train_and_predict(df: pd.DataFrame, target: str, model_name: str = "random_forest") -> dict:
    """Simple, explainable predictive analytics. Given a target column the
    user picks, we train a regression model on the other columns and report
    MAE / RMSE / R2 on a held-out test split. Plain scikit-learn, no
    pipeline framework, so every step is easy to point to and explain."""
    if target not in df.columns:
        return {"available": False, "reason": f"Target column '{target}' not found in the dataset."}
    if not pd.api.types.is_numeric_dtype(df[target]):
        return {"available": False, "reason": "The target column must be numeric for regression."}

    X, y, feature_cols = _prepare_features(df, target)
    if len(X) < 10:
        return {"available": False, "reason": "Not enough rows to train a model (need at least 10)."}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_name = model_name if model_name in MODELS else "random_forest"
    model = MODELS[model_name]()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        feature_importance = {col: round(float(imp), 4) for col, imp in zip(feature_cols, model.feature_importances_)}
        feature_importance = dict(sorted(feature_importance.items(), key=lambda kv: -kv[1])[:8])

    return {
        "available": True,
        "model_used": model_name,
        "target": target,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 3),
        "sample_predictions": [
            {"actual": round(float(a), 2), "predicted": round(float(p), 2)}
            for a, p in list(zip(y_test.tolist(), y_pred.tolist()))[:10]
        ],
        "feature_importance": feature_importance,
    }
