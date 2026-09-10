"""
app.py
------
FastAPI entry point. Every route renders a server-side HTML page with
Jinja2 - there is no separate frontend build step and no JSON API for the
browser to call. Forms submit normally (HTTP POST) and the server
re-renders the page with the result.

Run with:  uvicorn app:app --reload --port 8000
"""

from datetime import datetime

from fastapi import FastAPI, Request, Depends, UploadFile, File, Form
from fastapi.responses import RedirectResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session
import io

import config
from database import Base, engine, get_db
from models import User, CopilotMessage
import auth

# Each import below is one backend file, grouped by what it's responsible
# for - see the module docstring at the top of each file for details.
import data          # upload / clean / store dataset in MySQL + data quality score
import analytics     # KPIs, insights, business health score, recommendations, simulator
import customers     # RFM, churn, opportunity & risk ranking
import prediction    # forecasting, anomaly detection, root cause, custom predictions
import output        # SVG charts + Excel/PDF report export
import ai            # Gemini copilot, safe SQL, autonomous analyst

# Create every table that doesn't already exist yet. Safe to call on every
# startup - it never touches tables that are already there.
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Business Intelligence Copilot")
app.add_middleware(SessionMiddleware, secret_key=config.SECRET_KEY)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def base_context(request: Request, user, dataset=None, roles=None) -> dict:
    """Context every template needs: who's logged in, and which sidebar
    sections make sense for the currently loaded dataset."""
    roles = roles or {}
    return {
        "request": request,
        "user": user,
        "dataset": dataset,
        "roles": roles,
        "has_dataset": dataset is not None,
        "has_revenue": bool(roles.get("revenue_column")),
        "has_date": bool(roles.get("date_column")),
        "has_customer": bool(roles.get("customer_column")),
    }


def render(name: str, request: Request, user, dataset=None, roles=None, **extra):
    ctx = base_context(request, user, dataset, roles)
    ctx.update(extra)
    return templates.TemplateResponse(name, ctx)


# ---------------------------------------------------------------------------
# Landing / Auth
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def landing(request: Request, db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    return render("landing.html", request, user)


@app.get("/about", response_class=HTMLResponse)
def about_page(request: Request, db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    return render("about.html", request, user)


@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request, db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if user:
        return RedirectResponse("/upload", status_code=303)
    return render("signup.html", request, None)


@app.post("/signup", response_class=HTMLResponse)
def signup_submit(request: Request, name: str = Form(...), email: str = Form(...),
                   password: str = Form(...), db: Session = Depends(get_db)):
    email = email.strip().lower()
    if db.query(User).filter(User.email == email).first():
        return render("signup.html", request, None, error="An account with that email already exists.")
    if len(password) < 6:
        return render("signup.html", request, None, error="Password must be at least 6 characters.")

    user = User(name=name.strip(), email=email, password_hash=auth.hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)

    auth.log_in_user(request, user)
    return RedirectResponse("/upload", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if user:
        return RedirectResponse("/upload", status_code=303)
    return render("login.html", request, None)


@app.post("/login", response_class=HTMLResponse)
def login_submit(request: Request, email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email.strip().lower()).first()
    if not user or not auth.verify_password(password, user.password_hash):
        return render("login.html", request, None, error="Incorrect email or password.")

    auth.log_in_user(request, user)
    return RedirectResponse("/dashboard", status_code=303)


@app.get("/logout")
def logout(request: Request):
    auth.log_out_user(request)
    return RedirectResponse("/", status_code=303)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request, db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)
    dataset = data.get_active_dataset(db, user.id)
    return render("upload.html", request, user, dataset)


@app.post("/upload", response_class=HTMLResponse)
async def upload_submit(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    filename = file.filename or ""
    if not filename.lower().endswith((".csv", ".xlsx", ".xls")):
        return render("upload.html", request, user, error="Please upload a .csv, .xlsx, or .xls file.")

    contents = await file.read()
    if not contents:
        return render("upload.html", request, user, error="The uploaded file is empty.")

    try:
        raw_df = data.load_dataframe(contents, filename)
    except Exception as exc:
        return render("upload.html", request, user, error=f"Could not read that file: {exc}")

    if raw_df is None or raw_df.empty:
        return render("upload.html", request, user, error="No rows could be read from that file.")

    profile = data.profile_dataframe(raw_df)
    clean_df = data.clean_dataframe(raw_df)
    roles = data.detect_column_roles(clean_df)

    try:
        dataset = data.save_dataset_to_mysql(db, user.id, filename, clean_df, roles)
    except Exception as exc:
        return render("upload.html", request, user, error=f"Could not save dataset to the database: {exc}")

    return render("upload.html", request, user, dataset, roles, profile=profile, uploaded=True)


# ---------------------------------------------------------------------------
# Executive Dashboard
# ---------------------------------------------------------------------------
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request, db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    df, roles, dataset = data.load_active_dataframe(db, user.id)
    if df is None:
        return render("dashboard.html", request, user)

    kpis = analytics.calculate_kpis(df, roles)
    insights = analytics.generate_insights(df, roles, kpis)
    trend = analytics.revenue_trend(df, roles)
    top_products = analytics.breakdown_by(df, roles.get("product_column"), roles.get("revenue_column"), top_n=8)
    top_regions = analytics.breakdown_by(df, roles.get("region_column"), roles.get("revenue_column"), top_n=8)
    top_categories = analytics.breakdown_by(df, roles.get("category_column"), roles.get("revenue_column"), top_n=8)
    customer_dist = analytics.category_distribution(df, roles.get("customer_column"))

    anomalies = prediction.detect_anomalies(df, roles)
    churn = customers.predict_churn(df, roles)
    forecast = prediction.forecast_metric(df, roles)
    health = analytics.compute_health_score(kpis, anomalies, churn, forecast, len(df))
    rfm = customers.rfm_analysis(df, roles)
    opportunities = customers.get_opportunities(df, roles, rfm, churn)
    risks = customers.get_risks(df, roles, kpis, churn, anomalies, forecast)
    recommendations = analytics.generate_recommendations(kpis, opportunities, risks, top_products)

    charts = {
        "revenue_trend": output.line_chart(trend, value_key="value", label_key="label"),
        "top_products": output.bar_chart(top_products),
        "top_regions": output.bar_chart(top_regions),
        "customer_dist": output.donut_chart(customer_dist),
    }

    return render(
        "dashboard.html", request, user, dataset, roles,
        kpis=kpis, insights=insights, top_products=top_products, top_regions=top_regions,
        top_categories=top_categories, health=health, charts=charts, recommendations=recommendations,
    )


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
@app.get("/analytics", response_class=HTMLResponse)
def analytics_page(request: Request, db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    df, roles, dataset = data.load_active_dataframe(db, user.id)
    if df is None:
        return render("analytics.html", request, user)

    stats = analytics.summary_statistics(df)
    correlation = analytics.correlation_matrix(df)
    category_dist = analytics.category_distribution(df, roles.get("category_column"))
    quality = data.compute_quality_score(df)

    charts = {"category_dist": output.bar_chart(category_dist)}

    return render(
        "analytics.html", request, user, dataset, roles,
        stats=stats, correlation=correlation, quality=quality, charts=charts,
    )


# ---------------------------------------------------------------------------
# AI Copilot
# ---------------------------------------------------------------------------
@app.get("/copilot", response_class=HTMLResponse)
def copilot_page(request: Request, db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    dataset = data.get_active_dataset(db, user.id)
    history = (
        db.query(CopilotMessage)
        .filter(CopilotMessage.user_id == user.id)
        .order_by(CopilotMessage.created_at.desc())
        .limit(10)
        .all()
    )
    return render("copilot.html", request, user, dataset, history=list(reversed(history)), gemini_configured=bool(config.GEMINI_API_KEY))


@app.post("/copilot", response_class=HTMLResponse)
def copilot_ask(request: Request, question: str = Form(...), db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    df, roles, dataset = data.load_active_dataframe(db, user.id)
    if df is None:
        return RedirectResponse("/upload", status_code=303)

    kpis = analytics.calculate_kpis(df, roles)
    insights = analytics.generate_insights(df, roles, kpis)
    top_products = analytics.breakdown_by(df, roles.get("product_column"), roles.get("revenue_column"), top_n=5)
    top_regions = analytics.breakdown_by(df, roles.get("region_column"), roles.get("revenue_column"), top_n=5)
    top_categories = analytics.breakdown_by(df, roles.get("category_column"), roles.get("revenue_column"), top_n=5)
    forecast = prediction.forecast_metric(df, roles)
    anomalies = prediction.detect_anomalies(df, roles)

    context = ai.build_business_context(
        kpis, insights, top_products, top_regions, top_categories, forecast, anomalies.get("anomaly_count", 0)
    )
    result = ai.ask_copilot(question, context, kpis, top_products, top_regions)

    message = CopilotMessage(user_id=user.id, question=question, answer=result["answer"], source=result["source"])
    db.add(message)
    db.commit()

    return RedirectResponse("/copilot", status_code=303)


# ---------------------------------------------------------------------------
# Autonomous Analyst
# ---------------------------------------------------------------------------
@app.get("/analyst", response_class=HTMLResponse)
def analyst_page(request: Request, db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    df, roles, dataset = data.load_active_dataframe(db, user.id)
    if df is None:
        return render("analyst.html", request, user)

    report = ai.run_full_analysis(df, roles)
    return render("analyst.html", request, user, dataset, roles, report=report)


# ---------------------------------------------------------------------------
# Root Cause Analysis
# ---------------------------------------------------------------------------
@app.get("/root-cause", response_class=HTMLResponse)
def root_cause_page(request: Request, metric: str = None, db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    df, roles, dataset = data.load_active_dataframe(db, user.id)
    if df is None:
        return render("root_cause.html", request, user)

    result = prediction.analyze_kpi_change(df, roles, metric_col=metric)
    return render(
        "root_cause.html", request, user, dataset, roles,
        result=result, numeric_columns=roles.get("numeric_columns", []), selected_metric=metric or roles.get("revenue_column"),
    )


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------
@app.get("/forecast", response_class=HTMLResponse)
def forecast_page(request: Request, metric: str = None, periods: int = 6, db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    df, roles, dataset = data.load_active_dataframe(db, user.id)
    if df is None:
        return render("forecast.html", request, user)

    result = prediction.forecast_metric(df, roles, metric_col=metric, periods=periods)
    chart = ""
    if result.get("available"):
        chart = output.forecast_chart(result["history"], result["forecast"])

    return render(
        "forecast.html", request, user, dataset, roles,
        result=result, chart=chart, numeric_columns=roles.get("numeric_columns", []),
        selected_metric=metric or roles.get("revenue_column"), periods=periods,
    )


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
@app.get("/predictions", response_class=HTMLResponse)
def predictions_page(request: Request, db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    df, roles, dataset = data.load_active_dataframe(db, user.id)
    if df is None:
        return render("predictions.html", request, user)

    targets = prediction.get_predictable_columns(roles)
    return render("predictions.html", request, user, dataset, roles, targets=targets, result=None)


@app.post("/predictions", response_class=HTMLResponse)
def predictions_submit(request: Request, target: str = Form(...), model: str = Form("random_forest"), db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    df, roles, dataset = data.load_active_dataframe(db, user.id)
    if df is None:
        return RedirectResponse("/upload", status_code=303)

    targets = prediction.get_predictable_columns(roles)
    result = prediction.train_and_predict(df, target, model)
    return render("predictions.html", request, user, dataset, roles, targets=targets, result=result, selected_target=target, selected_model=model)


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------
@app.get("/anomalies", response_class=HTMLResponse)
def anomalies_page(request: Request, db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    df, roles, dataset = data.load_active_dataframe(db, user.id)
    if df is None:
        return render("anomalies.html", request, user)

    result = prediction.detect_anomalies(df, roles)
    chart = output.scatter_chart(result.get("chart_data", []))
    return render("anomalies.html", request, user, dataset, roles, result=result, chart=chart)


# ---------------------------------------------------------------------------
# Customer Intelligence
# ---------------------------------------------------------------------------
@app.get("/customers", response_class=HTMLResponse)
def customers_page(request: Request, db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    df, roles, dataset = data.load_active_dataframe(db, user.id)
    if df is None:
        return render("customers.html", request, user)

    rfm = customers.rfm_analysis(df, roles)
    churn = customers.predict_churn(df, roles)
    segment_chart = output.donut_chart(
        [{"name": k, "value": v} for k, v in rfm.get("segment_counts", {}).items()]
    ) if rfm.get("available") else ""

    return render("customers.html", request, user, dataset, roles, rfm=rfm, churn=churn, segment_chart=segment_chart)


# ---------------------------------------------------------------------------
# Opportunity Center
# ---------------------------------------------------------------------------
@app.get("/opportunities", response_class=HTMLResponse)
def opportunities_page(request: Request, db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    df, roles, dataset = data.load_active_dataframe(db, user.id)
    if df is None:
        return render("opportunities.html", request, user)

    kpis = analytics.calculate_kpis(df, roles)
    rfm = customers.rfm_analysis(df, roles)
    churn = customers.predict_churn(df, roles)
    result = customers.get_opportunities(df, roles, rfm, churn)
    return render("opportunities.html", request, user, dataset, roles, result=result)


# ---------------------------------------------------------------------------
# Risk Center
# ---------------------------------------------------------------------------
@app.get("/risks", response_class=HTMLResponse)
def risks_page(request: Request, db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    df, roles, dataset = data.load_active_dataframe(db, user.id)
    if df is None:
        return render("risks.html", request, user)

    kpis = analytics.calculate_kpis(df, roles)
    churn = customers.predict_churn(df, roles)
    anomalies = prediction.detect_anomalies(df, roles)
    forecast = prediction.forecast_metric(df, roles)
    result = customers.get_risks(df, roles, kpis, churn, anomalies, forecast)
    return render("risks.html", request, user, dataset, roles, result=result)


# ---------------------------------------------------------------------------
# What-If Simulator
# ---------------------------------------------------------------------------
@app.get("/simulator", response_class=HTMLResponse)
def simulator_page(request: Request, db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    df, roles, dataset = data.load_active_dataframe(db, user.id)
    if df is None:
        return render("simulator.html", request, user)

    result = analytics.simulate(df, roles, 0, 0, 0)
    return render("simulator.html", request, user, dataset, roles, result=result, inputs={"price_change_pct": 0, "demand_change_pct": 0, "cost_change_pct": 0})


@app.post("/simulator", response_class=HTMLResponse)
def simulator_submit(request: Request, price_pct: float = Form(0), demand_pct: float = Form(0), cost_pct: float = Form(0), db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    df, roles, dataset = data.load_active_dataframe(db, user.id)
    if df is None:
        return RedirectResponse("/upload", status_code=303)

    result = analytics.simulate(df, roles, price_pct, demand_pct, cost_pct)
    return render(
        "simulator.html", request, user, dataset, roles, result=result,
        inputs={"price_change_pct": price_pct, "demand_change_pct": demand_pct, "cost_change_pct": cost_pct},
    )


# ---------------------------------------------------------------------------
# Data Quality
# ---------------------------------------------------------------------------
@app.get("/data-quality", response_class=HTMLResponse)
def data_quality_page(request: Request, db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    df, roles, dataset = data.load_active_dataframe(db, user.id)
    if df is None:
        return render("data_quality.html", request, user)

    result = data.compute_quality_score(df)
    return render("data_quality.html", request, user, dataset, roles, result=result)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
@app.get("/reports", response_class=HTMLResponse)
def reports_page(request: Request, db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    dataset = data.get_active_dataset(db, user.id)
    return render("reports.html", request, user, dataset)


def _report_ingredients(db: Session, user_id: int):
    df, roles, dataset = data.load_active_dataframe(db, user_id)
    kpis = analytics.calculate_kpis(df, roles)
    insights = analytics.generate_insights(df, roles, kpis)
    top_products = analytics.breakdown_by(df, roles.get("product_column"), roles.get("revenue_column"), top_n=10)
    top_regions = analytics.breakdown_by(df, roles.get("region_column"), roles.get("revenue_column"), top_n=10)
    rfm = customers.rfm_analysis(df, roles)
    churn = customers.predict_churn(df, roles)
    opportunities = customers.get_opportunities(df, roles, rfm, churn)
    risks = customers.get_risks(df, roles, kpis, churn, prediction.detect_anomalies(df, roles), prediction.forecast_metric(df, roles))
    recommendations = analytics.generate_recommendations(kpis, opportunities, risks, top_products)
    return kpis, insights, top_products, top_regions, recommendations


@app.get("/reports/download/excel")
def download_excel_report(request: Request, db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    kpis, insights, top_products, top_regions, recommendations = _report_ingredients(db, user.id)
    file_bytes = output.build_excel_report(kpis, insights, top_products, top_regions, recommendations)
    filename = f"bi-copilot-report-{datetime.utcnow().strftime('%Y%m%d')}.xlsx"
    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/reports/download/pdf")
def download_pdf_report(request: Request, db: Session = Depends(get_db)):
    user = auth.get_logged_in_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    kpis, insights, top_products, top_regions, recommendations = _report_ingredients(db, user.id)
    file_bytes = output.build_pdf_report(kpis, insights, top_products, top_regions, recommendations)
    filename = f"bi-copilot-report-{datetime.utcnow().strftime('%Y%m%d')}.pdf"
    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
