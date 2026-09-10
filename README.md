# Business Intelligence Copilot

Upload a sales/operations spreadsheet and get an executive dashboard,
forecasts, root cause analysis, anomaly detection, customer intelligence,
opportunity/risk ranking, a what-if simulator, and an AI copilot that
answers questions about your own data.

## Architecture

```
HTML + CSS (server-rendered, no frontend JS)
        |
        v
FastAPI + Jinja2 templates            (app.py)
        |
        v
Python business logic (plain functions, 11 backend files)
        |
   +----+----+
   v         v
 MySQL    Gemini API
(all data) (natural-language answers)
```

- **Frontend:** plain HTML/CSS only. No React, Vite, Node, or JavaScript
  anywhere. Charts are SVG generated server-side in Python (`output.py`).
- **Backend:** FastAPI + Jinja2. Every page is a normal server-rendered
  HTML response; forms submit with regular HTTP POST. Routes are plain
  functions in `app.py` - no routers, no dependency-injection framework,
  no service/repository layers.
- **Database:** MySQL only. Each user's uploaded dataset becomes a real
  MySQL table (`dataset_<user_id>`), created dynamically from the
  cleaned data - not an in-memory DataFrame.
- **AI:** Gemini API, called with a compact, pre-computed summary of your
  data (never the raw dataset).

## 1. Prerequisites

- Python 3.10+
- A running MySQL (or MariaDB) server
- (Optional) a Gemini API key from https://aistudio.google.com/apikey -
  the AI Copilot works without one too, using a simple rule-based
  fallback, but you'll get much better answers with a real key.

## 2. Set up the database

Connect to your MySQL server and create an empty database:

```sql
CREATE DATABASE business_intelligence;
```

The application creates all of its own tables automatically on first run
(see `models.py` - `users`, `datasets`, `dataset_columns`,
`copilot_messages`). You don't need to run any migrations.

## 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your own values:

```env
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=business_intelligence

GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash

SECRET_KEY=change-this-to-a-long-random-string
```

## 4. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Run the app

```bash
uvicorn app:app --reload --port 8000
```

Open http://localhost:8000, sign up, and upload `sample_data/sample_sales_data.csv`
to try every page immediately with realistic data.

## Backend structure (11 Python files)

The backend was consolidated from many small `services/*.py` files into a
small number of files grouped by what they're responsible for. Each file
still has one clear job - it's fewer files, not one giant file.

```
business-intelligence-copilot/
├── app.py               # Every FastAPI route (~20 pages). Plain functions,
│                         # no routers/middleware/schemas beyond the session cookie.
├── database.py           # The ONLY place that knows how to connect to MySQL
├── models.py              # MySQL tables: users, datasets, dataset_columns, copilot_messages
├── auth.py                 # Password hashing (bcrypt) + session login/logout
├── config.py                 # Reads all settings from .env
│
├── data.py                 # Upload -> clean -> detect column roles -> store/load MySQL,
│                         # plus the Data Quality score
├── analytics.py            # KPIs, trends, breakdowns, insights, Business Health score,
│                         # recommendations, and the What-If Simulator
├── customers.py            # RFM segmentation, churn prediction (Random Forest),
│                         # and the Opportunity/Risk ranking built from them
├── prediction.py            # Forecasting, anomaly detection (Isolation Forest),
│                         # root cause analysis, and the generic Predictions page
├── output.py                  # Hand-built SVG charts + Excel/PDF report export
├── ai.py                        # Gemini Copilot (+ rule-based fallback), safe read-only
│                         # SQL execution, and the Autonomous Analyst orchestrator
│
├── requirements.txt
├── .env.example
│
├── templates/            # Jinja2 HTML, one file per page (incl. about.html)
├── static/css/style.css  # The only stylesheet
├── sample_data/          # A ready-to-upload example dataset
└── uploads/              # Scratch folder (uploaded files are read then discarded - the data itself lives in MySQL)
```

Why these six groupings (`data`, `analytics`, `customers`, `prediction`,
`output`, `ai`) and not one file per feature: each one is a single,
explainable responsibility that several pages share, rather than an
arbitrary split.

- **`data.py`** - anything that touches the uploaded file or the raw
  dataset itself (reading it in, cleaning it, scoring its quality,
  saving/loading it from MySQL).
- **`analytics.py`** - anything that turns the dataset into a business
  number shown on the Dashboard/Analytics pages (KPIs, health score,
  recommendations, the simulator).
- **`customers.py`** - everything about customers specifically (RFM,
  churn) plus the opportunity/risk ranking, since both are built
  directly from those two customer analyses.
- **`prediction.py`** - every scikit-learn model in the app (forecast,
  anomaly detection, root cause, generic prediction) - same tools
  (LinearRegression / RandomForest / IsolationForest), same shape of
  "backtest, then predict forward" logic.
- **`output.py`** - the two ways computed numbers leave the page as
  something other than HTML: an inline SVG chart, or a downloadable
  Excel/PDF file.
- **`ai.py`** - the AI Copilot, its safety-checked SQL execution, and the
  Autonomous Analyst (which simply calls the other five files in
  sequence and narrates the result).

## How your data is stored

1. You upload a CSV/XLSX file.
2. `data.py` reads it with pandas, cleans it (drops duplicates, parses
   dates, fills missing values), and detects which columns are
   revenue/product/region/customer/date/etc. using keyword matching.
3. The cleaned data is written to a real MySQL table named
   `dataset_<your_user_id>` (via `DataFrame.to_sql`), replacing any
   previous dataset you had.
4. Every page loads fresh from that MySQL table
   (`pd.read_sql_table(...)`) - MySQL is the permanent source of truth,
   not an in-memory DataFrame.

## Major pages/features

Landing, Sign up/Log in, Upload, Executive Dashboard, Analytics, AI
Copilot, Autonomous Analyst, Root Cause Analysis, Forecast, Predictions,
Anomaly Detection, Customer Intelligence, Opportunity Center, Risk
Center, What-If Simulator, Data Quality, Reports (Excel/PDF export), and
About.

## Notes

- If a page's feature isn't available for your dataset (e.g. no date
  column for forecasting), the page explains what's missing instead of
  erroring.
- The AI Copilot never sends your raw dataset to Gemini - only
  pre-computed totals, top products/regions, and growth rates.
- If Gemini generates a SQL query (used internally by the Copilot), it
  is validated to be a single read-only `SELECT` against your own
  dataset table before it's ever executed - see `is_safe_select()` in
  `ai.py`.
- No SQLite, no Node.js/JavaScript, and no ORM/enterprise architecture
  beyond plain SQLAlchemy models are used anywhere in this project.
