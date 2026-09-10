"""
data.py
--------
Everything about turning an uploaded CSV/Excel file into a clean, stored
MySQL dataset, plus scoring how good that data actually is:

    load_dataframe          read the uploaded file into a pandas DataFrame
    profile_dataframe       rows/columns/missing/duplicates/preview for the Upload page
    clean_dataframe         a small, explainable cleaning pipeline
    detect_column_roles     guess which column is "revenue", "product", etc.
    save_dataset_to_mysql   create a real MySQL table for this dataset and store it
    load_active_dataframe   read a user's active dataset back out of MySQL
    compute_quality_score   0-100 Data Quality Score for the Data Quality page

Each user gets exactly ONE active dataset, stored in a table named
`dataset_<user_id>`. Uploading a new file replaces it - this is the
"simple safe approach for active datasets" the project calls for.
"""

import io
import re

import pandas as pd
from sqlalchemy import text, types as satypes
from sqlalchemy.orm import Session

from database import engine
from models import Dataset, DatasetColumn


# ---------------------------------------------------------------------------
# Step 1: read the raw file
# ---------------------------------------------------------------------------
def load_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file into a DataFrame."""
    if filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))
    return df


# ---------------------------------------------------------------------------
# Step 2: profile the raw data (for the Upload page)
# ---------------------------------------------------------------------------
def profile_dataframe(df: pd.DataFrame) -> dict:
    """Basic profiling info shown on the Upload page before cleaning."""
    missing = df.isnull().sum()
    dtypes = df.dtypes.astype(str)

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "dtypes": dtypes.to_dict(),
        "missing_values": {col: int(v) for col, v in missing.items() if v > 0},
        "total_missing": int(missing.sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "preview": df.head(10).fillna("").astype(str).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Step 3: clean the data
# ---------------------------------------------------------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Small, explainable cleaning pipeline:
    1. Drop exact duplicate rows
    2. Try to parse any column that looks like a date
    3. Fill missing numeric values with the column median
    4. Fill missing text values with "Unknown"
    """
    clean = df.copy()
    clean = clean.drop_duplicates()

    for col in clean.columns:
        if "date" in col.lower():
            clean[col] = pd.to_datetime(clean[col], errors="coerce")

    numeric_cols = clean.select_dtypes(include="number").columns
    for col in numeric_cols:
        if clean[col].isnull().any():
            clean[col] = clean[col].fillna(clean[col].median())

    text_cols = clean.select_dtypes(include="object").columns
    for col in text_cols:
        if clean[col].isnull().any():
            clean[col] = clean[col].fillna("Unknown")

    return clean


# ---------------------------------------------------------------------------
# Step 4: figure out which column plays which business role
# ---------------------------------------------------------------------------
def detect_column_roles(df: pd.DataFrame) -> dict:
    """
    Small keyword-based classifier so the rest of the app never has to
    hard-code a column name like "Revenue" - it asks for roles["revenue_column"]
    instead, and this function figures out which real column that is.
    """
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    numeric_cols = list(df.select_dtypes(include="number").columns)
    category_cols = [
        c for c in df.select_dtypes(include="object").columns if c not in date_cols
    ]

    def find(keywords):
        for c in df.columns:
            if any(k in c.lower() for k in keywords):
                return c
        return None

    return {
        "numeric_columns": numeric_cols,
        "date_columns": date_cols,
        "category_columns": category_cols,
        "revenue_column": find(["revenue", "sales", "amount", "total"]),
        "quantity_column": find(["quantity", "qty", "units"]),
        "product_column": find(["product", "item", "sku"]),
        "region_column": find(["region", "location", "state", "country"]),
        "customer_column": find(["customer", "client"]),
        "category_column": find(["category", "segment"]),
        "date_column": date_cols[0] if date_cols else None,
        "cost_column": find(["cost", "cogs", "expense"]),
        "profit_column": find(["profit", "margin"]),
        "order_id_column": find(["order id", "order_id", "orderid", "transaction"]),
    }


# ---------------------------------------------------------------------------
# Step 5: store the cleaned dataset in MySQL as a real table
# ---------------------------------------------------------------------------
def _safe_column_name(name: str, used_names: set) -> str:
    """Turn an arbitrary uploaded column name into a valid, unique MySQL
    identifier, e.g. "Total Sales ($)" -> "total_sales"."""
    safe = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    if not safe:
        safe = "column"
    if safe[0].isdigit():
        safe = f"c_{safe}"
    safe = safe[:60]

    candidate = safe
    counter = 2
    while candidate in used_names:
        candidate = f"{safe}_{counter}"
        counter += 1
    used_names.add(candidate)
    return candidate


def _sqlalchemy_type_for(series: pd.Series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return satypes.DateTime()
    if pd.api.types.is_bool_dtype(series):
        return satypes.Boolean()
    if pd.api.types.is_integer_dtype(series):
        return satypes.BigInteger()
    if pd.api.types.is_float_dtype(series):
        return satypes.Float()
    return satypes.String(500)


def _invert_roles(roles: dict) -> dict:
    """Turn {"revenue_column": "revenue"} into {"revenue": "revenue_column"}
    (minus the "_column" suffix) so we can label each stored column with
    its business role."""
    inverted = {}
    for key, value in roles.items():
        if key.endswith("_column") and value:
            role_name = key.replace("_column", "")
            inverted[value] = role_name
    return inverted


def save_dataset_to_mysql(db: Session, user_id: int, filename: str, df: pd.DataFrame, roles: dict) -> Dataset:
    """
    Create a fresh MySQL table for this user's dataset and load the cleaned
    DataFrame into it, replacing whatever dataset they had before.
    """
    table_name = f"dataset_{user_id}"

    # Rename columns to safe MySQL identifiers, keeping a map back to the
    # original names so column roles still line up correctly.
    used_names: set = set()
    rename_map = {col: _safe_column_name(col, used_names) for col in df.columns}
    safe_df = df.rename(columns=rename_map)

    # Drop the old table (if any) and create the new one from the DataFrame.
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS `{table_name}`"))

    sqlalchemy_dtype = {
        rename_map[col]: _sqlalchemy_type_for(df[col]) for col in df.columns
    }
    safe_df.to_sql(table_name, con=engine, if_exists="replace", index=False, dtype=sqlalchemy_dtype)

    # Translate the detected roles (original column names) into the safe names.
    safe_roles = {}
    for key, value in roles.items():
        if key.endswith("_columns") and isinstance(value, list):
            safe_roles[key] = [rename_map[v] for v in value]
        elif key.endswith("_column") and value:
            safe_roles[key] = rename_map.get(value)
        else:
            safe_roles[key] = value

    # Replace this user's previous dataset metadata (old table was already dropped above).
    old_dataset_ids = [d.id for d in db.query(Dataset).filter(Dataset.user_id == user_id).all()]
    if old_dataset_ids:
        db.query(DatasetColumn).filter(DatasetColumn.dataset_id.in_(old_dataset_ids)).delete(synchronize_session=False)
        db.query(Dataset).filter(Dataset.user_id == user_id).delete(synchronize_session=False)

    missing = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    dataset = Dataset(
        user_id=user_id,
        filename=filename,
        table_name=table_name,
        row_count=len(safe_df),
        column_count=len(safe_df.columns),
        missing_values=missing,
        duplicate_rows=duplicate_rows,
    )
    db.add(dataset)
    db.flush()  # assigns dataset.id before we insert the column rows

    role_by_column = _invert_roles(safe_roles)
    for col in safe_df.columns:
        db.add(DatasetColumn(
            dataset_id=dataset.id,
            column_name=col,
            data_type=str(safe_df[col].dtype),
            role=role_by_column.get(col),
        ))

    db.commit()
    db.refresh(dataset)
    return dataset


# ---------------------------------------------------------------------------
# Step 6: load a user's active dataset back out of MySQL for analysis
# ---------------------------------------------------------------------------
def get_active_dataset(db: Session, user_id: int):
    return db.query(Dataset).filter(Dataset.user_id == user_id).first()


def load_active_dataframe(db: Session, user_id: int):
    """Load the user's active dataset from MySQL into a DataFrame, along
    with its column roles. Returns (None, None, None) if no dataset yet."""
    dataset = get_active_dataset(db, user_id)
    if not dataset:
        return None, None, None

    df = pd.read_sql_table(dataset.table_name, con=engine)

    columns = db.query(DatasetColumn).filter(DatasetColumn.dataset_id == dataset.id).all()
    roles = _rebuild_roles(columns, df)
    return df, roles, dataset


def _rebuild_roles(columns, df: pd.DataFrame) -> dict:
    """Rebuild the roles dict (same shape as detect_column_roles) from the
    dataset_columns table, using the *safe* column names now in MySQL."""
    roles = {
        "numeric_columns": list(df.select_dtypes(include="number").columns),
        "date_columns": list(df.select_dtypes(include="datetime").columns),
        "category_columns": list(df.select_dtypes(include="object").columns),
    }
    single_roles = [
        "revenue", "quantity", "product", "region", "customer",
        "category", "date", "cost", "profit", "order_id",
    ]
    for role in single_roles:
        roles[f"{role}_column"] = None
    for col in columns:
        if col.role and f"{col.role}_column" in roles:
            roles[f"{col.role}_column"] = col.column_name
    return roles


# ---------------------------------------------------------------------------
# Step 7: score how good the active dataset actually is (Data Quality page)
# ---------------------------------------------------------------------------
def compute_quality_score(df: pd.DataFrame) -> dict:
    """A 0-100 Data Quality Score computed directly from the active dataset -
    no external validation rules, everything measured from the data itself."""
    total_cells = df.shape[0] * df.shape[1]
    missing = int(df.isnull().sum().sum())
    completeness = 100 - (missing / total_cells * 100 if total_cells else 0)

    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = (duplicate_rows / len(df) * 100) if len(df) else 0
    uniqueness = 100 - duplicate_pct

    # Validity: for columns that look like dates, how many actually parsed?
    invalid_date_pct = 0
    date_like_cols = [c for c in df.columns if "date" in c.lower()]
    if date_like_cols:
        col = date_like_cols[0]
        parsed = pd.to_datetime(df[col], errors="coerce")
        invalid_date_pct = round(parsed.isnull().mean() * 100, 1)
    validity = 100 - invalid_date_pct

    # Consistency: category columns with near-duplicate casing/whitespace variants
    category_cols = df.select_dtypes(include="object").columns
    inconsistent_pct = 0
    if len(category_cols) > 0:
        col = category_cols[0]
        raw_unique = df[col].astype(str).nunique()
        normalized_unique = df[col].astype(str).str.strip().str.lower().nunique()
        if raw_unique:
            inconsistent_pct = round((raw_unique - normalized_unique) / raw_unique * 100, 1)
    consistency = 100 - min(inconsistent_pct * 5, 100)

    score = round(completeness * 0.35 + uniqueness * 0.25 + validity * 0.2 + consistency * 0.2, 1)
    score = max(0, min(100, score))

    if score >= 90:
        status = "Excellent"
    elif score >= 75:
        status = "Good"
    elif score >= 55:
        status = "Fair"
    else:
        status = "Poor"

    empty_columns = [c for c in df.columns if df[c].isnull().all()]

    return {
        "score": score,
        "status": status,
        "breakdown": {
            "completeness": round(completeness, 1),
            "uniqueness": round(uniqueness, 1),
            "validity": round(validity, 1),
            "consistency": round(consistency, 1),
        },
        "details": {
            "total_rows": int(len(df)),
            "total_columns": int(df.shape[1]),
            "missing_values": missing,
            "missing_pct": round(missing / total_cells * 100, 2) if total_cells else 0,
            "duplicate_rows": duplicate_rows,
            "duplicate_pct": round(duplicate_pct, 2),
            "invalid_dates_pct": invalid_date_pct,
            "inconsistent_categories_pct": inconsistent_pct,
            "empty_columns": empty_columns,
        },
    }
