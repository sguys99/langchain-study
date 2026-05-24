# data_loader.py
# ─────────────────────────────────────────────────────────────────────────────
# Downloads the Kaggle e-commerce dataset and ingests the train/ CSVs into
# a local SQLite database.
#
# Table mapping (Kaggle train/ folder → SQLite table):
#   orders_dataset.csv       → orders
#   order_items_dataset.csv  → order_items
#   customers_dataset.csv    → customers
#   payments_dataset.csv     → payments  (olist_order_payments_dataset.csv)
#   products_dataset.csv     → products
#
# Usage:
#   python data_loader.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sqlite3
import glob
import shutil
import pandas as pd
import kagglehub
from config import DB_PATH, RAW_DATA_DIR

# ── File name fragments to detect each table ───────────────────────────────────
TABLE_HINTS = {
    "orders":      ["orders_dataset", "orders.csv"],
    "order_items": ["order_items_dataset", "orderitems.csv"],
    "customers":   ["customers_dataset", "customers.csv"],
    "payments":    ["payments_dataset", "payment"],
    "products":    ["products_dataset", "products.csv"],
}


def _find_csv(train_dir: str, hints: list[str]) -> str | None:
    """Return the first CSV in train_dir whose name contains any of hints."""
    for f in glob.glob(os.path.join(train_dir, "*.csv")):
        fname = os.path.basename(f).lower()
        if any(h.lower() in fname for h in hints):
            return f
    return None


def _resolve_delivery_col(df: pd.DataFrame) -> pd.DataFrame:
    """Kaggle orders files may use different column names for delivery date."""
    if "order_delivered_timestamp" in df.columns:
        return df
    for candidate in ("order_delivered_customer_date", "order_delivered_carrier_date"):
        if candidate in df.columns:
            df = df.rename(columns={candidate: "order_delivered_timestamp"})
            return df
    return df


def _copy_raw_csvs(source_dir: str, target_dir: str) -> None:
    """Copy raw CSVs from the downloaded dataset into the local raw data folder."""
    csv_files = glob.glob(os.path.join(source_dir, "*.csv"))
    if not csv_files:
        print(f"[data_loader] WARNING: No CSV files found in {source_dir} to copy to raw data.")
        return

    for csv_path in csv_files:
        target_path = os.path.join(target_dir, os.path.basename(csv_path))
        shutil.copy2(csv_path, target_path)


def _find_local_csv_dir() -> str | None:
    """Return RAW_DATA_DIR (or its train/ subdir) if it already holds CSVs."""
    for path in (os.path.join(RAW_DATA_DIR, "train"), RAW_DATA_DIR):
        if os.path.isdir(path) and glob.glob(os.path.join(path, "*.csv")):
            return path
    return None


def download_and_ingest(force: bool = False) -> str:
    """
    Download the Kaggle dataset (cached after first run) and write all tables
    into the SQLite database at DB_PATH.

    Returns DB_PATH.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    raw_train_dir = os.path.join(RAW_DATA_DIR, "train")
    os.makedirs(raw_train_dir, exist_ok=True)

    if os.path.exists(DB_PATH):
        if not force:
            print(f"[data_loader] SQLite DB already exists at {DB_PATH}. Skipping ingest.")
            print("              Pass force=True to re-ingest.")
            return DB_PATH
        os.remove(DB_PATH)
        print(f"[data_loader] force=True → removed existing DB at {DB_PATH}")

    local_dir = _find_local_csv_dir()
    if local_dir is not None:
        print(f"[data_loader] Using local CSVs at {local_dir}. Skipping Kaggle download.")
        train_dir = local_dir
    else:
        print("[data_loader] Downloading Kaggle dataset …")
        kaggle_root = kagglehub.dataset_download("bytadit/ecommerce-order-dataset")
        print(f"[data_loader] Dataset root: {kaggle_root}")

        # Find the train/ sub-directory
        train_dir = None
        for root, dirs, _ in os.walk(kaggle_root):
            if "train" in dirs:
                train_dir = os.path.join(root, "train")
                break
            # Some versions have the CSVs directly under kaggle_root/train
            if os.path.basename(root).lower() == "train":
                train_dir = root
                break

        if train_dir is None:
            # Fallback: use root directly if no train sub-folder found
            train_dir = kaggle_root
        print(f"[data_loader] Using CSV directory: {train_dir}")

        print(f"[data_loader] Copying raw CSVs to {raw_train_dir} …")
        _copy_raw_csvs(train_dir, raw_train_dir)
        train_dir = raw_train_dir

    with sqlite3.connect(DB_PATH) as conn:
        for table_name, hints in TABLE_HINTS.items():
            csv_path = _find_csv(train_dir, hints)
            if csv_path is None:
                print(f"[data_loader] WARNING: Could not find CSV for table '{table_name}'. Skipping.")
                continue

            print(f"[data_loader] Loading {os.path.basename(csv_path)} → table '{table_name}' …")
            df = pd.read_csv(csv_path, low_memory=False)

            # Normalise column names: lowercase, strip whitespace
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

            # Fix delivery column name for orders table
            if table_name == "orders":
                df = _resolve_delivery_col(df)

            df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"           → {len(df):,} rows loaded.")

    print(f"[data_loader] ✓ All tables written to {DB_PATH}")
    return DB_PATH


def get_table_schema(table_name: str) -> str:
    """Return CREATE TABLE SQL for a given table (useful for prompt building)."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
    return row[0] if row else f"-- Table '{table_name}' not found"


def list_tables() -> list[str]:
    """Return all table names in the SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    return [r[0] for r in rows]


if __name__ == "__main__":
    download_and_ingest()
    print("\nTables in DB:", list_tables())
    for t in list_tables():
        print(f"\n── {t} ──")
        print(get_table_schema(t))