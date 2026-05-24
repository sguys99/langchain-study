#!/usr/bin/env python3
# setup.py
# ─────────────────────────────────────────────────────────────────────────────
# Run this ONCE before starting the agent.
#
# What it does:
#   1. Downloads the Kaggle dataset (bytadit/ecommerce-order-dataset)
#   2. Loads the train/ CSVs into a local SQLite database
#   3. Builds the FAISS vector index from any PDFs in pdf_docs/
#
# Default: skips any step whose output already exists (DB / FAISS index).
#
# Usage:
#   python setup.py                  # skip steps whose output already exists
#   python setup.py --force          # re-ingest DB AND rebuild FAISS index
#   python setup.py --force-db       # re-ingest DB only
#   python setup.py --force-index    # rebuild FAISS index only
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os
import sqlite3
import sys

# Ensure project root is on the path when running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import download_and_ingest, list_tables
from tools.rag_tool import build_index
from config import PDF_DIR, DB_PATH, FAISS_INDEX_DIR, RAW_DATA_DIR


FAISS_INDEX_FILE = os.path.join(FAISS_INDEX_DIR, "index.faiss")


def _mark(exists: bool) -> str:
    return "✓ exists " if exists else "✗ missing"


def print_status() -> tuple[bool, bool]:
    """Print pre-flight status of DB / FAISS / PDFs. Returns (db_exists, idx_exists)."""
    db_exists  = os.path.exists(DB_PATH)
    idx_exists = os.path.exists(FAISS_INDEX_FILE)
    raw_train  = os.path.join(RAW_DATA_DIR, "train")
    raw_exists = os.path.isdir(raw_train) and any(
        f.endswith(".csv") for f in os.listdir(raw_train)
    ) if os.path.isdir(raw_train) else False
    pdf_count  = (
        len([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
        if os.path.isdir(PDF_DIR) else 0
    )

    print("[Status] SQLite DB     :", _mark(db_exists),  f"({DB_PATH})")
    print("[Status] FAISS index   :", _mark(idx_exists), f"({FAISS_INDEX_FILE})")
    print("[Status] Raw CSV cache :", _mark(raw_exists), f"({raw_train})")
    print(f"[Status] PDF docs      : {pdf_count} file(s) in {PDF_DIR}")
    return db_exists, idx_exists


def verify_db() -> None:
    """Print row counts for every table in the DB."""
    tables = list_tables()
    if not tables:
        print("[Verify] DB has no tables — ingestion likely failed.")
        return
    with sqlite3.connect(DB_PATH) as conn:
        for t in tables:
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                print(f"[Verify] {t:<14}: {n:>10,} rows")
            except sqlite3.Error as e:
                print(f"[Verify] {t:<14}: ERROR ({e})")


def verify_index() -> None:
    """Print FAISS index size."""
    if not os.path.exists(FAISS_INDEX_FILE):
        print("[Verify] FAISS index : not found.")
        return
    try:
        import faiss
        index = faiss.read_index(FAISS_INDEX_FILE)
        print(f"[Verify] FAISS index : {index.ntotal:,} vectors (dim={index.d})")
    except Exception as e:
        print(f"[Verify] FAISS index : ERROR ({e})")


def main(force_db: bool, force_index: bool) -> None:
    print("=" * 60)
    print("  E-Commerce Agent — Setup")
    print("=" * 60)

    print_status()

    # ── Step 1: SQLite DB ─────────────────────────────────────────────────────
    print("\n[1/2] Ingesting Kaggle dataset into SQLite …")
    try:
        download_and_ingest(force=force_db)
    except Exception as e:
        print(f"\n[ERROR] SQLite 데이터 적재 실패: {e}")
        print("        Kaggle API 키가 필요할 수 있습니다.")
        print("        설정 안내: https://github.com/Kaggle/kaggle-api#api-credentials")
        sys.exit(1)

    # ── Step 2: FAISS index ───────────────────────────────────────────────────
    print(f"\n[2/2] Building FAISS index from PDFs in '{PDF_DIR}' …")
    build_index(force=force_index)

    # ── Verification ──────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  Verification")
    print("-" * 60)
    verify_db()
    verify_index()

    print("\n" + "=" * 60)
    print("  Setup complete!  Run the agent with:  python main.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="E-Commerce Agent setup. By default, skips steps whose output already exists.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-ingest DB AND rebuild FAISS index even if they already exist.",
    )
    parser.add_argument(
        "--force-db", action="store_true",
        help="Re-ingest the SQLite DB only.",
    )
    parser.add_argument(
        "--force-index", action="store_true",
        help="Rebuild the FAISS index only.",
    )
    args = parser.parse_args()

    main(
        force_db    = args.force or args.force_db,
        force_index = args.force or args.force_index,
    )
