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
# Usage:
#   python setup.py
#   python setup.py --force    # re-ingest even if DB/index already exist
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os
import sys

# Ensure project root is on the path when running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import download_and_ingest, list_tables
from tools.rag_tool import build_index
from config import PDF_DIR, DB_PATH, FAISS_INDEX_DIR


def main(force: bool = False) -> None:
    print("=" * 60)
    print("  E-Commerce Agent — Setup")
    print("=" * 60)

    # ── Step 1: SQLite DB ─────────────────────────────────────────────────────
    print("\n[1/2] Ingesting Kaggle dataset into SQLite …")
    download_and_ingest(force=force)
    print(f"      Tables in DB: {list_tables()}")

    # ── Step 2: FAISS index ───────────────────────────────────────────────────
    print(f"\n[2/2] Building FAISS index from PDFs in '{PDF_DIR}' …")
    os.makedirs(PDF_DIR, exist_ok=True)
    pdf_count = len([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
    if pdf_count == 0:
        print(f"      No PDFs found in {PDF_DIR}.")
        print("      Add PDF files there and re-run: python setup.py --force")
        print("      The agent will still work for SQL and web-search routes.")
    build_index(force=force)

    print("\n" + "=" * 60)
    print("  Setup complete!  Run the agent with:  python main.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E-Commerce Agent setup")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-ingest data and rebuild FAISS index even if they already exist."
    )
    args = parser.parse_args()
    main(force=args.force)