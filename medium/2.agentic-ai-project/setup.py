#!/usr/bin/env python3
# setup.py
# ─────────────────────────────────────────────────────────────────────────────
# 에이전트를 처음 실행하기 전에 단 한 번 수행해야 하는 초기화 스크립트.
#
# 수행 작업:
#   1. Kaggle 데이터셋(bytadit/ecommerce-order-dataset)을 내려받음
#   2. train/ 폴더의 CSV 파일들을 로컬 SQLite 데이터베이스에 적재
#   3. pdf_docs/ 디렉터리에 있는 PDF 문서로부터 FAISS 벡터 인덱스를 구축
#
# 기본 동작: 이미 결과물(DB / FAISS 인덱스)이 존재하는 단계는 건너뜀.
#           → 반복 실행 시 불필요한 재작업을 막아 시간을 절약.
#
# 사용법:
#   python setup.py                  # 기존 결과물이 있으면 해당 단계 스킵
#   python setup.py --force          # DB 재적재 AND FAISS 인덱스 재구축 (둘 다)
#   python setup.py --force-db       # SQLite DB만 재적재
#   python setup.py --force-index    # FAISS 인덱스만 재구축
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os
import sqlite3
import sys

# 스크립트를 직접 실행할 때(python setup.py) 프로젝트 루트를 import path에 추가.
# 이렇게 해야 data_loader / tools / config 같은 로컬 모듈을 정상적으로 import 가능.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import download_and_ingest, list_tables
from tools.rag_tool import build_index
from config import PDF_DIR, DB_PATH, FAISS_INDEX_DIR, RAW_DATA_DIR


# FAISS 인덱스 본체 파일(.faiss)의 전체 경로.
# 단순히 디렉터리 존재 여부가 아니라 실제 인덱스 파일 존재 여부로 판단하기 위함.
FAISS_INDEX_FILE = os.path.join(FAISS_INDEX_DIR, "index.faiss")


def _mark(exists: bool) -> str:
    # 상태 표시용 시각적 마커: 존재하면 체크, 아니면 X 표시를 반환.
    return "✓ exists " if exists else "✗ missing"


def print_status() -> tuple[bool, bool]:
    """사전 점검 단계: DB / FAISS 인덱스 / 원본 CSV / PDF 문서의 존재 여부를 출력.

    반환값: (db_exists, idx_exists)
        - db_exists  : SQLite DB 파일 존재 여부
        - idx_exists : FAISS 인덱스 파일 존재 여부
    """
    # 핵심 산출물 두 가지의 존재 여부 확인
    db_exists  = os.path.exists(DB_PATH)
    idx_exists = os.path.exists(FAISS_INDEX_FILE)

    # 원본 CSV 캐시 디렉터리(data/raw/train) 확인
    # — 디렉터리가 있고, 그 안에 적어도 하나의 .csv 파일이 존재해야 "있다"고 판단.
    raw_train  = os.path.join(RAW_DATA_DIR, "train")
    raw_exists = os.path.isdir(raw_train) and any(
        f.endswith(".csv") for f in os.listdir(raw_train)
    ) if os.path.isdir(raw_train) else False

    # RAG 인덱싱 대상 PDF 파일 개수 세기 (대소문자 구분 없이 .pdf 확장자 검사)
    pdf_count  = (
        len([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
        if os.path.isdir(PDF_DIR) else 0
    )

    # 사용자가 한눈에 현재 상태를 파악할 수 있도록 정렬된 형식으로 출력
    print("[Status] SQLite DB     :", _mark(db_exists),  f"({DB_PATH})")
    print("[Status] FAISS index   :", _mark(idx_exists), f"({FAISS_INDEX_FILE})")
    print("[Status] Raw CSV cache :", _mark(raw_exists), f"({raw_train})")
    print(f"[Status] PDF docs      : {pdf_count} file(s) in {PDF_DIR}")
    return db_exists, idx_exists


def verify_db() -> None:
    """SQLite DB의 모든 테이블에 대한 행(row) 개수를 출력하여 적재 결과를 검증.

    - 테이블이 하나도 없으면 적재 실패 가능성이 높으므로 경고 메시지 출력.
    - 각 테이블별로 COUNT(*) 쿼리를 실행해 실제 적재된 데이터 양을 확인.
    """
    tables = list_tables()
    if not tables:
        print("[Verify] DB has no tables — ingestion likely failed.")
        return
    # with 문으로 connection 자동 종료 보장
    with sqlite3.connect(DB_PATH) as conn:
        for t in tables:
            try:
                # 테이블당 한 줄씩, 이름은 왼쪽 정렬 / 행 수는 천단위 콤마 포함 우측 정렬
                n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                print(f"[Verify] {t:<14}: {n:>10,} rows")
            except sqlite3.Error as e:
                # 특정 테이블에서 쿼리 실패해도 다른 테이블 검증은 계속 진행
                print(f"[Verify] {t:<14}: ERROR ({e})")


def verify_index() -> None:
    """FAISS 인덱스의 크기(벡터 개수 / 차원)를 출력하여 구축 결과를 검증."""
    if not os.path.exists(FAISS_INDEX_FILE):
        print("[Verify] FAISS index : not found.")
        return
    try:
        # faiss는 무거운 패키지이므로 필요한 시점에만 지연 import
        import faiss
        index = faiss.read_index(FAISS_INDEX_FILE)
        # ntotal: 인덱스에 저장된 총 벡터 수, d: 각 벡터의 차원
        print(f"[Verify] FAISS index : {index.ntotal:,} vectors (dim={index.d})")
    except Exception as e:
        # 인덱스 파일이 손상되었거나 faiss 버전 호환 문제 등을 안전하게 처리
        print(f"[Verify] FAISS index : ERROR ({e})")


def main(force_db: bool, force_index: bool) -> None:
    # ── 헤더 출력 ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  E-Commerce Agent — Setup")
    print("=" * 60)

    # 작업 시작 전 현재 산출물 상태를 먼저 보여줘서 사용자가 진행 여부를 가늠
    print_status()

    # ── Step 1: SQLite DB 적재 ──────────────────────────────────────────────
    # Kaggle 데이터셋을 받아 train/*.csv → SQLite 테이블로 변환.
    # force_db=True 이면 기존 DB가 있어도 다시 적재.
    print("\n[1/2] Ingesting Kaggle dataset into SQLite …")
    try:
        download_and_ingest(force=force_db)
    except Exception as e:
        # 대부분 Kaggle API 인증 실패가 원인이므로 안내 메시지와 함께 즉시 종료
        print(f"\n[ERROR] SQLite 데이터 적재 실패: {e}")
        print("        Kaggle API 키가 필요할 수 있습니다.")
        print("        설정 안내: https://github.com/Kaggle/kaggle-api#api-credentials")
        sys.exit(1)

    # ── Step 2: FAISS 벡터 인덱스 구축 ──────────────────────────────────────
    # pdf_docs/ 안의 PDF들을 청킹 → 임베딩 → FAISS 인덱스로 저장.
    # force_index=True 이면 기존 인덱스가 있어도 처음부터 다시 구축.
    print(f"\n[2/2] Building FAISS index from PDFs in '{PDF_DIR}' …")
    build_index(force=force_index)

    # ── 검증 단계 ────────────────────────────────────────────────────────────
    # 적재/구축 직후 즉시 결과를 확인해 다음 단계(에이전트 실행) 진입 전에 문제 파악
    print("\n" + "-" * 60)
    print("  Verification")
    print("-" * 60)
    verify_db()
    verify_index()

    # ── 완료 메시지 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Setup complete!  Run the agent with:  python main.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # CLI 인자 파서 구성.
    # 기본 동작은 "이미 결과물이 있으면 건너뜀" 이므로 --force* 플래그로 강제 재실행 제어.
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

    # --force는 두 단계를 모두 강제 실행하는 단축 플래그.
    # 개별 단계만 강제하려면 --force-db / --force-index 사용.
    main(
        force_db    = args.force or args.force_db,
        force_index = args.force or args.force_index,
    )
