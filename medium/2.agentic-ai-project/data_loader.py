# data_loader.py
# ─────────────────────────────────────────────────────────────────────────────
# Kaggle 이커머스 데이터셋을 다운로드하여 train/ 폴더의 CSV 파일들을
# 로컬 SQLite 데이터베이스로 적재(ingest)하는 스크립트.
#
# 테이블 매핑 (Kaggle train/ 폴더 → SQLite 테이블):
#   orders_dataset.csv       → orders        (주문 정보)
#   order_items_dataset.csv  → order_items   (주문 항목)
#   customers_dataset.csv    → customers     (고객 정보)
#   payments_dataset.csv     → payments      (결제 정보, olist_order_payments_dataset.csv)
#   products_dataset.csv     → products      (상품 정보)
#
# 실행 방법:
#   python data_loader.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sqlite3
import glob
import shutil
import pandas as pd
import kagglehub
from config import DB_PATH, RAW_DATA_DIR

# ── 각 테이블을 식별하기 위한 파일명 키워드 매핑 ───────────────────────────────
# CSV 파일명에 아래 문자열 중 하나라도 포함되어 있으면 해당 테이블로 인식한다.
TABLE_HINTS = {
    "orders":      ["orders_dataset", "orders.csv"],
    "order_items": ["order_items_dataset", "orderitems.csv"],
    "customers":   ["customers_dataset", "customers.csv"],
    "payments":    ["payments_dataset", "payment"],
    "products":    ["products_dataset", "products.csv"],
}


def _find_csv(train_dir: str, hints: list[str]) -> str | None:
    """train_dir 내에서 hints 중 하나라도 파일명에 포함된 첫 번째 CSV 경로를 반환한다."""
    for f in glob.glob(os.path.join(train_dir, "*.csv")):
        fname = os.path.basename(f).lower()
        if any(h.lower() in fname for h in hints):
            return f
    return None


def _resolve_delivery_col(df: pd.DataFrame) -> pd.DataFrame:
    """Kaggle orders 파일마다 배송일 컬럼명이 다를 수 있어 통일된 이름으로 정규화한다."""
    # 이미 표준 컬럼명을 사용하고 있다면 그대로 반환
    if "order_delivered_timestamp" in df.columns:
        return df
    # 후보 컬럼명을 찾아 표준 이름(order_delivered_timestamp)으로 rename
    for candidate in ("order_delivered_customer_date", "order_delivered_carrier_date"):
        if candidate in df.columns:
            df = df.rename(columns={candidate: "order_delivered_timestamp"})
            return df
    return df


def _copy_raw_csvs(source_dir: str, target_dir: str) -> None:
    """다운로드한 데이터셋의 원본 CSV 파일들을 로컬 raw 데이터 폴더로 복사한다."""
    csv_files = glob.glob(os.path.join(source_dir, "*.csv"))
    if not csv_files:
        # 복사할 CSV 가 없는 경우 경고만 출력하고 종료
        print(f"[data_loader] WARNING: No CSV files found in {source_dir} to copy to raw data.")
        return

    # 메타데이터(수정시각 등)를 보존하기 위해 shutil.copy2 사용
    for csv_path in csv_files:
        target_path = os.path.join(target_dir, os.path.basename(csv_path))
        shutil.copy2(csv_path, target_path)


def _find_local_csv_dir() -> str | None:
    """RAW_DATA_DIR (또는 그 하위 train/ 폴더)에 이미 CSV가 있으면 해당 경로를 반환한다."""
    # train/ 하위 폴더를 우선 확인하고, 없으면 RAW_DATA_DIR 자체를 확인
    for path in (os.path.join(RAW_DATA_DIR, "train"), RAW_DATA_DIR):
        if os.path.isdir(path) and glob.glob(os.path.join(path, "*.csv")):
            return path
    return None


def download_and_ingest(force: bool = False) -> str:
    """
    Kaggle 데이터셋을 다운로드(최초 실행 후에는 캐시 사용)하여 DB_PATH 위치의
    SQLite 데이터베이스에 모든 테이블을 적재한다.

    Returns:
        DB_PATH: 적재가 완료된 SQLite 파일 경로
    """
    # DB 파일과 raw 데이터 폴더를 미리 생성
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    raw_train_dir = os.path.join(RAW_DATA_DIR, "train")
    os.makedirs(raw_train_dir, exist_ok=True)

    # 이미 DB 가 존재하는 경우의 처리
    if os.path.exists(DB_PATH):
        if not force:
            # force=False 면 재적재하지 않고 종료 (시간 절약)
            print(f"[data_loader] SQLite DB already exists at {DB_PATH}. Skipping ingest.")
            print("              Pass force=True to re-ingest.")
            return DB_PATH
        # force=True 인 경우 기존 DB 삭제 후 재생성
        os.remove(DB_PATH)
        print(f"[data_loader] force=True → removed existing DB at {DB_PATH}")

    # 로컬에 이미 CSV 가 있으면 Kaggle 다운로드를 건너뛴다
    local_dir = _find_local_csv_dir()
    if local_dir is not None:
        print(f"[data_loader] Using local CSVs at {local_dir}. Skipping Kaggle download.")
        train_dir = local_dir
    else:
        # Kaggle Hub 를 통해 데이터셋 다운로드 (kagglehub 가 캐시 관리)
        print("[data_loader] Downloading Kaggle dataset …")
        kaggle_root = kagglehub.dataset_download("bytadit/ecommerce-order-dataset")
        print(f"[data_loader] Dataset root: {kaggle_root}")

        # 다운로드 디렉터리에서 train/ 하위 폴더 탐색
        train_dir = None
        for root, dirs, _ in os.walk(kaggle_root):
            if "train" in dirs:
                train_dir = os.path.join(root, "train")
                break
            # 데이터셋 버전에 따라 root 자체가 train 폴더일 수 있음
            if os.path.basename(root).lower() == "train":
                train_dir = root
                break

        if train_dir is None:
            # train/ 폴더를 찾지 못하면 root 디렉터리를 직접 사용 (fallback)
            train_dir = kaggle_root
        print(f"[data_loader] Using CSV directory: {train_dir}")

        # 원본 CSV 를 프로젝트 raw 폴더로 복사 (이후 재실행 시 다운로드 생략)
        print(f"[data_loader] Copying raw CSVs to {raw_train_dir} …")
        _copy_raw_csvs(train_dir, raw_train_dir)
        train_dir = raw_train_dir

    # SQLite 연결 후 각 CSV 파일을 테이블로 적재
    with sqlite3.connect(DB_PATH) as conn:
        for table_name, hints in TABLE_HINTS.items():
            csv_path = _find_csv(train_dir, hints)
            if csv_path is None:
                # 해당 테이블용 CSV 를 못 찾으면 경고만 남기고 다음 테이블로 진행
                print(f"[data_loader] WARNING: Could not find CSV for table '{table_name}'. Skipping.")
                continue

            print(f"[data_loader] Loading {os.path.basename(csv_path)} → table '{table_name}' …")
            # low_memory=False: 컬럼 타입 추론 시 dtype 경고 방지
            df = pd.read_csv(csv_path, low_memory=False)

            # 컬럼명 정규화: 소문자 변환 + 양쪽 공백 제거 + 공백을 '_' 로 치환
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

            # orders 테이블은 배송일 컬럼명 통일 처리가 필요
            if table_name == "orders":
                df = _resolve_delivery_col(df)

            # if_exists="replace": 기존 테이블이 있으면 덮어쓴다
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"           → {len(df):,} rows loaded.")

    print(f"[data_loader] ✓ All tables written to {DB_PATH}")
    return DB_PATH


def get_table_schema(table_name: str) -> str:
    """주어진 테이블의 CREATE TABLE SQL 을 반환한다 (프롬프트 구성 시 활용)."""
    with sqlite3.connect(DB_PATH) as conn:
        # sqlite_master 시스템 테이블에서 DDL 조회
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
    return row[0] if row else f"-- Table '{table_name}' not found"


def list_tables() -> list[str]:
    """SQLite 데이터베이스에 존재하는 모든 테이블명을 반환한다."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    return [r[0] for r in rows]


if __name__ == "__main__":
    # 스크립트 직접 실행 시: 데이터 적재 후 테이블 목록과 스키마를 출력
    download_and_ingest()
    print("\nTables in DB:", list_tables())
    for t in list_tables():
        print(f"\n── {t} ──")
        print(get_table_schema(t))