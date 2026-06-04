"""traces.json 파일을 로드하여 타임스탬프를 현재 시각으로 시프트하고,
ID를 새로 생성한 뒤 LangSmith의 RunTree API를 통해 업로드하는 스크립트입니다.

[전체 흐름]
1. JSON 파일에서 trace(실행 기록) 목록을 읽어온다.
2. 가장 최신 start_time을 현재 시각으로 맞추기 위해 모든 타임스탬프에 동일한
   시간 차이(time_delta)를 더한다. (트레이스를 "방금 일어난 것처럼" 보이게 함)
3. 모든 trace의 id / trace_id / parent_run_id 를 uuid7 기반의 새 ID로 재발급한다.
   (uuid7은 시간 순으로 정렬 가능한 UUID 형식)
4. trace_id 기준으로 run들을 묶고, parent-child 트리를 RunTree로 재구성한다.
5. 루트(root) RunTree를 LangSmith로 post 하여 한 번에 업로드한다.
"""

import json
from collections import defaultdict
from datetime import datetime, timezone

# .env 파일에서 LANGSMITH_API_KEY 등 환경 변수를 로드한다.
from dotenv import load_dotenv
load_dotenv()

# LangSmith Python SDK에서 필요한 모듈을 가져온다.
# - Client: LangSmith API와 통신하는 클라이언트
# - uuid7: 시간 정보를 포함한 정렬 가능한 UUID 생성 함수
# - RunTree: 부모-자식 관계를 가진 실행 트리(run tree)를 구성하는 헬퍼 클래스
from langsmith import Client, uuid7
from langsmith.run_trees import RunTree


def parse_dt(s: str | None) -> datetime | None:
    """ISO 형식의 문자열을 datetime 객체로 변환한다.

    - 입력값이 None이면 None을 그대로 반환한다.
    - tzinfo(타임존)가 포함되어 있으면 naive datetime(타임존 미포함)으로
      변환한다. 이후 산술 연산 시 타임존이 섞이는 문제를 방지하기 위함.
    """
    if s is None:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


def main():
    import argparse

    # ------------------------------------------------------------------
    # 1. 커맨드라인 인자 파싱
    # ------------------------------------------------------------------
    # --project : LangSmith에 업로드할 대상 프로젝트명 (기본: "default")
    # --input   : 읽어올 JSON 파일 경로 (기본: "synthetic_traces.json")
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="default", help="업로드 대상 LangSmith 프로젝트명")
    parser.add_argument("--input", default="synthetic_traces.json", help="입력 JSON 파일 경로")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2. JSON 파일에서 trace 목록 로드
    # ------------------------------------------------------------------
    with open(args.input) as f:
        runs = json.load(f)

    print(f"{args.input} 파일에서 {len(runs)}개의 run을 로드했습니다.")

    # ------------------------------------------------------------------
    # 3. 타임스탬프 시프트 양 계산
    # ------------------------------------------------------------------
    # 합성된 트레이스의 가장 늦은 start_time을 찾아, 현재 시각(UTC)과의
    # 차이를 계산한다. 이 값을 모든 run의 시간에 동일하게 더해주면
    # 트레이스가 최근에 발생한 것처럼 LangSmith에 표시된다.
    latest = max(parse_dt(r["start_time"]) for r in runs if r["start_time"])
    time_delta = datetime.now(timezone.utc).replace(tzinfo=None) - latest
    print(f"타임스탬프 시프트 양: {time_delta}")

    # ------------------------------------------------------------------
    # 4. ID 매핑 테이블 생성
    # ------------------------------------------------------------------
    # 기존 JSON에 들어있는 id / trace_id / parent_run_id 값들을 모아
    # 각각에 대해 새로운 uuid7 값을 만들어 매핑한다.
    # - uuid7을 쓰는 이유: 시간 순서대로 정렬 가능 → LangSmith UI에서
    #   trace 정렬이 자연스러워진다.
    # - 같은 ID가 여러 run에서 참조될 수 있으므로(부모-자식 관계),
    #   매핑은 ID 단위로 한 번씩만 생성한다.
    id_map = {}
    for run in runs:
        for field in ("id", "trace_id", "parent_run_id"):
            old_id = run.get(field)
            if old_id and old_id not in id_map:
                id_map[old_id] = str(uuid7())

    # ------------------------------------------------------------------
    # 5. trace_id 기준으로 run 그룹화 및 변환
    # ------------------------------------------------------------------
    # defaultdict(list)를 사용해 trace_id별 run 리스트를 자동으로 만든다.
    # 각 run에 대해 새 ID, 시프트된 타임스탬프 등 RunTree에 필요한
    # 필드만 추출하여 dict로 변환한다.
    traces = defaultdict(list)
    for run in runs:
        traces[run["trace_id"]].append({
            "id": id_map[run["id"]],
            "parent_run_id": id_map.get(run["parent_run_id"]),  # 루트 run이면 None
            "name": run["name"],
            "run_type": run["run_type"],
            "inputs": run["inputs"],
            "outputs": run.get("outputs"),
            "error": run.get("error"),
            "extra": run.get("extra"),
            "tags": run.get("tags"),
            "start_time": parse_dt(run["start_time"]) + time_delta,
            "end_time": parse_dt(run["end_time"]) + time_delta if run.get("end_time") else None,
        })

    # ------------------------------------------------------------------
    # 6. LangSmith 클라이언트 초기화 및 업로드 시작
    # ------------------------------------------------------------------
    client = Client()
    print(f"{len(traces)}개의 trace를 '{args.project}' 프로젝트로 업로드합니다...")

    # 각 trace(= run들의 묶음)를 하나씩 처리한다.
    for i, trace_runs in enumerate(traces.values()):
        # 정렬 키:
        #   (parent_run_id is not None, start_time)
        # → 루트 run(parent_run_id가 None)을 가장 먼저 처리하고,
        #   나머지는 시작 시간 순으로 정렬한다.
        # 이렇게 해야 자식 run을 만들 때 부모 RunTree가 이미 tree_map에 있음.
        trace_runs.sort(key=lambda r: (r["parent_run_id"] is not None, r["start_time"]))

        # tree_map: run["id"] → 해당 RunTree 객체
        # 자식 run이 부모를 찾을 때 사용한다.
        tree_map = {}
        root_tree = None

        # --- 1단계: RunTree 객체 생성 (루트와 자식 분기) ---
        for run in trace_runs:
            if run["parent_run_id"] is None:
                # 루트 run: 새 RunTree를 직접 생성한다.
                root_tree = RunTree(
                    id=run["id"],
                    name=run["name"],
                    run_type=run["run_type"],
                    inputs=run["inputs"],
                    start_time=run["start_time"],
                    extra=run.get("extra"),
                    tags=run.get("tags"),
                    project_name=args.project,
                    client=client,
                )
                tree_map[run["id"]] = root_tree
            else:
                # 자식 run: 부모 RunTree에 create_child()로 붙인다.
                parent = tree_map.get(run["parent_run_id"])
                if parent:
                    child = parent.create_child(
                        name=run["name"],
                        run_type=run["run_type"],
                        run_id=run["id"],
                        inputs=run["inputs"],
                        start_time=run["start_time"],
                        extra=run.get("extra"),
                        tags=run.get("tags"),
                    )
                    tree_map[run["id"]] = child

        # --- 2단계: 각 RunTree에 outputs/error/end_time 적용 ---
        # 자식 → 부모 순으로 end()를 호출하기 위해 reversed로 순회한다.
        # (정렬상 루트가 맨 앞이므로, 뒤집으면 자식이 먼저 처리됨)
        for run in reversed(trace_runs):
            tree = tree_map.get(run["id"])
            if tree:
                tree.end(outputs=run.get("outputs"), error=run.get("error"), end_time=run["end_time"])

        # --- 3단계: 루트 RunTree를 LangSmith에 업로드 ---
        # exclude_child_runs=False 옵션으로 자식 run도 함께 전송된다.
        if root_tree:
            root_tree.post(exclude_child_runs=False)

        # 10개마다 진행 상황 출력
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(traces)} trace 업로드 완료")

    # ------------------------------------------------------------------
    # 7. 백그라운드 업로드 작업 완료 대기
    # ------------------------------------------------------------------
    # LangSmith 클라이언트는 비동기로 데이터를 전송하므로,
    # 모든 요청이 끝날 때까지 flush()로 명시적으로 기다린다.
    print("백그라운드 업로드 작업을 마무리하는 중...")
    client.flush()
    print("완료!")


if __name__ == "__main__":
    main()
