"""OfficeFlow Supply Co. 고객 지원용 합성 trace 1000건을 생성하는 스크립트.

synthetic_traces.json 파일에 1000개의 trace(카테고리당 200건)를 생성한다.
  - inventory          : 상품 재고/취급 여부 질문
  - policy             : 회사 정책 관련 질문 (반품, 배송 등)
  - out_of_scope       : Emma가 도와줄 수 없는 범위 외 질문
  - both               : 한 메시지에 inventory + policy 가 같이 들어 있는 경우
  - website_troubleshooting : 웹사이트/포털 사용 중 발생한 문제

각 trace 는 최소 2개의 run으로 구성된다:
  - 루트 "Emma" chain run
  - 자식 "ChatAnthropic" llm run (필요 시 tool call run 추가)

[모델] 이 데이터셋은 Anthropic Claude(claude-haiku-4-5) 호출을 기준으로 한
합성 trace 다. (LangSmith trace 구조는 OpenAI chat completion 포맷을
그대로 따른다. ls_provider 만 "anthropic"으로 표기.)
"""

import json
import random
import uuid
from datetime import datetime, timezone, timedelta

# ---------------------------------------------------------------------------
# \uc0c1\uc218 \uc815\uc758
# ---------------------------------------------------------------------------
# \ud2b8\ub808\uc774\uc2a4 \uce74\ud14c\uace0\ub9ac 5\uc885. \uac01 \uce74\ud14c\uace0\ub9ac\ubcc4\ub85c TRACES_PER_CATEGORY \ub9cc\ud07c \uc0dd\uc131\ud55c\ub2e4.
CATEGORIES = ["inventory", "policy", "out_of_scope", "both", "website_troubleshooting"]
TRACES_PER_CATEGORY = 200          # \uce74\ud14c\uace0\ub9ac\ub2f9 trace \uac1c\uc218 (\ucd1d 1000\uac74)
OUTPUT_FILE = "synthetic_traces.json"   # \uacb0\uacfc JSON \ud30c\uc77c\uba85

# Emma \uc5d0\uc774\uc804\ud2b8\uc758 \uc2dc\uc2a4\ud15c \ud504\ub86c\ud504\ud2b8(\ud55c\uad6d\uc5b4). LLM run\uc758 messages[0] \ub85c \ub4e4\uc5b4\uac04\ub2e4.
# \uc774 \ud504\ub86c\ud504\ud2b8\ub294 Emma \uc758 \uc5ed\ud560/\ucc45\uc784/\ud55c\uacc4\ub97c \uc815\uc758\ud558\uba70, \ubaa8\ub4e0 trace \uc5d0\uc11c \ub3d9\uc77c\ud558\uac8c \uc0ac\uc6a9\ub41c\ub2e4.
SYSTEM_PROMPT = (
    "\ub2f9\uc2e0\uc740 OfficeFlow Supply Co.\uc758 \uace0\uac1d \uc9c0\uc6d0 \uc804\ubb38\uac00 'Emma' \uc785\ub2c8\ub2e4. "
    "OfficeFlow Supply Co.\ub294 \uad6d\ub0b4 \uc911\uc18c\uae30\uc5c5\uc744 \ub300\uc0c1\uc73c\ub85c \uc885\uc774 \ubc0f \uc0ac\ubb34\uc6a9\ud488\uc744 "
    "\uc720\ud1b5\ud558\ub294 \ud68c\uc0ac\uc785\ub2c8\ub2e4.\n\n"
    "[\ub2f9\uc2e0\uc758 \uc5ed\ud560]\n"
    "\ub2f9\uc2e0\uc740 \uace0\uac1d \uacbd\ud5d8(Customer Experience) \ud300 \uc18c\uc18d\uc774\uba70 OfficeFlow \uc5d0\uc11c "
    "3\ub144\uc9f8 \uadfc\ubb34\ud558\uace0 \uc788\uc2b5\ub2c8\ub2e4. \ub2f9\uc2e0\uc740 \uce5c\uc808\ud558\uace0 \ud6a8\uc728\uc801\uc774\uba70, \uace0\uac1d\uc758 \ubb38\uc81c\ub97c "
    "\uc9c4\uc2ec\uc73c\ub85c \ud574\uacb0\ud574 \uc8fc\ub294 \uc0ac\ub78c\uc73c\ub85c \uc54c\ub824\uc838 \uc788\uc2b5\ub2c8\ub2e4. \ub9e4\ub2c8\uc800\ub294 \ubaa8\ub4e0 \uc0c1\ub2f4\uc774 "
    "\uace0\uac1d\uc758 \uc2e0\ub8b0\uc640 \ucda9\uc131\ub3c4\ub97c \uc313\uc744 \uae30\ud68c\uc784\uc744 \ud56d\uc0c1 \uac15\uc870\ud569\ub2c8\ub2e4.\n\n"
    "[\ub2f9\uc2e0\uc774 \ub3c4\uc640\uc904 \uc218 \uc788\ub294 \uc77c]\n"
    "\u2713 \uc0c1\ud488 \uc815\ubcf4 \uc548\ub0b4 - \uc0ac\ubb34\uc6a9\ud488, \uc885\uc774\ub958, \ud544\uae30\uad6c, \uc815\ub9ac\uc6a9\ud488, \ucc45\uc0c1 \uc561\uc138\uc11c\ub9ac "
    "\ub4f1 \uce74\ud0c8\ub85c\uadf8\uc5d0 \ub300\ud55c \ubb38\uc758 \ub2f5\ubcc0\n"
    "\u2713 \uc7ac\uace0 \ubc0f \ucde8\uae09 \uc5ec\ubd80 \ud655\uc778 - \ud604\uc7ac \uc7ac\uace0\ub97c \uc870\ud68c\ud558\uc5ec \uace0\uac1d\uc774 \uc6d0\ud558\ub294 \uc0c1\ud488\uc744 "
    "\ucc3e\ub3c4\ub85d \ub3c4\uc6c0\n"
    "\u2713 \uc0c1\ud488 \ucd94\ucc9c - \uace0\uac1d\uc758 \ud544\uc694, \uc0ac\uc6a9 \ud328\ud134, \uc608\uc0b0\uc5d0 \ub9de\ub294 \uc0c1\ud488 \ucd94\ucc9c\n"
    "\u2713 \uc77c\ubc18 \ubb38\uc758 \uc751\ub300 - \ud68c\uc0ac, \uc0c1\ud488 \ub77c\uc778\uc5c5, \uc11c\ube44\uc2a4 \uad00\ub828 \uc77c\ubc18 \uc9c8\ubb38 \uc751\ub300\n\n"
    "[\ub2f9\uc2e0\uc774 \uc9c1\uc811 \ucc98\ub9ac\ud560 \uc218 \uc5c6\ub294 \uc77c]\n"
    "\u2717 \uc8fc\ubb38 \uc811\uc218 - \uc0c1\ud488 \uc815\ubcf4 \uc81c\uacf5\uc740 \uac00\ub2a5\ud558\uc9c0\ub9cc, \uc2e4\uc81c \uc8fc\ubb38\uc740 \uc6f9 \ud3ec\ud138 \ub610\ub294 "
    "\uc601\uc5c5\ud300(sales@officeflow.com)\uc744 \ud1b5\ud574 \uc9c4\ud589\ub429\ub2c8\ub2e4.\n"
    "\u2717 \uc8fc\ubb38 \uc0c1\ud0dc \ud655\uc778 \ubc0f \ubc30\uc1a1 \uc870\ud68c - \uace0\uac1d \ud3ec\ud138 \ub610\ub294 fulfillment@officeflow.com "
    "\uc73c\ub85c \uc548\ub0b4\ud569\ub2c8\ub2e4.\n"
    "\u2717 \ubc18\ud488 \ubc0f \ud658\ubd88 - \ubc18\ud488\ud300(returns@officeflow.com)\uc758 \uc2b9\uc778\uc774 \ud544\uc694\ud569\ub2c8\ub2e4.\n"
    "\u2717 \uacc4\uc815 \ubcc0\uacbd - \uacb0\uc81c \uc815\ubcf4, \uacb0\uc81c \uc218\ub2e8, \uacc4\uc815 \uc124\uc815\uc740 "
    "accounts@officeflow.com \uc73c\ub85c \uc548\ub0b4\ud569\ub2c8\ub2e4.\n"
    "\u2717 \uae30\uc220 \uc9c0\uc6d0 - \uc6f9\uc0ac\uc774\ud2b8 \uad00\ub828 \ubb38\uc81c\ub294 support@officeflow.com \uc73c\ub85c \uc548\ub0b4\ud569\ub2c8\ub2e4."
)

# 상품 카탈로그 (template의 {product}, {product2}, {product3} 자리에 들어감).
# 한국 사무환경에서 자주 쓰이는 사무용품 중심으로 구성.
PRODUCTS = [
    "복사용지 500매", "블루 볼펜 (12자루)",
    "스테이플러 세트", "스프링 노트 (3권)",
    "마닐라 서류 폴더 (25개)", "포스트잇 메모지 (4개)",
    "보드마카 (8개)", "클립 (100개)",
    "더블 클립 (24개)", "리갈 패드 (6권)",
    "형광펜 (5개)", "데스크 정리함",
    "수정테이프 (3개)", "사무용 가위 8인치",
    "고무밴드 모음팩", "테이프 디스펜서",
    "DL 봉투 (100매)", "인덱스 카드 (100매)",
    "화이트보드 지우개", "압정 (100개)",
    "걸이형 서류 폴더 (25개)", "투명 비닐 속지 (50매)",
    "샤프 (12자루)", "흑색 젤펜 (6자루)",
    "2026 탁상 달력", "2단 서류 트레이",
    "딱풀 (6개)", "포스트잇 플래그 (4색)",
    "코팅 필름 (50매)", "기본 클립보드",
]

# 한국 이름 풀. template의 {name} 자리에 들어감.
NAMES = [
    "민준", "서연", "도윤", "지우", "예준", "서윤", "주원", "하은",
    "지호", "수아", "현우", "지유", "건우", "다은", "우진", "유나",
    "선우", "윤서", "준서", "지민", "은우", "예린", "시우", "수빈",
    "준우", "지아", "도현", "유진", "재윤", "혜린",
]

# 한국 회사명 풀. template의 {company} 자리에 들어감.
COMPANY_NAMES = [
    "한빛 어소시에이츠", "서밋 그룹", "케스케이드 오피스 솔루션",
    "브라이트 패스 컨설팅", "메트로 법무법인", "그린리프 회계법인",
    "피너클 디자인", "하버 헬스 클리닉", "이스트사이드 부동산 관리",
    "노바 금융그룹", "크레스트우드 엔지니어링", "레이크사이드 마케팅",
    "레드우드 컨설팅", "스털링 엔터프라이즈", "아틀라스 물류",
    "컴퍼스 부동산", "아이언브릿지 캐피털", "오크몬트 파트너스",
    "사파이어 테크놀로지스", "웨스트레이크 어드바이저리",
]


# --- 랜덤 값 생성 헬퍼들 ---
# 템플릿의 placeholder({order}, {product}, {name}, {company} 등)에
# 채워 넣을 임의의 값을 만들어 준다.

def _rand_order():
    """가짜 주문번호 (예: 'ORD-3724')."""
    return f"ORD-{random.randint(1000, 9999)}"


def _rand_product():
    """PRODUCTS 리스트에서 임의의 상품 하나 선택."""
    return random.choice(PRODUCTS)


def _rand_products(n=2):
    """PRODUCTS 리스트에서 중복 없이 n개 상품 선택."""
    return random.sample(PRODUCTS, min(n, len(PRODUCTS)))


def _rand_name():
    """NAMES 리스트에서 임의의 한국 이름 하나 선택."""
    return random.choice(NAMES)


def _rand_company():
    """COMPANY_NAMES 리스트에서 임의의 회사명 하나 선택."""
    return random.choice(COMPANY_NAMES)


# ---------------------------------------------------------------------------
# 카테고리별 질문 / 응답 템플릿
# ---------------------------------------------------------------------------
# 각 카테고리별 함수는 (질문, 응답) 튜플 리스트를 반환한다.
# 템플릿 안의 {product}, {name}, {company}, {order} 등은 호출 시점에
# 임의 값으로 치환된다.

def _inventory_questions():
    """inventory 카테고리: 재고/취급 여부에 대한 (질문, 응답) 튜플 리스트 반환."""
    templates = [
        (
            "{product} \uc7ac\uace0 \uc788\ub098\uc694?",
            "\ub124, {product}\ub294 \ud604\uc7ac \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4! \ub2e4\ub978 \ub3c4\uc6c0\uc774 \ud544\uc694\ud558\uc2e0\uac00\uc694?"
        ),
        (
            "{product} \uc7ac\uace0 \uc880 \ud655\uc778\ud574\uc8fc\uc138\uc694. \ucde8\uae09\ud558\uc2dc\ub294 \uc0c1\ud488 \ub9de\ub098\uc694?",
            "{product}\ub294 \ucde8\uae09 \uc911\uc774\uace0 \uc7ac\uace0\ub3c4 \uc788\uc2b5\ub2c8\ub2e4. \ub2e4\ub978 \uc0c1\ud488 \uc815\ubcf4\ub3c4 \ud544\uc694\ud558\uc2dc\uba74 \ub9d0\uc500\ud574\uc8fc\uc138\uc694!"
        ),
        (
            "{product} \uc9c0\uae08 \uad6c\ub9e4 \uac00\ub2a5\ud55c\uc9c0 \uc54c\ub824\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\uc88b\uc740 \uc18c\uc2dd\uc774\uc5d0\uc694 \u2014 {product}\ub294 \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4. \ub2e4\ub978 \ub3c4\uc6c0 \ud544\uc694\ud558\uc2e0 \ubd80\ubd84 \uc788\uc744\uae4c\uc694?"
        ),
        (
            "\ud639\uc2dc OfficeFlow \uc5d0\uc11c {product} \ucde8\uae09\ud558\uc2dc\ub098\uc694?",
            "\ub124 \ucde8\uae09\ud569\ub2c8\ub2e4! {product}\ub294 \uc800\ud76c \uce74\ud0c8\ub85c\uadf8\uc5d0 \uc788\uace0 \ud604\uc7ac \uc7ac\uace0\ub3c4 \uc788\uc2b5\ub2c8\ub2e4."
        ),
        (
            "{product}\ub97c \ucc3e\uace0 \uc788\ub294\ub370 \ud310\ub9e4\ud558\uc2dc\ub294 \uc0c1\ud488\uc778\uac00\uc694?",
            "\ubb3c\ub860\uc785\ub2c8\ub2e4! {product}\ub294 \uc800\ud76c \ucde8\uae09 \uc0c1\ud488\uc774\uace0 \uc9c0\uae08 \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4."
        ),
        (
            "{product} \uc7ac\uace0 \uc5b4\ub5a4\uac00\uc694? 50\uac1c \uc815\ub3c4 \ud544\uc694\ud569\ub2c8\ub2e4.",
            "{product}\ub294 \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4. 50\uac1c \uc815\ub3c4\uc758 \ub300\ub7c9 \uc8fc\ubb38\uc774\ub77c\uba74 \uc6f9 \ud3ec\ud138\uc744 \ud1b5\ud574 \uc9c4\ud589\ud558\uc2dc\uac70\ub098 sales@officeflow.com \uc73c\ub85c \ubb38\uc758 \uc8fc\uc138\uc694."
        ),
        (
            "{product} \uc7ac\uace0\uac00 \ub5a8\uc5b4\uc9c0\uae30 \uc804\uc5d0 \ube68\ub9ac \uc8fc\ubb38\ud558\uace0 \uc2f6\uc740\ub370, \uc9c0\uae08 \uc7ac\uace0 \uc788\ub098\uc694?",
            "{product}\ub294 \uc7ac\uace0\uac00 \uc788\uc9c0\ub9cc, \uc548\uc804\ud558\uac8c \ubc1b\uc73c\uc2dc\ub824\uba74 \ube68\ub9ac \uc8fc\ubb38\ud558\uc2dc\ub294 \uac78 \uad8c\uc7a5\ub4dc\ub9bd\ub2c8\ub2e4. \ud3ec\ud138\uc5d0\uc11c \uc8fc\ubb38\ud558\uc2dc\uac70\ub098 sales@officeflow.com \uc73c\ub85c \uba54\uc77c \uc8fc\uc154\ub3c4 \ub429\ub2c8\ub2e4."
        ),
        (
            "\uc548\ub155\ud558\uc138\uc694, {product}\uc640 {product2} \ub458 \ub2e4 \uad6c\ub9e4 \uac00\ub2a5\ud55c\uc9c0 \uc54c\ub824\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "{product}\uc640 {product2} \ubaa8\ub450 \ud604\uc7ac \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4! \ucd94\uac00\ub85c \ud544\uc694\ud558\uc2e0 \uc815\ubcf4\uac00 \uc788\uc73c\uba74 \uc54c\ub824\uc8fc\uc138\uc694."
        ),
        (
            "\uc0ac\ubb34\uc2e4 \ube44\ud488\uc744 \ub2e4\uc2dc \ucc44\uc6b0\ub824\uace0 \ud558\ub294\ub370\uc694, {product} \uac00\uc9c0\uace0 \uacc4\uc2e0\uac00\uc694?",
            "\ub124, {product}\ub294 \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4! \ub300\ub7c9 \uc7ac\uc785\uace0\ub77c\uba74 sales@officeflow.com \uc73c\ub85c \ubb38\uc758\ud558\uc2dc\uba74 \ub300\ub7c9 \ud560\uc778 \uac00\uaca9\uc744 \uc548\ub0b4\ud574 \ub4dc\ub9bd\ub2c8\ub2e4."
        ),
        (
            "\uac04\ub2e8\ud788 \uc5ec\ucb64\ubcfc\uac8c\uc694 \u2014 {product} \uc544\uc9c1 \ud310\ub9e4 \uc911\uc778\uac00\uc694, \uc544\ub2c8\uba74 \ud488\uc808\uc778\uac00\uc694?",
            "{product}\ub294 \uc544\uc9c1 \ud310\ub9e4 \uc911\uc785\ub2c8\ub2e4. \uc900\ube44\ub418\uc2dc\uba74 \ud3ec\ud138\uc5d0\uc11c \ubc14\ub85c \uc8fc\ubb38\ud558\uc138\uc694!"
        ),
        (
            "\uc9c0\ub09c\uc8fc\uc5d0 {product} \uc8fc\ubb38\ud588\ub294\ub370 \ucd94\uac00\ub85c \ub354 \ud544\uc694\ud560 \uac83 \uac19\uc544\uc694. \uc544\uc9c1 \uc7ac\uace0 \uc788\ub098\uc694?",
            "{product}\ub294 \uc544\uc9c1 \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4. \uc6f9 \ud3ec\ud138\uc5d0\uc11c \uc5b8\uc81c\ub4e0 \ucd94\uac00 \uc8fc\ubb38 \uac00\ub2a5\ud569\ub2c8\ub2e4."
        ),
        (
            "{product}\uac00 \uc7ac\uace0\uc5d0 \uc788\ub294\uc9c0 \ud55c \ubc88 \ud655\uc778\ud574\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\ubc29\uae08 \ud655\uc778\ud588\ub294\ub370 {product}\ub294 \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4. \ub354 \ud544\uc694\ud55c \uac8c \uc788\uc73c\uc2dc\uba74 \ub9d0\uc500\ud574\uc8fc\uc138\uc694!"
        ),
        (
            "\uc548\ub155\ud558\uc138\uc694, \uc800\ub294 {name} \uc774\uace0 {company} \uc0ac\ubb34\uc2e4\uc5d0 {product}\uac00 \ud544\uc694\ud569\ub2c8\ub2e4. \uad6c\ub9e4 \uac00\ub2a5\ud560\uae4c\uc694?",
            "{name}\ub2d8 \uc548\ub155\ud558\uc138\uc694! {product}\ub294 \uc7ac\uace0\uac00 \uc788\uace0 \ubc14\ub85c \ubc1c\uc1a1 \uac00\ub2a5\ud569\ub2c8\ub2e4. \ud3ec\ud138\uc5d0\uc11c \uc8fc\ubb38\ud558\uc2dc\uac70\ub098 sales@officeflow.com \uc73c\ub85c \uc5f0\ub77d \uc8fc\uc138\uc694."
        ),
        (
            "{product} \uac19\uc740 \uac70 \ucde8\uae09\ud558\uc2dc\ub098\uc694? \uc0ac\ubb34\uc2e4\uc5d0\uc11c \uc880 \ud544\uc694\ud574\uc11c\uc694.",
            "\ub124, {product}\ub294 \ud604\uc7ac \ucde8\uae09 \uc911\uc774\uace0 \uc7ac\uace0\ub3c4 \uc788\uc2b5\ub2c8\ub2e4. \ucd94\uac00 \uc815\ubcf4\ub098 \uc720\uc0ac \uc0c1\ud488 \ucd94\ucc9c \ud544\uc694\ud558\uc2dc\uba74 \uc54c\ub824\uc8fc\uc138\uc694!"
        ),
        (
            "{product}\ub97c \uc6f9\uc0ac\uc774\ud2b8\uc5d0\uc11c \ubd24\ub294\ub370 \uc7ac\uace0\uac00 \uc788\ub294\uc9c0 \uc798 \ubaa8\ub974\uaca0\uc5b4\uc694. \ud655\uc778 \ubd80\ud0c1\ub4dc\ub824\uc694.",
            "{product}\ub294 \uc7ac\uace0\uac00 \uc788\ub294 \uac83\uc73c\ub85c \ud655\uc778\ub429\ub2c8\ub2e4! \uc0ac\uc774\ud2b8\uc5d0\uc11c \ubc14\ub85c \uc8fc\ubb38\ud558\uc2e4 \uc218 \uc788\uc744 \uac70\uc608\uc694."
        ),
        (
            "\uc11c\ub958 \uc815\ub9ac\uc6a9 \uc0ac\ubb34\uc6a9\ud488\uc73c\ub85c \uc5b4\ub5a4 \uac83\ub4e4\uc774 \uc788\ub098\uc694?",
            "\uc11c\ub958 \uc815\ub9ac\uc6a9\uc73c\ub85c\ub294 \ub9c8\ub2d0\ub77c \uc11c\ub958 \ud3f4\ub354 (25\uac1c), \uac78\uc774\ud615 \uc11c\ub958 \ud3f4\ub354 (25\uac1c), \ud22c\uba85 \ube44\ub2d0 \uc18d\uc9c0 (50\ub9e4) \ub4f1\uc774 \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4. \uad00\uc2ec \uc788\ub294 \ud56d\ubaa9 \uc54c\ub824\uc8fc\uc138\uc694!"
        ),
        (
            "\ud544\uae30\uad6c \uc7ac\uace0 \uc788\ub294 \uac83 \uc788\ub098\uc694? \ud39c\uc774\ub791 \uc5f0\ud544 \ucc3e\uace0 \uc788\uc5b4\uc694.",
            "\ubb3c\ub860\uc785\ub2c8\ub2e4! \ube14\ub8e8 \ubcfc\ud39c (12\uc790\ub8e8), \ud751\uc0c9 \uc824\ud39c (6\uc790\ub8e8), \uc0e4\ud504 (12\uc790\ub8e8), \ud615\uad11\ud39c (5\uac1c) \ubaa8\ub450 \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4."
        ),
        (
            "\ucc45\uc0c1 \uc561\uc138\uc11c\ub9ac\uac00 \ud544\uc694\ud569\ub2c8\ub2e4. \uc7ac\uace0 \uc788\ub294 \uac83 \ubb50\uac00 \uc788\ub098\uc694?",
            "\ucc45\uc0c1 \uc561\uc138\uc11c\ub9ac\ub294 \ub370\uc2a4\ud06c \uc815\ub9ac\ud568, 2\ub2e8 \uc11c\ub958 \ud2b8\ub808\uc774, 2026 \ud0c1\uc0c1 \ub2ec\ub825, \ud14c\uc774\ud504 \ub514\uc2a4\ud39c\uc11c \ub4f1\uc774 \uc788\uc2b5\ub2c8\ub2e4. \ub9c8\uc74c\uc5d0 \ub4dc\ub294 \uac8c \uc788\uc73c\uc2dc\uba74 \uc54c\ub824\uc8fc\uc138\uc694!"
        ),
        (
            "{product} \uc7ac\uace0 \uc788\ub098\uc694? \uc800\ud76c {company}\uc5d0\uc11c \uc774\uac70 \uc815\ub9d0 \ube68\ub9ac \uc18c\uc9c4\ub3fc\uc694.",
            "{product}\ub294 \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4. \ube60\ub974\uac8c \uc18c\uc9c4\ub418\uc2e0\ub2e4\uba74 sales@officeflow.com \uc73c\ub85c \uc815\uae30 \uc8fc\ubb38 \uc124\uc815\uc5d0 \ub300\ud574 \ubb38\uc758\ud574\ubcf4\uc138\uc694."
        ),
        (
            "\uc548\ub155\ud558\uc138\uc694, {name}\uc785\ub2c8\ub2e4. {product} \uc7ac\uace0 \ud655\uc778 \ubd80\ud0c1\ub4dc\ub824\uc694.",
            "{name}\ub2d8 \uc548\ub155\ud558\uc138\uc694! {product}\ub294 \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4. \uc8fc\ubb38 \ub3c4\uc640\ub4dc\ub9b4\uae4c\uc694?"
        ),
        (
            "{product}\uac00 \uae09\ud558\uac8c \ud544\uc694\ud55c\ub370, \uc7ac\uace0 \uc788\ub2e4\uace0 \ub9d0\uc500\ud574\uc8fc\uc138\uc694.",
            "\uc88b\uc740 \uc18c\uc2dd\uc774\uc5d0\uc694 \u2014 {product}\ub294 \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4! \uc6f9 \ud3ec\ud138\uc5d0\uc11c \ubc14\ub85c \uc8fc\ubb38 \uac00\ub2a5\ud569\ub2c8\ub2e4."
        ),
        (
            "{product}\uac00 \ubc29\uae08 \ub2e4 \ub5a8\uc5b4\uc84c\ub294\ub370\uc694. \uc7ac\uace0 \uc788\ub098\uc694?",
            "{product} \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! \ud3ec\ud138\uc5d0\uc11c \uc8fc\ubb38\ud558\uc2dc\uba74 \uae08\ubc29 \ub2e4\uc2dc \ucc44\uc6b0\uc2e4 \uc218 \uc788\uc5b4\uc694."
        ),
        (
            "{product}\ud558\uace0 {product2} \uc7ac\uace0 \ud655\uc778 \uc880 \ud574\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "{product}\uc640 {product2} \ubaa8\ub450 \ud604\uc7ac \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4. \uc8fc\ubb38\ud558\uc2dc\ub824\uba74 \uc54c\ub824\uc8fc\uc138\uc694!"
        ),
        (
            "{product} \uc5bc\ub9c8\ub098 \uc788\ub098\uc694? \ub300\ub7c9\uc73c\ub85c \ud544\uc694\ud569\ub2c8\ub2e4.",
            "{product}\ub294 \ud604\uc7ac \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4. \ub300\ub7c9 \uc8fc\ubb38\uc740 sales@officeflow.com \uc73c\ub85c \uc601\uc5c5\ud300\uc5d0 \ubb38\uc758\ud558\uc2dc\uba74 \ucd5c\uc801\uc758 \uac00\uaca9\uc744 \uc548\ub0b4\ud574 \ub4dc\ub9bd\ub2c8\ub2e4."
        ),
        (
            "\uc800 {company}\uc5d0\uc11c \uc77c\ud558\uace0 \uc788\uc5b4\uc694. {product}\uac00 \ud544\uc694\ud55c\ub370, \ucde8\uae09\ud558\uc2dc\ub294 \uc0c1\ud488\uc778\uac00\uc694?",
            "\uc548\ub155\ud558\uc138\uc694! {product}\ub294 \ubd84\uba85\ud788 \ucde8\uae09\ud558\uace0 \uc788\uace0 \uc9c0\uae08 \uc7ac\uace0\ub3c4 \uc788\uc2b5\ub2c8\ub2e4. {company}\ud300\uc5d0\uc11c \ud3ec\ud138\uc744 \ud1b5\ud574 \uc8fc\ubb38\ud558\uc2dc\uac70\ub098 sales@officeflow.com \uc73c\ub85c \uba54\uc77c \uc8fc\uc154\ub3c4 \ub429\ub2c8\ub2e4."
        ),
        (
            "{product} \ub0a8\uc740 \uac70 \uc788\ub098\uc694? \uc8fc\ubb38\ud558\ub824\uace0 \uacc4\uc18d \ubbf8\ub8e8\uace0 \uc788\uc5c8\uc5b4\uc694.",
            "\ub124, {product}\ub294 \uc544\uc9c1 \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4! \uc900\ube44\ub418\uc2dc\uba74 \uc5b8\uc81c\ub4e0 \uc8fc\ubb38 \uc9c4\ud589\ud558\uc138\uc694."
        ),
        (
            "{product}\uac00 \ucc3d\uace0\uc5d0 \uc788\ub294\uc9c0 \ud655\uc778\ud574\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\ubc29\uae08 \ud655\uc778\ud588\uc5b4\uc694 \u2014 {product}\ub294 \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4. \uc8fc\ubb38 \uac00\ub2a5\ud569\ub2c8\ub2e4!"
        ),
        (
            "\ud604\uc7ac \uc7ac\uace0\uac00 \uc788\ub294 \uc885\uc774\ub958 \uc0c1\ud488\uc73c\ub85c\ub294 \uc5b4\ub5a4 \uac8c \uc788\ub098\uc694?",
            "\uc885\uc774\ub958 \uc0c1\ud488\uc73c\ub85c\ub294 \ubcf5\uc0ac\uc6a9\uc9c0 500\ub9e4, \ub9ac\uac08 \ud328\ub4dc (6\uad8c), \uc2a4\ud504\ub9c1 \ub178\ud2b8 (3\uad8c), \uc778\ub371\uc2a4 \uce74\ub4dc (100\ub9e4) \ub4f1\uc774 \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4. \uc790\uc138\ud55c \uc815\ubcf4\uac00 \ud544\uc694\ud55c \ud56d\ubaa9 \uc788\uc73c\uc2dc\uba74 \uc54c\ub824\uc8fc\uc138\uc694!"
        ),
        (
            "\ud3ec\uc2a4\ud2b8\uc787\uc774\ub098 \ube44\uc2b7\ud55c \uc885\ub958\uc758 \uc0c1\ud488 \ucc3e\uace0 \uc788\uc5b4\uc694. \uc5b4\ub5a4 \uac8c \uc788\ub098\uc694?",
            "\ud604\uc7ac \ud3ec\uc2a4\ud2b8\uc787 \uba54\ubaa8\uc9c0 (4\uac1c)\uc640 \ud3ec\uc2a4\ud2b8\uc787 \ud50c\ub798\uadf8 (4\uc0c9)\uac00 \uc7ac\uace0\uc5d0 \uc788\uc2b5\ub2c8\ub2e4. \ub458 \ub2e4 \uc815\ub9ac\uc6a9\uc73c\ub85c \uc544\uc8fc \uc88b\uc544\uc694!"
        ),
        (
            "\uc548\ub155\ud558\uc138\uc694, \uc800\ub294 {company}\uc758 {name}\uc785\ub2c8\ub2e4. {product} \uc788\ub294\uc9c0 \ud655\uc778 \uc880 \ubd80\ud0c1\ub4dc\ub824\uc694.",
            "{name}\ub2d8 \uc548\ub155\ud558\uc138\uc694! \ub124, {product}\ub294 \ud604\uc7ac \uad6c\ub9e4 \uac00\ub2a5\ud569\ub2c8\ub2e4. {company}\uc5d0 \ub2e4\ub978 \ud544\uc694\ud558\uc2e0 \uac8c \uc788\uc73c\uba74 \uc54c\ub824\uc8fc\uc138\uc694!"
        ),
        (
            "\uc800\ud76c\uac00 {product}\ub97c \ub9ce\uc774 \uc0ac\uc6a9\ud574\uc694. \uc8fc\ubb38 \uc804\uc5d0 \uc7ac\uace0 \ud655\uc778 \uc880 \ud574\uc8fc\uc2dc\uaca0\uc5b4\uc694?",
            "\ud655\uc778\ud588\uc2b5\ub2c8\ub2e4 \u2014 {product}\ub294 \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4. \uc8fc\ubb38 \uc9c4\ud589\ud574 \uc8fc\uc138\uc694!"
        ),
        (
            "\uce74\ud0c8\ub85c\uadf8\ub97c \ubcf4\uba74\uc11c {product}\uac00 \uc788\ub294\uc9c0 \uad81\uae08\ud574\uc11c \ubb38\uc758\ub4dc\ub9bd\ub2c8\ub2e4.",
            "{product}\ub294 \uc7ac\uace0\uac00 \uc788\uace0 \ubc1c\uc1a1 \uc900\ube44\ub3c4 \ub418\uc5b4 \uc788\uc2b5\ub2c8\ub2e4. \ub2e4\ub978 \uc0c1\ud488\uc5d0 \ub300\ud574 \uad81\uae08\ud55c \uc810 \uc788\uc73c\uc2dc\uba74 \uc54c\ub824\uc8fc\uc138\uc694!"
        ),
        (
            "\ube44\ud488 \ucc3d\uace0\uac00 \ube44\uc5c8\uc5b4\uc694. {product}, {product2}, {product3} \ub2e4 \uc7ac\uace0 \uc788\ub098\uc694?",
            "\uc88b\uc740 \uc18c\uc2dd\uc785\ub2c8\ub2e4 \u2014 {product}, {product2}, {product3} \ubaa8\ub450 \ud604\uc7ac \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4! \ud55c \ubc88\uc5d0 \ub2e4 \ucc44\uc6b0\uc2dc\uae30 \ub531 \uc88b\uc544\uc694."
        ),
        (
            "\ud639\uc2dc {product} \uc7ac\uace0\uac00 \ubd80\uc871\ud55c\uac00\uc694?",
            "{product}\ub294 \uc7ac\uace0\uac00 \uc788\uae34 \ud55c\ub370 \uc7ac\uace0 \uc218\ub7c9\uc774 \ubcc0\ub3d9\ub420 \uc218 \uc788\uc73c\ub2c8, \ud544\uc694\ud558\uc2dc\uba74 \ube68\ub9ac \uc8fc\ubb38\ud558\uc2dc\ub294 \uac78 \ucd94\ucc9c\ub4dc\ub824\uc694!"
        ),
        (
            "{company}\ub85c \ubc30\uc1a1 \uac00\ub2a5\ud55c\uc9c0, {product} \uc7ac\uace0 \uc880 \ud655\uc778\ud574\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "{product} \uc7ac\uace0 \uc788\uace0, {company} \uc8fc\uc18c\ub85c \ubc30\uc1a1 \uac00\ub2a5\ud569\ub2c8\ub2e4. \ud3ec\ud138\uc5d0\uc11c \ubc14\ub85c \uc8fc\ubb38\ud574 \uc8fc\uc138\uc694!"
        ),
        (
            "{product}\ub97c \uc8fc\ubb38\ud558\ub824\uace0 \ud558\ub294\ub370\uc694. \uc7ac\uace0 \uc0c1\ud669\uc740 \uc5b4\ub5a4\uac00\uc694?",
            "{product}\ub294 \uad6c\ub9e4 \uac00\ub2a5\ud569\ub2c8\ub2e4. \ud3b8\ud558\uc2e4 \ub54c \uc6f9 \ud3ec\ud138\uc5d0\uc11c \uc8fc\ubb38 \uc9c4\ud589\ud574 \uc8fc\uc138\uc694."
        ),
        (
            "\uc774\ubc88 \uc8fc \uc548\uc5d0 {product}\ub97c \uc0ac\ubb34\uc2e4\ub85c \ubc1b\uc744 \uc218 \uc788\uc744\uae4c\uc694?",
            "{product}\ub294 \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4! KST \uc624\ud6c4 2\uc2dc \uc804\uc5d0 \uc8fc\ubb38\ud558\uc2dc\uba74 \ub2f9\uc77c \ucd9c\uace0\ub429\ub2c8\ub2e4. \ud45c\uc900 \ubc30\uc1a1\uc740 3~5\uc601\uc5c5\uc77c \uc18c\uc694\ub429\ub2c8\ub2e4."
        ),
        (
            "{product} \ucde8\uae09\ud558\uc2dc\ub098\uc694? \uc6f9\uc0ac\uc774\ud2b8\uc5d0\uc11c \ubabb \ucc3e\uaca0\uc5b4\uc694.",
            "\ub124, {product} \ucde8\uae09\ud558\uace0 \uc788\uace0 \uc7ac\uace0\ub3c4 \uc788\uc2b5\ub2c8\ub2e4! \uc0ac\uc774\ud2b8\uc5d0\uc11c \ubabb \ucc3e\uc73c\uc168\ub2e4\uba74 \uc0c1\ud488\uba85\uc73c\ub85c \uac80\uc0c9\ud574\ubcf4\uc2dc\uac70\ub098 support@officeflow.com \uc73c\ub85c \ubb38\uc758 \uc8fc\uc2dc\uba74 \ub3c4\uc640\ub4dc\ub9ac\uaca0\uc2b5\ub2c8\ub2e4."
        ),
        (
            "\ube44\uc11c\uac00 {product} \uc7ac\uace0 \ud655\uc778\ud574 \ubcf4\ub77c\uace0 \ud574\uc11c\uc694. \uc7ac\uace0 \uc788\ub098\uc694?",
            "{product} \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! \ube44\uc11c\ubd84\uaed8 \uc5b8\uc81c\ub4e0 \ud3ec\ud138\uc5d0\uc11c \uc8fc\ubb38 \uac00\ub2a5\ud558\ub2e4\uace0 \uc804\ud574\uc8fc\uc138\uc694."
        ),
        (
            "\uc800\ub294 {name}\uc785\ub2c8\ub2e4. \ud734\uac8c\uc2e4 \ube44\ud488\uc744 \ub2e4\uc2dc \ucc44\uc6b0\ub824\uace0 \ud558\ub294\ub370, {product} \ucde8\uae09\ud558\uc2dc\ub098\uc694?",
            "{name}\ub2d8 \uc548\ub155\ud558\uc138\uc694! {product} \ucde8\uae09\ud558\uace0 \uc788\uace0 \ud604\uc7ac \uc7ac\uace0\ub3c4 \uc788\uc2b5\ub2c8\ub2e4. \ub354 \ud544\uc694\ud55c \uac8c \uc788\uc73c\uc2dc\uba74 \uc54c\ub824\uc8fc\uc138\uc694!"
        ),
    ]
    pairs = []
    for t in templates:
        p1, p2 = _rand_products(2)
        p3 = _rand_product()
        name = _rand_name()
        company = _rand_company()
        q = t[0].format(product=p1, product2=p2, product3=p3, name=name, company=company)
        a = t[1].format(product=p1, product2=p2, product3=p3, name=name, company=company)
        pairs.append((q, a))
    return pairs


def _policy_questions():
    """policy \uce74\ud14c\uace0\ub9ac: \ud68c\uc0ac \uc815\ucc45(\ubc18\ud488/\ubc30\uc1a1/\uc601\uc5c5\uc2dc\uac04 \ub4f1)\uc5d0 \ub300\ud55c (\uc9c8\ubb38, \uc751\ub2f5) \ud29c\ud50c \ub9ac\uc2a4\ud2b8 \ubc18\ud658."""
    templates = [
        (
            "\ubc18\ud488 \uc815\ucc45\uc774 \uc5b4\ub5bb\uac8c \ub418\ub098\uc694?",
            "\uc218\ub839 \ud6c4 30\uc77c \uc774\ub0b4\uc5d0 \ubc18\ud488 \uac00\ub2a5\ud569\ub2c8\ub2e4. \ubc18\ud488\ud300(returns@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 3\ubc88)\uc5d0\uc11c RMA \ubc88\ud638\ub97c \ubc1c\uae09\ubc1b\uc73c\uc154\uc57c \ud569\ub2c8\ub2e4. \ubd88\ub7c9\ud488\uc740 \uc5b8\uc81c\ub4e0 \ubc18\ud488 \uac00\ub2a5\ud569\ub2c8\ub2e4."
        ),
        (
            "\ubd88\ub7c9\ud488\uc740 \uc5b4\ub5bb\uac8c \ubc18\ud488\ud558\ub098\uc694?",
            "\ubd88\ub7c9\ud488 \ubc18\ud488\uc740 returns@officeflow.com \uc73c\ub85c \uc8fc\ubb38\ubc88\ud638, \ubd88\ub7c9 \ub0b4\uc6a9, \uac00\ub2a5\ud558\uba74 \uc0ac\uc9c4\uc744 \ucca8\ubd80\ud574\uc11c \uba54\uc77c \uc8fc\uc138\uc694. RMA \ubc88\ud638\uc640 \ubb34\ub8cc \ubc18\ud488 \ub77c\ubca8\uc744 \ubc1c\uae09\ud574 \ub4dc\ub9bd\ub2c8\ub2e4."
        ),
            (
            "\ubc30\uc1a1 \uc635\uc158\uacfc \ube44\uc6a9\uc774 \uc5b4\ub5bb\uac8c \ub418\ub098\uc694?",
            "\uc77c\ubc18 \ubc30\uc1a1(8,950\uc6d0, 3~5\uc77c), \ube60\ub978 \ubc30\uc1a1(15,950\uc6d0, 2~3\uc77c), \ud2b9\uae09 \ubc30\uc1a1(24,950\uc6d0, 1~2\uc77c), \uc775\uc77c \ubc30\uc1a1(39,950\uc6d0, \uc775\uc77c)\uc774 \uc788\uc2b5\ub2c8\ub2e4. 10\ub9cc\uc6d0 \uc774\uc0c1 \uc8fc\ubb38 \uc2dc \uc77c\ubc18 \ubc30\uc1a1 \ubb34\ub8cc!"
        ),
        (
            "\ubb34\ub8cc \ubc30\uc1a1\ub3c4 \uc788\ub098\uc694?",
            "\ub124! 10\ub9cc\uc6d0 \uc774\uc0c1 \uc8fc\ubb38 \uc2dc \uc77c\ubc18 \ubc30\uc1a1 \ubb34\ub8cc\uc785\ub2c8\ub2e4. 25\ub9cc\uc6d0 \uc774\uc0c1 \uc8fc\ubb38\uc740 \ube60\ub978 \ubc30\uc1a1\ub3c4 \ubb34\ub8cc\uc785\ub2c8\ub2e4."
        ),
        (
            "\uc77c\ubc18 \ubc30\uc1a1\uc740 \uc5bc\ub9c8\ub098 \uac78\ub9ac\ub098\uc694?",
            "\uc77c\ubc18 \ubc30\uc1a1\uc740 3~5\uc601\uc5c5\uc77c \uc18c\uc694\ub429\ub2c8\ub2e4. KST \uc624\ud6c4 2\uc2dc \uc774\uc804 \uc8fc\ubb38\uc740 \ub2f9\uc77c \ucd9c\uace0\ub429\ub2c8\ub2e4."
        ),
        (
            "\ud574\uc678 \ubc30\uc1a1 \uac00\ub2a5\ud55c\uac00\uc694?",
            "\ub124! \uac00\uae4c\uc6b4 \uc544\uc2dc\uc544 \uad6d\uac00\ub294 \ubaa8\ub450 \ubc30\uc1a1 \uac00\ub2a5\ud569\ub2c8\ub2e4. \uad00\uc138 \ubc0f \uc138\uae08\uc774 \ubd80\uacfc\ub420 \uc218 \uc788\uc73c\uba70, \ud1b5\uad00\uc5d0 1~2\uc77c \ucd94\uac00 \uc18c\uc694\ub429\ub2c8\ub2e4."
        ),
        (
            "\uc601\uc5c5\uc2dc\uac04\uc774 \uc5b4\ub5bb\uac8c \ub418\ub098\uc694?",
            "\uba54\uc778 \uc804\ud654 \uc0c1\ub2f4\uc740 \ud3c9\uc77c \uc624\uc804 7\uc2dc ~ \uc624\ud6c4 7\uc2dc (KST), \ud1a0\uc694\uc77c \uc624\uc804 9\uc2dc ~ \uc624\ud6c4 2\uc2dc (KST)\uc5d0 \uac00\ub2a5\ud569\ub2c8\ub2e4. \uc77c\uc694\uc77c\uc740 \ud734\ubb34\uc785\ub2c8\ub2e4."
        ),
        (
            "\uc8fc\ubb38\uc740 \uc5b4\ub5bb\uac8c \ud558\ub098\uc694?",
            "\uc6f9 \ud3ec\ud138(www.officeflow.com)\uc744 \ud1b5\ud574 \uc8fc\ubb38\ud558\uc2dc\uac70\ub098, sales@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 2\ubc88\uc73c\ub85c \uc601\uc5c5\ud300\uc5d0 \uc5f0\ub77d \uc8fc\uc138\uc694."
        ),
        (
            "\ubc18\ud488 \uac00\ub2a5 \uae30\uac04\uc774 \uc5b4\ub5bb\uac8c \ub418\ub098\uc694? \ubc1b\uc740 \uc9c0 2\uc8fc \uc815\ub3c4 \ub410\uc5b4\uc694.",
            "\ubc18\ud488 \uac00\ub2a5 \uae30\uac04 \ub0b4\uc5d0 \uc788\uc2b5\ub2c8\ub2e4! \uc218\ub839 \ud6c4 30\uc77c \uc774\ub0b4\uc5d0 \ubc18\ud488 \uac00\ub2a5\ud569\ub2c8\ub2e4. returns@officeflow.com \uc73c\ub85c \uc8fc\ubb38\ubc88\ud638 \ubcf4\ub0b4\uc2dc\uba74 RMA\ub97c \ubc1c\uae09\ud574 \ub4dc\ub9bd\ub2c8\ub2e4."
        ),
        (
            "{order} \uc8fc\ubb38 \uac74\uc744 \ubc18\ud488\ud558\uace0 \uc2f6\uc740\ub370 \uc5b4\ub5bb\uac8c \uc9c4\ud589\ud558\ub098\uc694?",
            "{order} \uc8fc\ubb38 \ubc18\ud488\uc744 \uc704\ud574\uc11c\ub294 returns@officeflow.com \uc73c\ub85c \uc8fc\ubb38\ubc88\ud638\uc640 \ubc18\ud488 \uc0ac\uc720\ub97c \ubcf4\ub0b4\uc8fc\uc138\uc694. 4\uc601\uc5c5\uc2dc\uac04 \uc774\ub0b4\uc5d0 RMA\uac00 \ubc1c\uae09\ub429\ub2c8\ub2e4."
        ),
        (
            "\ud658\ubd88\uc740 \uc5bc\ub9c8\ub098 \uac78\ub9ac\ub098\uc694?",
            "\ud658\ubd88\uc740 \ubc18\ud488 \ub3c4\ucc29 \ud6c4 5~7\uc601\uc5c5\uc77c \uc774\ub0b4\uc5d0 \ucc98\ub9ac\ub429\ub2c8\ub2e4. \ud658\ubd88 \uae08\uc561\uc740 \uc6d0\ub798 \uacb0\uc81c \uc218\ub2e8\uc73c\ub85c \ub2e4\uc2dc \uc785\uae08\ub429\ub2c8\ub2e4."
        ),
        (
            "\ud574\uc678 \ubc30\uc1a1\ub3c4 \ud558\uc2dc\ub098\uc694?",
            "\uae30\ubcf8\uc801\uc73c\ub85c \uad6d\ub0b4 \ubc30\uc1a1\uc744 \uc81c\uacf5\ud558\uace0, \uac00\uae4c\uc6b4 \uc544\uc2dc\uc544 \uad6d\uac00\ub3c4 \ubc30\uc1a1 \uac00\ub2a5\ud569\ub2c8\ub2e4. \uadf8 \uc678 \uc9c0\uc5ed\uc740 sales@officeflow.com \uc73c\ub85c \uacac\uc801 \ubb38\uc758 \uc8fc\uc138\uc694 \u2014 \ub300\ub7c9 \uc8fc\ubb38 \uae30\uc900\uc73c\ub85c \uac00\ub2a5\ud569\ub2c8\ub2e4."
        ),
        (
            "\ub2f9\uc77c \ubc30\uc1a1 \ub9c8\uac10 \uc2dc\uac04\uc774 \uc5b8\uc81c\uc778\uac00\uc694?",
            "KST \uc624\ud6c4 2\uc2dc \uc774\uc804\uc5d0 \uc8fc\ubb38\ud558\uc2dc\uba74 \ub2f9\uc77c \ucd9c\uace0\ub429\ub2c8\ub2e4. \uadf8 \uc774\ud6c4 \uc8fc\ubb38\uc740 \ub2e4\uc74c \uc601\uc5c5\uc77c\uc5d0 \ucd9c\uace0\ub429\ub2c8\ub2e4."
        ),
        (
            "\ubc30\uc1a1 \ub300\uc2e0 \uc9c1\uc811 \ubc29\ubb38 \uc218\ub839 \uac00\ub2a5\ud55c\uac00\uc694?",
            "\ub124! \uc11c\uc6b8 \ubcf8\uc0ac, \ubd80\uc0b0 \ubb3c\ub958\uc13c\ud130, \uad11\uc8fc \ubb3c\ub958\uc13c\ud130, \uc778\ucc9c \ubb3c\ub958\uc13c\ud130\uc5d0\uc11c \ubc29\ubb38 \uc218\ub839 \uac00\ub2a5\ud569\ub2c8\ub2e4. \uacb0\uc81c \uc2dc \ubc29\ubb38 \uc218\ub839\uc744 \uc120\ud0dd\ud558\uc2dc\uba74 \uc900\ube44 \uc644\ub8cc \uc2dc \uc548\ub0b4\ub4dc\ub9bd\ub2c8\ub2e4 \u2014 \ubcf4\ud1b5 2~4\uc2dc\uac04 \uc18c\uc694\ub429\ub2c8\ub2e4."
        ),
        (
            "\ubb3c\ub958\uc13c\ud130\ub294 \uc5b4\ub514\uc5d0 \uc788\ub098\uc694?",
            "\ucd1d 4\uacf3\uc758 \ubb3c\ub958\uc13c\ud130\uac00 \uc788\uc2b5\ub2c8\ub2e4: \uc11c\uc6b8 \ubcf8\uc0ac, \ubd80\uc0b0 \ubb3c\ub958\uc13c\ud130, \uad11\uc8fc \ubb3c\ub958\uc13c\ud130, \uc778\ucc9c \ubb3c\ub958\uc13c\ud130."
        ),
        (
            "\ube60\ub978 \ubc30\uc1a1\uc774\ub098 \uc775\uc77c \ubc30\uc1a1\ub3c4 \uc788\ub098\uc694?",
            "\ub124! \ube60\ub978 \ubc30\uc1a1\uc740 15,950\uc6d0 (2~3\uc77c), \ud2b9\uae09 \ubc30\uc1a1\uc740 24,950\uc6d0 (1~2\uc77c), \uc775\uc77c \ubc30\uc1a1\uc740 39,950\uc6d0 (\ub2e4\uc74c \uc601\uc5c5\uc77c)\uc785\ub2c8\ub2e4. \ud2b9\uae09\uacfc \uc775\uc77c \ubc30\uc1a1\uc740 \uac01\uac01 KST \uc624\ud6c4 2\uc2dc\uc640 \uc815\uc624\uae4c\uc9c0 \uc8fc\ubb38\ud558\uc154\uc57c \ud569\ub2c8\ub2e4."
        ),
        (
            "\ud0dd\ubc30\uac00 \ud30c\uc190\ub418\uc5b4 \ub3c4\ucc29\ud558\uba74 \uc5b4\ub5bb\uac8c \ud558\ub098\uc694?",
            "\ud30c\uc190 \ubd80\uc704\ub97c \uc0ac\uc9c4 \ucd2c\uc601\ud558\uc2dc\uace0 48\uc2dc\uac04 \uc774\ub0b4\uc5d0 fulfillment@officeflow.com \uc73c\ub85c \uc5f0\ub77d \uc8fc\uc138\uc694. \uc989\uc2dc \uad50\ud658 \ub610\ub294 \ud658\ubd88 \ucc98\ub9ac\ud574 \ub4dc\ub9bd\ub2c8\ub2e4 \u2014 \ud30c\uc190 \uc0c1\ud488\uc744 \uba3c\uc800 \ubc18\uc1a1\ud558\uc2e4 \ud544\uc694\ub294 \uc5c6\uc2b5\ub2c8\ub2e4."
        ),
        (
            "\ubc18\ud488 \ub300\uc2e0 \uad50\ud658\ub3c4 \uac00\ub2a5\ud55c\uac00\uc694?",
            "\ubb3c\ub860\uc785\ub2c8\ub2e4! \uad50\ud658\ub3c4 \ubc18\ud488\uacfc \ub3d9\uc77c\ud55c \uc808\ucc28\ub85c \uc9c4\ud589\ub429\ub2c8\ub2e4 \u2014 returns@officeflow.com \uc5d0\uc11c RMA\ub97c \ubc1c\uae09\ubc1b\uc73c\uc2dc\uba74 \ub429\ub2c8\ub2e4. \ub2e4\ub978 \uc0c1\ud488\uc73c\ub85c \uad50\ud658 \uc2dc \ucc28\uc561\uc774 \ubc1c\uc0dd\ud560 \uc218 \uc788\uc2b5\ub2c8\ub2e4."
        ),
        (
            "\uc8fc\ubb38 \ubc30\uc1a1 \uc870\ud68c\ub294 \uc5b4\ub5bb\uac8c \ud558\ub098\uc694?",
            "\uc8fc\ubb38\uc774 \ucd9c\uace0\ub418\uba74 \uc1a1\uc7a5\ubc88\ud638\uac00 \uc774\uba54\uc77c\ub85c \ubc1c\uc1a1\ub429\ub2c8\ub2e4. www.officeflow.com/track \uc5d0\uc11c \uc5b8\uc81c\ub4e0 \uc870\ud68c \uac00\ub2a5\ud558\uace0, \ub3c4\uc6c0\uc774 \ud544\uc694\ud558\uc2dc\uba74 fulfillment@officeflow.com \uc73c\ub85c \uc5f0\ub77d \uc8fc\uc138\uc694."
        ),
        (
            "\uc5b4\ub5a4 \uacb0\uc81c \uc218\ub2e8\uc744 \uc0ac\uc6a9\ud560 \uc218 \uc788\ub098\uc694?",
            "\uacb0\uc81c \ubc0f \uccad\uad6c \uad00\ub828 \ubb38\uc758\ub294 \uc815\uc0b0\ud300(accounts@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 4\ubc88)\uc73c\ub85c \uc548\ub0b4\ud574 \ub4dc\ub9bd\ub2c8\ub2e4. \ubaa8\ub4e0 \uacb0\uc81c \uc218\ub2e8 \uad00\ub828 \uc0ac\ud56d\uc744 \ucc98\ub9ac\ud574 \ub4dc\ub9bd\ub2c8\ub2e4."
        ),
        (
            "\uccad\uad6c\uc9c0 \uc8fc\uc18c\ub97c \ubcc0\uacbd\ud558\uace0 \uc2f6\uc5b4\uc694. \uc5b4\ub5bb\uac8c \ud558\ub098\uc694?",
            "\uccad\uad6c\uc9c0 \uc8fc\uc18c \ubcc0\uacbd \uac19\uc740 \uacc4\uc815 \uc815\ubcf4 \ubcc0\uacbd\uc740 \uc815\uc0b0\ud300\uc5d0\uc11c \ucc98\ub9ac\ud569\ub2c8\ub2e4. accounts@officeflow.com \uc73c\ub85c \uba54\uc77c \ubcf4\ub0b4\uc2dc\uac70\ub098 1588-0000 \ub0b4\uc120 4\ubc88\uc73c\ub85c \uc804\ud654 \uc8fc\uc138\uc694."
        ),
        (
            "\uacf5\ud734\uc77c\uc5d0\ub294 \ud734\ubb34\uc778\uac00\uc694?",
            "\uc124\ub0a0, \ucd94\uc11d, \uc5b4\ub9b0\uc774\ub0a0, \uad11\ubcf5\uc808, \uac1c\ucc9c\uc808, \ud55c\uae00\ub0a0, \uc131\ud0c4\uc808\uc5d0 \ud734\ubb34\uc785\ub2c8\ub2e4. \uba85\uc808 \uc804\ub0a0\uacfc \uc131\ud0c4\uc808 \uc774\ube0c\uc5d0\ub294 \ub2e8\ucd95 \uc601\uc5c5(KST \uc624\ud6c4 2\uc2dc\uae4c\uc9c0)\uc785\ub2c8\ub2e4."
        ),
        (
            "\uc800\ub294 {company}\uc758 {name}\uc785\ub2c8\ub2e4. \uae30\uc5c5 \uacc4\uc815\uc758 \ubc30\uc1a1 \uc815\ucc45\uc740 \uc5b4\ub5bb\uac8c \ub418\ub098\uc694?",
            "{name}\ub2d8 \uc548\ub155\ud558\uc138\uc694! {company}\ucc98\ub7fc \uae30\uc5c5 \uacc4\uc815 \uace0\uac1d\uc740 \ub9de\ucda4 \ubc30\uc1a1 \uc694\uae08\uacfc \uc8fc\uc694 \uc9c0\uc5ed \uc804\uc6a9 \ubc30\uc1a1 \ub77c\uc778\uc744 \uc774\uc6a9\ud558\uc2e4 \uc218 \uc788\uc2b5\ub2c8\ub2e4. \uc790\uc138\ud55c \uc0ac\ud56d\uc740 \ub2f4\ub2f9 \ub9e4\ub2c8\uc800\uc5d0\uac8c \ubb38\uc758\ud574 \uc8fc\uc138\uc694!"
        ),
        (
            "\uae30\uc5c5 \uacc4\uc815\uc740 \uc5b4\ub5bb\uac8c \uac1c\uc124\ud558\ub098\uc694?",
            "\uae30\uc5c5 \uacc4\uc815 \uac1c\uc124\uc740 \uc815\uc0b0\ud300(accounts@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 4\ubc88)\uc73c\ub85c \ubb38\uc758 \uc8fc\uc138\uc694. \uac1c\uc124 \uc808\ucc28\ub97c \ub3c4\uc640\ub4dc\ub9bd\ub2c8\ub2e4."
        ),
        (
            "\uac1c\ubd09\ud55c \uc0c1\ud488\ub3c4 \ubc18\ud488 \uac00\ub2a5\ud55c\uac00\uc694?",
            "\uc6d0\ub798 \ud3ec\uc7a5\uc774 \uc788\ub294 \ub300\ubd80\ubd84\uc758 \uac1c\ubd09 \uc0c1\ud488\uc740 30\uc77c \uc774\ub0b4 \ubc18\ud488 \uac00\ub2a5\ud569\ub2c8\ub2e4. \ub2e4\ub9cc \uc704\uc0dd\uc0c1\uc758 \uc774\uc720\ub85c \uac1c\ubd09\ud55c \uc885\uc774 \ubb36\uc74c\uc740 \ubc18\ud488 \ubd88\uac00\uc774\uace0, \ub9de\ucda4 \uc778\uc1c4 \uc0c1\ud488\uc774\ub098 \ud2b9\uac00 \uc0c1\ud488\ub3c4 \ubc18\ud488 \ub300\uc0c1\uc774 \uc544\ub2d9\ub2c8\ub2e4."
        ),
        (
            "\uc8fc\ubb38\ud55c \uac83\uacfc \ub2e4\ub978 \uc0c1\ud488\uc744 \ubc1b\uc558\uc5b4\uc694. \uc5b4\ub5bb\uac8c \ud574\uc57c \ud558\ub098\uc694?",
            "\uc8c4\uc1a1\ud569\ub2c8\ub2e4! fulfillment@officeflow.com \uc73c\ub85c \uc8fc\ubb38\ubc88\ud638\uc640 \ud568\uaed8 \ubc1b\uc73c\uc2e0 \uc0c1\ud488 \ubc0f \uc8fc\ubb38\ud558\uc2e0 \uc0c1\ud488\uc758 \ub0b4\uc6a9\uc744 \ubcf4\ub0b4\uc8fc\uc138\uc694. \ub2f9\uc77c \ub610\ub294 \ub2e4\uc74c \uc601\uc5c5\uc77c\uc5d0 \uc815\ud655\ud55c \uc0c1\ud488\uc744 \ubc1c\uc1a1\ud574 \ub4dc\ub9ac\uace0 \ubc18\ud488 \ub77c\ubca8\ub3c4 \ud568\uaed8 \uc81c\uacf5\ud574 \ub4dc\ub9bd\ub2c8\ub2e4."
        ),
        (
            "\ub300\ub7c9 \uc8fc\ubb38 \uc2dc \ud560\uc778\uc774\ub098 \uac00\uaca9 \ud61c\ud0dd\uc774 \uc788\ub098\uc694?",
            "\ub300\ub7c9 \uc8fc\ubb38 \uac00\uaca9\uc740 sales@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 2\ubc88\uc73c\ub85c \uc601\uc5c5\ud300\uc5d0 \ubb38\uc758\ud574 \uc8fc\uc138\uc694. \ub300\ub7c9 \uc8fc\ubb38 \uacac\uc801\uc744 \uc548\ub0b4\ud574 \ub4dc\ub9bd\ub2c8\ub2e4."
        ),
        (
            "\uc8fc\ubb38 \ud6c4 \ubc30\uc1a1\uc9c0 \uc8fc\uc18c \ubcc0\uacbd \uac00\ub2a5\ud55c\uac00\uc694?",
            "\ucd9c\uace0 \uc804\uc774\ub77c\uba74 \uc989\uc2dc \uc601\uc5c5\ud300\uc5d0 \uc5f0\ub77d \uc8fc\uc138\uc694 \u2014 \uc8fc\ubb38 \ud6c4 2\uc2dc\uac04 \uc774\ub0b4\uc5d0\ub294 \ubb34\ub8cc \ubcc0\uacbd\uc774 \uac00\ub2a5\ud569\ub2c8\ub2e4. \ucd9c\uace0 \ud6c4\uc5d0\ub294 fulfillment \ud300\uc5d0\uc11c \uc7ac\ubc30\uc1a1 \ucc98\ub9ac\ud574 \ub4dc\ub9ac\ub294\ub370 10,000~15,000\uc6d0\uc758 \uc218\uc218\ub8cc\uac00 \ubd80\uacfc\ub429\ub2c8\ub2e4."
        ),
        (
            "\ud1a0\uc694\uc77c \ubc30\uc1a1\ub3c4 \uac00\ub2a5\ud55c\uac00\uc694?",
            "\uc8fc\ub9d0\uc5d0\ub294 \ubb3c\ub958\uc13c\ud130\uac00 \uc6b4\uc601\ub418\uc9c0 \uc54a\uc544 \ud1a0\uc694\uc77c \ubc30\uc1a1\uc740 \uc81c\uacf5\ub418\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4. \ud2b9\uae09 \ubc30\uc1a1\uacfc \uc775\uc77c \ubc30\uc1a1\uc740 \uc6d4\uc694\uc77c~\ubaa9\uc694\uc77c\uc5d0\ub9cc \uc774\uc6a9 \uac00\ub2a5\ud569\ub2c8\ub2e4."
        ),
        (
            "\ubc18\ud488\ud300\uc5d0\ub294 \uc5b4\ub5bb\uac8c \uc5f0\ub77d\ud558\ub098\uc694?",
            "\ubc18\ud488\ud300\uc740 returns@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 3\ubc88\uc73c\ub85c \uc5f0\ub77d \uac00\ub2a5\ud569\ub2c8\ub2e4. \ud3c9\uc77c \uc624\uc804 8\uc2dc ~ \uc624\ud6c4 6\uc2dc (KST)\uc5d0 \uc751\ub300\ud558\uba70 4\uc601\uc5c5\uc2dc\uac04 \uc774\ub0b4\uc5d0 \ud68c\uc2e0\ub4dc\ub9bd\ub2c8\ub2e4."
        ),
        (
            "\uccad\uad6c\uc11c \uad00\ub828 \ubb38\uc758\ub294 \uc5b4\ub514\uc5d0 \ud574\uc57c \ud558\ub098\uc694?",
            "\uccad\uad6c\uc11c \ubc0f \uacb0\uc81c \uad00\ub828 \ubb38\uc758\ub294 \uc815\uc0b0\ud300(accounts@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 4\ubc88)\uc73c\ub85c \uc5f0\ub77d \uc8fc\uc138\uc694. \ud3c9\uc77c \uc624\uc804 9\uc2dc ~ \uc624\ud6c4 5\uc2dc (KST)\uc5d0 \uc751\ub300\ud569\ub2c8\ub2e4."
        ),
        (
            "\ubc18\ud488 \ubc30\uc1a1\ube44\ub294 \ubb34\ub8cc\uc778\uac00\uc694?",
            "\ubd88\ub7c9\ud488 \ubc18\ud488\uc774\ub098 \uc800\ud76c \uce21 \ubc30\uc1a1 \uc624\ub958\uc778 \uacbd\uc6b0 \ubc18\ud488 \ubc30\uc1a1\ube44\ub294 \ubb34\ub8cc\uc785\ub2c8\ub2e4. \uadf8 \uc678\uc5d0\ub294 \ubc18\ud488 \ubc30\uc1a1\ube44\ub97c \uace0\uac1d\ub2d8\uc774 \ubd80\ub2f4\ud558\uc154\uc57c \ud569\ub2c8\ub2e4."
        ),
        (
            "\ub3c4\uc6c0\uc744 \ubc1b\uc73c\ub824\uba74 \uc5b4\ub514\ub85c \uc804\ud654\ud558\uba74 \ub418\ub098\uc694?",
            "\uba54\uc778 \uc804\ud654\ubc88\ud638\ub294 1588-0000 \uc785\ub2c8\ub2e4. \ubd80\uc11c\ubcc4 \ub0b4\uc120\ubc88\ud638: \uace0\uac1d \uc11c\ube44\uc2a4 1\ubc88, \uc601\uc5c5 2\ubc88, \ubc18\ud488 3\ubc88, \uc815\uc0b0 4\ubc88, \ubb3c\ub958 5\ubc88, \uae30\uc220 \uc9c0\uc6d0 6\ubc88."
        ),
        (
            "\ubc18\ud488\uc6a9 RMA \ubc88\ud638\ub294 \uc5b4\ub5bb\uac8c \ubc1b\ub098\uc694?",
            "\ubc18\ud488\ud300(returns@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 3\ubc88)\uc73c\ub85c \uc8fc\ubb38\ubc88\ud638\uc640 \ubc18\ud488 \uc0ac\uc720\ub97c \ubcf4\ub0b4\uc8fc\uc2dc\uba74 4\uc601\uc5c5\uc2dc\uac04 \uc774\ub0b4\uc5d0 RMA\uac00 \ubc1c\uae09\ub429\ub2c8\ub2e4."
        ),
        (
            "\uc0ac\uc11c\ud568\uc73c\ub85c\ub3c4 \ubc30\uc1a1 \uac00\ub2a5\ud55c\uac00\uc694?",
            "\ub124, \ub300\ubd80\ubd84\uc758 \uc0c1\ud488\uc740 \uc77c\ubc18 \ubc30\uc1a1\uc73c\ub85c \uc0ac\uc11c\ud568 \ubc30\uc1a1\uc774 \uac00\ub2a5\ud569\ub2c8\ub2e4. \ubd80\ud53c\uac00 \ud06c\uac70\ub098 \uc704\ud5d8\ubb3c\ud488\uc740 \uc77c\ubd80 \uc81c\ud55c\uc774 \uc788\uc2b5\ub2c8\ub2e4."
        ),
        (
            "\ud68c\uc0ac \uc5f0\ub77d\ucc98\ub294 \uc5b4\ub514\uc11c \ud655\uc778\ud560 \uc218 \uc788\ub098\uc694?",
            "\uba54\uc778 \uc804\ud654\ubc88\ud638\ub294 1588-0000 \uc785\ub2c8\ub2e4. \ubcf8\uc0ac\ub294 \uc11c\uc6b8\ud2b9\ubcc4\uc2dc \uac15\ub0a8\uad6c \ud14c\ud5e4\ub780\ub85c 123 \uc785\ub2c8\ub2e4. \uc774\uba54\uc77c\uc740 info@officeflow.com, \uc6f9\uc0ac\uc774\ud2b8\ub294 www.officeflow.com \uc785\ub2c8\ub2e4."
        ),
        (
            "\ud0dd\ubc30\uac00 \ubc30\uc1a1 \uc644\ub8cc\ub418\uc5c8\ub2e4\uace0 \ub5b4\ub294\ub370 \ubabb \ubc1b\uc558\uc5b4\uc694. \uc5b4\ub5bb\uac8c \ud558\ub098\uc694?",
            "fulfillment@officeflow.com \uc73c\ub85c \uc5f0\ub77d \uc8fc\uc138\uc694. \ud0dd\ubc30\uc0ac\uc5d0 \ubd84\uc2e4 \uc2e0\uace0\ub97c \uc811\uc218\ud558\uace0 \ub300\uccb4\ud488\uc744 \ubc1c\uc1a1\ud574 \ub4dc\ub9bd\ub2c8\ub2e4. \ubd84\uc2e4 \uc2e0\uace0 \ucc98\ub9ac\ub294 \ubcf4\ud1b5 3~5\uc601\uc5c5\uc77c \uc18c\uc694\ub429\ub2c8\ub2e4."
        ),
        (
            "\uad50\ud658 \uc815\ucc45\uc5d0 \ub300\ud574 \uc54c\ub824\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\uad50\ud658\uc740 \ud658\uc601\uc785\ub2c8\ub2e4! \ubc18\ud488\uacfc \ub3d9\uc77c\ud55c \uc808\ucc28\ub85c \uc9c4\ud589\ub429\ub2c8\ub2e4 \u2014 returns@officeflow.com \uc73c\ub85c RMA\ub97c \ubc1c\uae09\ubc1b\uc73c\uc138\uc694. \ucc28\uc561\uc774 \ubc1c\uc0dd\ud560 \uc218 \uc788\uace0, \ubc18\ud488 \ub3c4\ucc29 \ubc0f \uac80\uc218 \ud6c4\uc5d0 \uad50\ud658 \uc0c1\ud488\uc774 \ubc1c\uc1a1\ub429\ub2c8\ub2e4."
        ),
        (
            "\uc548\ub155\ud558\uc138\uc694, \uc800\ub294 {name}\uc785\ub2c8\ub2e4. \ubd88\ub7c9\ud488 \ubc18\ud488 \uc815\ucc45\uc5d0 \ub300\ud574 \uc54c\uace0 \uc2f6\uc5b4\uc694.",
            "{name}\ub2d8 \uc548\ub155\ud558\uc138\uc694! \ubd88\ub7c9\ud488\uc740 30\uc77c\uc774 \uc9c0\ub098\ub3c4 \uc5b8\uc81c\ub4e0 \ubc18\ud488 \uac00\ub2a5\ud569\ub2c8\ub2e4. returns@officeflow.com \uc73c\ub85c \ubd88\ub7c9 \ubd80\uc704 \uc0ac\uc9c4\uacfc \ud568\uaed8 \uba54\uc77c \uc8fc\uc138\uc694. \ubb34\ub8cc \ubc18\ud488 \ub77c\ubca8\uc744 \ubcf4\ub0b4\ub4dc\ub9ac\uba70 \ubc18\ud488 \ubc30\uc1a1\ube44\ub3c4 \uc800\ud76c\uac00 \ubd80\ub2f4\ud569\ub2c8\ub2e4."
        ),
        (
            "\ubc30\uc1a1 \uc81c\ud55c \uc0ac\ud56d\uc774 \uc788\ub098\uc694?",
            "\uc704\ud5d8\ubb3c(\uc77c\ubd80 \ub9c8\uce74, \uc811\ucc29\uc81c \ub4f1)\uc740 \uc721\ub85c \ubc30\uc1a1\ub9cc \uac00\ub2a5\ud558\uace0, \uac00\uad6c\ucc98\ub7fc \ubd80\ud53c\uac00 \ud070 \uc0c1\ud488\uc740 \ud654\ubb3c \ubc30\uc1a1\uc73c\ub85c \ucc98\ub9ac\ub429\ub2c8\ub2e4. 22kg \uc774\uc0c1 \uc0c1\ud488\uc740 \ud654\ubb3c \ube44\uc6a9\uc774 \ucd94\uac00\ub420 \uc218 \uc788\uace0, \ud574\uc678 \ubc30\uc1a1 \uc2dc \uc5d0\uc5b4\ub85c\uc878\uc774\ub098 \uc778\ud654\uc131 \ubb3c\uc9c8\uc740 \ubd88\uac00\ud569\ub2c8\ub2e4."
        ),
    ]
    pairs = []
    for t in templates:
        order = _rand_order()
        name = _rand_name()
        company = _rand_company()
        q = t[0].format(order=order, name=name, company=company)
        a = t[1].format(order=order, name=name, company=company)
        pairs.append((q, a))
    return pairs


def _out_of_scope_questions():
    """out_of_scope \uce74\ud14c\uace0\ub9ac: Emma\uac00 \ub3c4\uc640\uc904 \uc218 \uc5c6\ub294 \ubc94\uc704 \uc678 \uc9c8\ubb38 (\uc9c8\ubb38, \uc751\ub2f5) \ud29c\ud50c \ub9ac\uc2a4\ud2b8 \ubc18\ud658."""
    templates = [
        (
            "\uc790\uae30\uc18c\uac1c\uc11c \uc791\uc131 \uc880 \ub3c4\uc640\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\ubb38\uc758 \uc8fc\uc154\uc11c \uac10\uc0ac\ud569\ub2c8\ub2e4\ub9cc, \uc800\ub294 OfficeFlow \uc0c1\ud488 \ubc0f \uc11c\ube44\uc2a4 \uad00\ub828 \ubb38\uc758\ub9cc \ub3c4\uc640\ub4dc\ub9b4 \uc218 \uc788\uc2b5\ub2c8\ub2e4. \uc790\uae30\uc18c\uac1c\uc11c\ub294 \ucee4\ub9ac\uc5b4 \uad00\ub828 \uc0ac\uc774\ud2b8\ub97c \uc774\uc6a9\ud574\ubcf4\uc138\uc694. \uc0ac\ubb34\uc6a9\ud488 \uad00\ub828\ud574\uc11c \ub3c4\uc640\ub4dc\ub9b4 \uc77c\uc774 \uc788\uc744\uae4c\uc694?"
        ),
        (
            "\uc624\ub298 \uc11c\uc6b8 \ub0a0\uc528 \uc5b4\ub54c\uc694?",
            "\ub3c4\uc640\ub4dc\ub9ac\uace0 \uc2f6\uc9c0\ub9cc \uc800\ub294 OfficeFlow \uad00\ub828 \ubb38\uc758\ub85c \ud55c\uc815\ub418\uc5b4 \uc788\uc5b4\uc694! \ub0a0\uc528\ub294 \uc990\uaca8 \uc4f0\uc2dc\ub294 \uc571\uc5d0\uc11c \ud655\uc778\ud574 \ubcf4\uc138\uc694. \uc800\ud76c \uc0c1\ud488\uc774\ub098 \uc11c\ube44\uc2a4 \uad00\ub828\ud574\uc11c \ub3c4\uc640\ub4dc\ub9b4 \uac8c \uc788\uc744\uae4c\uc694?"
        ),
        (
            "\ubcf8\uc0ac \uadfc\ucc98\uc5d0 \uad1c\ucc2e\uc740 \uc2dd\ub2f9 \ucd94\ucc9c\ud574\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\uc7ac\ubbf8\uc788\ub294 \uc9c8\ubb38\uc778\ub370\uc694, \uc544\uc27d\uac8c\ub3c4 \uc800\ub294 OfficeFlow \uc0c1\ud488 \ubc0f \uc11c\ube44\uc2a4 \uad00\ub828 \ubb38\uc758\ub9cc \uc751\ub300\ud560 \uc218 \uc788\uc5b4\uc694. \uc2dd\ub2f9 \ucd94\ucc9c\uc740 \ub124\uc774\ubc84 \uc9c0\ub3c4\ub098 \uad6c\uae00 \uc9c0\ub3c4\uc5d0\uc11c \ucc3e\uc544\ubcf4\uc2dc\ub294 \uac8c \uc88b\uaca0\uc5b4\uc694! \uc0ac\ubb34\uc6a9\ud488 \uad00\ub828\ud574\uc11c \ub3c4\uc640\ub4dc\ub9b4 \uc77c\uc774 \uc788\uc744\uae4c\uc694?"
        ),
        (
            "\uc138\ubb34 \uad00\ub828\ud574\uc11c \ub3c4\uc6c0\uc774 \ud544\uc694\ud574\uc694. \ub3c4\uc640\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\uc138\ubb34\ub294 \uc800\ud76c\uac00 \ub3c4\uc640\ub4dc\ub9b4 \uc218 \uc5c6\ub294 \ubd80\ubd84\uc774\uace0, \uc138\ubb34\uc0ac\ub098 \uc138\ubb34 \ub300\ud589 \uc11c\ube44\uc2a4\ub97c \ucd94\ucc9c\ub4dc\ub9bd\ub2c8\ub2e4. \uc0ac\ubb34\uc6a9\ud488\uc774\ub098 \uc800\ud76c \uc11c\ube44\uc2a4 \uad00\ub828\ud574\uc11c \ud544\uc694\ud558\uc2e0 \uac8c \uc788\uc73c\uba74 \uc5b8\uc81c\ub4e0 \ub9d0\uc500\ud574 \uc8fc\uc138\uc694!"
        ),
        (
            "\uc0ac\ubb34\uc6a9 \uac00\uad6c\ub3c4 \ud310\ub9e4\ud558\ub098\uc694?",
            "\uc800\ud76c\ub294 \uc0ac\ubb34\uc6a9\ud488, \uc885\uc774\ub958, \ud544\uae30\uad6c, \ucc45\uc0c1 \uc561\uc138\uc11c\ub9ac \uc704\uc8fc\ub85c \ucde8\uae09\ud558\uace0 \uc788\uc5b4\uc694. \uac00\uad6c\ub294 \ud604\uc7ac \ucde8\uae09\ud558\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4. \ub2e4\ub978 \uc0c1\ud488 \uad00\ub828 \ubb38\uc758\ub294 \uc5b8\uc81c\ub4e0 \ub3c4\uc640\ub4dc\ub9b4\uac8c\uc694!"
        ),
        (
            "\uc81c \ucef4\ud4e8\ud130 \uc880 \uace0\uccd0\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694? \uacc4\uc18d \uba48\ucdb0\uc694.",
            "\uae30\uae30 \ubb38\uc81c\ub85c \ubd88\ud3b8\uc744 \ub4dc\ub824 \uc8c4\uc1a1\ud55c\ub370, \uc800\ub294 OfficeFlow \uc0c1\ud488\uacfc \uc11c\ube44\uc2a4\ub9cc \uc751\ub300\ud560 \uc218 \uc788\uc5b4\uc694. \ucef4\ud4e8\ud130 \ubb38\uc81c\ub294 \uc0ac\ub0b4 IT \ud300\uc774\ub098 \uac00\uae4c\uc6b4 PC \uc11c\ube44\uc2a4 \uc13c\ud130\uc5d0 \ubb38\uc758\ud558\uc2dc\ub294 \uac8c \uc88b\uaca0\uc5b4\uc694."
        ),
        (
            "OfficeFlow \uc8fc\uac00\uac00 \uc5b4\ub5bb\uac8c \ub418\ub098\uc694?",
            "\uae08\uc735\uc774\ub098 \uc8fc\uac00 \uc815\ubcf4\ub294 \uc548\ub0b4\ud574 \ub4dc\ub9b4 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4. IR \uad00\ub828 \ubb38\uc758\ub294 \ud68c\uc0ac \uc6f9\uc0ac\uc774\ud2b8\ub97c \ud655\uc778\ud558\uc2dc\uac70\ub098 corporate@officeflow.com \uc73c\ub85c \uc5f0\ub77d\ud574 \uc8fc\uc138\uc694. \uc0c1\ud488 \uad00\ub828 \ubb38\uc758\ub294 \uc5b8\uc81c\ub4e0 \ub3c4\uc640\ub4dc\ub9b4\uac8c\uc694!"
        ),
        (
            "\uc774 \ubb38\uc11c \uc880 \ubc88\uc5ed\ud574\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\ubc88\uc5ed\uc740 \uc81c \ub2f4\ub2f9 \ubc94\uc704\uac00 \uc544\ub2c8\uc5d0\uc694. \uc800\ub294 OfficeFlow \uc0c1\ud488\uacfc \uc11c\ube44\uc2a4 \ubb38\uc758\ub97c \uc751\ub300\ud569\ub2c8\ub2e4. \ubc88\uc5ed\uc740 \uad6c\uae00 \ubc88\uc5ed\uc774\ub098 \uc804\ubb38 \ubc88\uc5ed \uc11c\ube44\uc2a4\ub97c \uc774\uc6a9\ud574 \ubcf4\uc138\uc694!"
        ),
        (
            "OfficeFlow \uc5d0 \uc785\uc0ac\ud558\uace0 \uc2f6\uc740\ub370 \uc5b4\ub5bb\uac8c \uc9c0\uc6d0\ud558\ub098\uc694?",
            "\uc800\ud76c\uc640 \ud568\uaed8 \uc77c\ud558\uace0 \uc2f6\uc5b4 \ud574\uc8fc\uc154\uc11c \uac10\uc0ac\ud569\ub2c8\ub2e4! \ucc44\uc6a9 \uc9c0\uc6d0\uc740 \uc81c\uac00 \ub3c4\uc640\ub4dc\ub9b4 \uc218 \uc788\ub294 \uc601\uc5ed\uc774 \uc544\ub2c8\uc5d0\uc694. \ud68c\uc0ac \ucc44\uc6a9 \ud398\uc774\uc9c0\ub97c \ud655\uc778\ud558\uc2dc\uac70\ub098 corporate@officeflow.com \uc73c\ub85c \uba54\uc77c \uc8fc\uc138\uc694. \uc0ac\ubb34\uc6a9\ud488 \uad00\ub828\ud574\uc11c \ub3c4\uc640\ub4dc\ub9b4 \uc77c\uc774 \uc788\uc744\uae4c\uc694?"
        ),
        (
            "\ud68c\uc758\uc2e4 \uc608\uc57d \uc880 \ud574\uc8fc\uc2e4 \uc218 \uc788\uc5b4\uc694?",
            "\ud68c\uc758\uc2e4 \uc608\uc57d\uc740 \uc81c\uac00 \ub3c4\uc640\ub4dc\ub9b4 \uc218 \uc5c6\uc5b4\uc694 \u2014 \uc800\ub294 OfficeFlow \uc0c1\ud488 \ubc0f \uc11c\ube44\uc2a4 \uc9c0\uc6d0 \uc804\ubb38\uc774\uac70\ub4e0\uc694. \ud68c\uc758\uc2e4 \uc608\uc57d\uc740 \uc0ac\ub0b4 \ucd1d\ubb34 \ub2f4\ub2f9\uc790\ubd84\uaed8 \ubb38\uc758\ud574 \ubcf4\uc138\uc694. \ub2e4\ub978 \ub3c4\uc6c0 \ud544\uc694\ud55c \uac8c \uc788\ub098\uc694?"
        ),
        (
            "\uc5c5\ubb34\uc6a9\uc73c\ub85c \uc5b4\ub5a4 \ub178\ud2b8\ubd81\uc774 \uac00\uc7a5 \uc88b\uc744\uae4c\uc694?",
            "\uc88b\uc740 \uc9c8\ubb38\uc774\uc9c0\ub9cc \uc804\uc790\uae30\uae30\uc640 \ub178\ud2b8\ubd81\uc740 \uc800\ud76c \ucde8\uae09 \ubc94\uc704 \ubc16\uc774\uc5d0\uc694. \uc800\ud76c\ub294 \uc0ac\ubb34\uc6a9\ud488\uacfc \uc885\uc774\ub958\uc5d0 \uc9d1\uc911\ud558\uace0 \uc788\uc5b4\uc694. \ub178\ud2b8\ubd81 \ucd94\ucc9c\uc740 IT \ub9ac\ubdf0 \uc0ac\uc774\ud2b8\ub97c \ucc38\uace0\ud574 \ubcf4\uc138\uc694. \ub2e4\ub978 \ub3c4\uc6c0 \ud544\uc694\ud55c \uac8c \uc788\ub098\uc694?"
        ),
        (
            "\uc774\uba54\uc77c \uc124\uc815 \uc880 \ub3c4\uc640\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\uc774\uba54\uc77c \uc124\uc815\uc740 \uc81c \uc804\ubb38 \ubd84\uc57c\uac00 \uc544\ub2c8\uc5d0\uc694 \u2014 \uc800\ub294 OfficeFlow \uc0c1\ud488\uacfc \uc11c\ube44\uc2a4 \ubb38\uc758\ub97c \ub3c4\uc640\ub4dc\ub9bd\ub2c8\ub2e4. \uc0ac\ub0b4 IT \ud300\uc774\ub098 \uc774\uba54\uc77c \uc11c\ube44\uc2a4 \uc81c\uacf5\uc5c5\uccb4 \uc9c0\uc6d0\ud300\uc5d0 \ubb38\uc758\ud558\uc2dc\ub294 \uac8c \uc88b\uaca0\uc5b4\uc694!"
        ),
        (
            "\ub18d\ub2f4 \ud558\ub098 \ud574\uc8fc\uc138\uc694.",
            "\ud558\ud558! \uc694\uccad\uc740 \uac10\uc0ac\ud558\uc9c0\ub9cc, \uc800\ub294 \uc0ac\ubb34\uc6a9\ud488 \uad00\ub828 \ubb38\uc758\ub97c \uc81c\uc77c \uc798 \ub3c4\uc640\ub4dc\ub9b4 \uc218 \uc788\uc5b4\uc694. \uc0c1\ud488 \uc815\ubcf4\ub098 \uc11c\ube44\uc2a4 \ubb38\uc758 \uc788\uc73c\uc2dc\uba74 \uc5b8\uc81c\ub4e0 \ub9d0\uc500\ud574 \uc8fc\uc138\uc694!"
        ),
        (
            "\ud53c\uc790 \uc8fc\ubb38 \uc880 \ud574\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\ud560 \uc218 \uc788\uc73c\uba74 \uc88b\uaca0\uc9c0\ub9cc\uc694! \uc800\ub294 OfficeFlow \uc0c1\ud488\uacfc \uc11c\ube44\uc2a4\ub85c \ud55c\uc815\ub418\uc5b4 \uc788\uc5b4\uc694. \ud53c\uc790\ub294 \uc990\uaca8 \uc4f0\uc2dc\ub294 \ubc30\ub2ec \uc571\uc744 \uc774\uc6a9\ud574 \ubcf4\uc138\uc694. \uc2a4\ud14c\uc774\ud50c\ub7ec\ub098 \ud3ec\uc2a4\ud2b8\uc787\uc774\ub77c\ub3c4 \ud544\uc694\ud558\uc2e0\uac00\uc694?"
        ),
        (
            "\ud504\ub791\uc2a4\uc758 \uc218\ub3c4\ub294 \uc5b4\ub514\uc778\uac00\uc694?",
            "\ud30c\ub9ac\uc785\ub2c8\ub2e4! \ud558\uc9c0\ub9cc \uc81c \uc9c4\uc9dc \uc804\ubb38 \ubd84\uc57c\ub294 OfficeFlow \uc0c1\ud488\uacfc \uc11c\ube44\uc2a4\uc608\uc694. \uc0ac\ubb34\uc6a9\ud488 \uad00\ub828\ud574\uc11c \ub3c4\uc640\ub4dc\ub9b4 \uc77c\uc774 \uc788\uc744\uae4c\uc694?"
        ),
        (
            "\ud734\uac00 \uacc4\ud68d \uc880 \ub3c4\uc640\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\ud734\uac00 \uacc4\ud68d\uc740 \uc7ac\ubbf8\uc788\uc5b4 \ubcf4\uc774\ub294\ub370, \uc81c \ub2f4\ub2f9 \ubc94\uc704\uac00 \uc544\ub2c8\uc5d0\uc694. \uc800\ub294 OfficeFlow \uc0c1\ud488 \ubc0f \uc11c\ube44\uc2a4 \uc9c0\uc6d0\uc744 \ub2f4\ub2f9\ud574\uc694. \uc0ac\ubb34\uc2e4\uc5d0 \ud544\uc694\ud55c \uac8c \uc788\uc73c\uba74 \ub9d0\uc500\ud574 \uc8fc\uc138\uc694!"
        ),
        (
            "\ud504\ub9b0\ud130\ub098 \ud1a0\ub108 \uce74\ud2b8\ub9ac\uc9c0 \ud310\ub9e4\ud558\ub098\uc694?",
            "\ud604\uc7ac \ud504\ub9b0\ud130\ub098 \ud1a0\ub108 \uce74\ud2b8\ub9ac\uc9c0\ub294 \ucde8\uae09\ud558\uc9c0 \uc54a\uace0 \uc788\uc5b4\uc694 \u2014 \uc800\ud76c\ub294 \uc0ac\ubb34\uc6a9\ud488, \uc885\uc774\ub958, \ucc45\uc0c1 \uc561\uc138\uc11c\ub9ac\uc5d0 \uc9d1\uc911\ud558\uace0 \uc788\uc5b4\uc694. \uce74\ud0c8\ub85c\uadf8 \ub0b4\uc5d0\uc11c \ub3c4\uc640\ub4dc\ub9b4 \ub2e4\ub978 \uc0c1\ud488\uc774 \uc788\uc744\uae4c\uc694?"
        ),
        (
            "\uc790\ub3d9\ucc28 \ubcf4\ud5d8 \uccad\uad6c \uc880 \ub3c4\uc640\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\uc8c4\uc1a1\ud558\uc9c0\ub9cc \uadf8\uac74 \uc81c \uc601\uc5ed \ubc16\uc774\uc5d0\uc694! \uc800\ub294 OfficeFlow \uc0c1\ud488\uacfc \uc11c\ube44\uc2a4\ub9cc \ub3c4\uc640\ub4dc\ub9b4 \uc218 \uc788\uc5b4\uc694. \ubcf4\ud5d8\uc740 \uac00\uc785\ud558\uc2e0 \ubcf4\ud5d8\uc0ac\uc5d0 \uc9c1\uc811 \ubb38\uc758\ud574 \ubcf4\uc138\uc694."
        ),
        (
            "\uc8fc\uc2dd \ud22c\uc790 \uc870\uc5b8 \uc880 \ub4e3\uace0 \uc2f6\uc5b4\uc694.",
            "\ud22c\uc790 \uc870\uc5b8\uc740 \uc81c\uac00 \ub4dc\ub9b4 \uc218 \uc788\ub294 \uac8c \uc544\ub2c8\uc5d0\uc694 \u2014 \uc800\ub294 OfficeFlow \uc0c1\ud488 \uc9c0\uc6d0\uc744 \ub2f4\ub2f9\ud574\uc694. \uc7ac\ubb34 \uc0c1\ub2f4\uc740 \uc7ac\ubb34 \uc124\uacc4\uc0ac\uc640 \uc0c1\uc758\ud558\uc2dc\ub294 \uac8c \uc88b\uaca0\uc5b4\uc694. \uc0ac\ubb34\uc6a9\ud488\uc740 \ud544\uc694\ud558\uc2e0\uac00\uc694?"
        ),
        (
            "\ud30c\uc774\uc36c \uc2a4\ud06c\ub9bd\ud2b8 \uc880 \uc9dc \uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\ubd80\ud0c1\uc740 \uac10\uc0ac\ud55c\ub370, \ucf54\ub529\uc740 \uc81c \ub2f4\ub2f9 \ubc94\uc704 \ubc16\uc774\uc5d0\uc694! \uc800\ub294 OfficeFlow \uc0c1\ud488\uacfc \uc11c\ube44\uc2a4 \uad00\ub828 \ubb38\uc758\ub97c \ub3c4\uc640\ub4dc\ub824\uc694. \ud504\ub85c\uadf8\ub798\ubc0d\uc740 Stack Overflow \uac19\uc740 \uac1c\ubc1c\uc790 \ucee4\ubba4\ub2c8\ud2f0\ub97c \uc774\uc6a9\ud574 \ubcf4\uc138\uc694."
        ),
        (
            "\ub2e4\uc774\uc5b4\ud2b8\ud558\ub294 \uc88b\uc740 \ubc29\ubc95\uc774 \uc788\uc744\uae4c\uc694?",
            "\uac74\uac15\uc774\ub098 \uc6b4\ub3d9 \uad00\ub828 \uc870\uc5b8\uc740 \uc81c \uc601\uc5ed\uc774 \uc544\ub2c8\uc5d0\uc694 \u2014 \uc758\uc0ac \uac19\uc740 \uc804\ubb38\uac00\uc640 \uc0c1\ub2f4\ud558\uc2dc\ub294 \uac78 \ucd94\ucc9c\ub4dc\ub824\uc694. \uc0ac\ubb34\uc6a9\ud488 \uad00\ub828\uc740 \uc798 \ub3c4\uc640\ub4dc\ub9b4 \uc218 \uc788\uc5b4\uc694! \ub3c4\uc6c0 \ud544\uc694\ud55c \uac8c \uc788\ub098\uc694?"
        ),
        (
            "\ud0dd\uc2dc \uc880 \ubd88\ub7ec\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\ud0dd\uc2dc \uc608\uc57d\uc740 \uc548 \ub418\uc9c0\ub9cc \uc0ac\ubb34\uc6a9\ud488\uc740 \ub3c4\uc640\ub4dc\ub9b4 \uc218 \uc788\uc5b4\uc694! OfficeFlow \uad00\ub828\ud574\uc11c \ud544\uc694\ud558\uc2e0 \uac8c \uc788\ub098\uc694?"
        ),
        (
            "\uc0ac\ubb34\uc2e4\uc6a9 \ucee4\ud53c\ub098 \uac04\uc2dd\ub3c4 \ud310\ub9e4\ud558\ub098\uc694?",
            "\uc74c\uc2dd\uc774\ub098 \uc74c\ub8cc\ub294 \ucde8\uae09\ud558\uc9c0 \uc54a\uc544\uc694 \u2014 \uc800\ud76c \uce74\ud0c8\ub85c\uadf8\ub294 \uc0ac\ubb34\uc6a9\ud488, \uc885\uc774\ub958, \ucc45\uc0c1 \uc561\uc138\uc11c\ub9ac \uc704\uc8fc\uc608\uc694. \uadf8 \ubc94\uc704 \uc548\uc5d0\uc11c \ub3c4\uc640\ub4dc\ub9b4 \uac8c \uc788\uc744\uae4c\uc694?"
        ),
        (
            "\uc774 \uae30\uc0ac \uc694\uc57d \uc880 \ud574\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\uae30\uc0ac \uc694\uc57d\uc740 \uc81c \ub2f4\ub2f9 \ubc94\uc704 \ubc16\uc774\uc5d0\uc694. \uc800\ub294 OfficeFlow \uc0c1\ud488 \ubc0f \uc11c\ube44\uc2a4 \uc9c0\uc6d0\uc744 \ub2f4\ub2f9\ud574\uc694. \uce74\ud0c8\ub85c\uadf8\uc5d0 \uc788\ub294 \uc0c1\ud488 \uad00\ub828\ud574\uc11c \uc5b8\uc81c\ub4e0 \ubb38\uc758\ud574 \uc8fc\uc138\uc694!"
        ),
        (
            "\uc9c0\uae08 \uac00\uc7a5 \uc88b\uc740 OTT \uc11c\ube44\uc2a4\uac00 \ubb50\uc608\uc694?",
            "\uc5d4\ud130\ud14c\uc778\uba3c\ud2b8 \ucd94\ucc9c\uc740 \uc81c \uc601\uc5ed \ubc16\uc774\uc5d0\uc694! \uc800\ub294 OfficeFlow \uc0c1\ud488\uacfc \uc11c\ube44\uc2a4\ub97c \ub2f4\ub2f9\ud574\uc694. \uc0ac\ubb34\uc2e4\uc5d0 \ud544\uc694\ud55c \uac8c \uc788\uc73c\uba74 \ub9d0\uc500\ud574 \uc8fc\uc138\uc694."
        ),
        (
            "\uc774\uc0ac \uc880 \ub3c4\uc640\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\uc774\uc0ac\ub294 \ub3c4\uc640\ub4dc\ub9b4 \uc218 \uc5c6\uc9c0\ub9cc, \uc774\uc0ac\uc6a9 \ubc15\uc2a4 \ud14c\uc774\ud504\ub098 \ub77c\ubca8\uc774 \ud544\uc694\ud558\uc2dc\uba74 \uc800\ud76c\uac00 \ub3c4\uc640\ub4dc\ub9b4 \uc218 \uc788\uc5b4\uc694! \uc0ac\ubb34\uc6a9\ud488 \uad00\ub828\ud574\uc11c \ub3c4\uc640\ub4dc\ub9b4 \uc77c\uc774 \uc788\uc744\uae4c\uc694?"
        ),
        (
            "\uccad\uc18c\uc6a9\ud488\ub3c4 \ud310\ub9e4\ud558\ub098\uc694?",
            "\uccad\uc18c\uc6a9\ud488\uc740 \ud604\uc7ac \uce74\ud0c8\ub85c\uadf8\uc5d0 \uc5c6\uc5b4\uc694 \u2014 \uc800\ud76c\ub294 \uc0ac\ubb34\uc6a9\ud488, \uc885\uc774\ub958, \ud544\uae30\uad6c, \ucc45\uc0c1 \uc561\uc138\uc11c\ub9ac \uc804\ubb38\uc774\uc5d0\uc694. \ud574\ub2f9 \uce74\ud14c\uace0\ub9ac \ub0b4\uc5d0\uc11c \ud544\uc694\ud558\uc2e0 \uac8c \uc788\uc744\uae4c\uc694?"
        ),
        (
            "\ud504\ub9b0\ud130 \uc880 \uace0\uccd0\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\ud504\ub9b0\ud130 \uc218\ub9ac\ub294 \uc81c \ub2f4\ub2f9\uc774 \uc544\ub2c8\uc5d0\uc694 \u2014 \uc800\ub294 OfficeFlow \uc0c1\ud488\uacfc \uc11c\ube44\uc2a4 \ubb38\uc758\ub97c \ub3c4\uc640\ub4dc\ub824\uc694. \ud504\ub9b0\ud130 \ubb38\uc81c\ub294 \uc81c\uc870\uc0ac \uace0\uac1d\uc9c0\uc6d0\uc13c\ud130\uc5d0 \ubb38\uc758\ud558\uc2dc\ub294 \uac78 \ucd94\ucc9c\ub4dc\ub824\uc694."
        ),
        (
            "\uacc4\uc57d\uc11c \uad00\ub828 \ubc95\ub960 \uc790\ubb38\uc774 \ud544\uc694\ud574\uc694. \ub3c4\uc640\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\ubc95\ub960 \uc790\ubb38\uc740 \uc81c\uac00 \uc81c\uacf5\ud560 \uc218 \uc5c6\uc5b4\uc694. \ubcc0\ud638\uc0ac\uc640 \uc0c1\ub2f4\ud558\uc2dc\ub294 \uac78 \ucd94\ucc9c\ub4dc\ub9bd\ub2c8\ub2e4. \uc800\ub294 OfficeFlow \uc0c1\ud488 \ubc0f \uc11c\ube44\uc2a4 \ubb38\uc758 \ub2f4\ub2f9\uc774\uc5d0\uc694 \u2014 \uce74\ud0c8\ub85c\uadf8 \uad00\ub828 \ud544\uc694\ud558\uc2e0 \uac70 \uc788\ub098\uc694?"
        ),
        (
            "\uc6b0\uccb4\uad6d \uba87 \uc2dc\uae4c\uc9c0 \uc6b4\uc601\ud558\ub098\uc694?",
            "\uc6b0\uccb4\uad6d \uc6b4\uc601\uc2dc\uac04\uc740 \uc81c\uac00 \ubaa8\ub974\uc9c0\ub9cc, \uc6b0\uccb4\uad6d \uc0ac\uc774\ud2b8\uc5d0\uc11c \ud655\uc778\ud558\uc2e4 \uc218 \uc788\uc744 \uac70\uc608\uc694. \uc800\ub294 OfficeFlow \uc0c1\ud488\uacfc \uc11c\ube44\uc2a4\ub97c \ub3c4\uc640\ub4dc\ub824\uc694 \u2014 \ub3c4\uc6c0 \ud544\uc694\ud55c \uac8c \uc788\ub098\uc694?"
        ),
        (
            "\uc800\ub294 {name}\uc785\ub2c8\ub2e4. \uc0c8 \uc9d1 \uc54c\uc544\ubcf4\ub294 \uac83 \uc880 \ub3c4\uc640\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "{name}\ub2d8 \uc548\ub155\ud558\uc138\uc694! \ubb38\uc758 \uc8fc\uc154\uc11c \uac10\uc0ac\ud558\uc9c0\ub9cc, \uc9d1 \uc54c\uc544\ubcf4\ub294 \uc77c\uc740 \uc81c \ub2f4\ub2f9 \ubc94\uc704 \ubc16\uc774\uc5d0\uc694. \uc800\ub294 OfficeFlow \uc0c1\ud488 \ubc0f \uc11c\ube44\uc2a4 \uc9c0\uc6d0\uc744 \ub2f4\ub2f9\ud569\ub2c8\ub2e4. \uc0ac\ubb34\uc6a9\ud488\uc740 \ud544\uc694\ud558\uc2e0\uac00\uc694?"
        ),
        (
            "OfficeFlow \uc9c0\uae08 \ucc44\uc6a9 \uc911\uc778\uac00\uc694?",
            "\ucc44\uc6a9 \uacf5\uace0\ub294 \ud68c\uc0ac \uc6f9\uc0ac\uc774\ud2b8\ub97c \ud655\uc778\ud558\uc2dc\uac70\ub098 corporate@officeflow.com \uc73c\ub85c \ubb38\uc758 \uc8fc\uc138\uc694. \uc800\ub294 \uc0c1\ud488 \ubc0f \uc11c\ube44\uc2a4 \uc9c0\uc6d0\uc744 \ub2f4\ub2f9\ud574\uc694. \uc0ac\ubb34\uc6a9\ud488 \uad00\ub828\ud574\uc11c \ub3c4\uc640\ub4dc\ub9b4 \uc77c\uc774 \uc788\uc744\uae4c\uc694?"
        ),
        (
            "\uc5d1\uc140 \uc0ac\uc6a9\ubc95 \uc880 \uc54c\ub824\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\uc5d1\uc140 \uad50\uc721\uc740 \uc81c \uc804\ubb38 \ubd84\uc57c\uac00 \uc544\ub2c8\uc5d0\uc694 \u2014 \uc720\ud29c\ube0c\ub098 \uc778\ud504\ub7f0 \ub4f1\uc5d0 \uc88b\uc740 \uac15\uc758\uac00 \ub9ce\uc544\uc694. \uc800\ub294 OfficeFlow \uc0c1\ud488 \uad00\ub828 \ubb38\uc758\ub97c \ub2f4\ub2f9\ud569\ub2c8\ub2e4!"
        ),
        (
            "\uc5b4\uba38\ub2c8\uaed8 \uaf43 \uc880 \ubcf4\ub0b4\uace0 \uc2f6\uc5b4\uc694. \ub3c4\uc640\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\ub9c8\uc74c\uc774 \ub530\ub73b\ud558\ub124\uc694! \ud558\uc9c0\ub9cc \uaf43 \ubc30\ub2ec\uc740 \uc81c \ub2f4\ub2f9\uc774 \uc544\ub2c8\uc5d0\uc694. \uc800\ub294 OfficeFlow \uc0ac\ubb34\uc6a9\ud488\uacfc \uc11c\ube44\uc2a4 \ub2f4\ub2f9\uc774\uc5d0\uc694. \uc0ac\ubb34\uc2e4\uc5d0 \ud544\uc694\ud55c \uac8c \uc788\uc73c\uba74 \uc54c\ub824\uc8fc\uc138\uc694!"
        ),
        (
            "\uc804\uc790\uae30\uae30\ub098 \ucef4\ud4e8\ud130 \uc561\uc138\uc11c\ub9ac\ub3c4 \ud310\ub9e4\ud558\uc2dc\ub098\uc694?",
            "\uc804\uc790\uae30\uae30\ub098 \ucef4\ud4e8\ud130 \uc561\uc138\uc11c\ub9ac\ub294 \ucde8\uae09\ud558\uc9c0 \uc54a\uc544\uc694. \uc800\ud76c \uce74\ud0c8\ub85c\uadf8\ub294 \uc0ac\ubb34\uc6a9\ud488, \uc885\uc774\ub958, \ud544\uae30\uad6c, \uc815\ub9ac\uc6a9\ud488 \uc704\uc8fc\uc608\uc694. \uadf8 \uce74\ud14c\uace0\ub9ac\uc5d0\uc11c \ub3c4\uc640\ub4dc\ub9b4 \uac8c \uc788\uc744\uae4c\uc694?"
        ),
        (
            "\uc608\uc57d\ub41c \ud56d\uacf5\ud3b8 \uc0c1\ud0dc \uc880 \ud655\uc778\ud574\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\ud56d\uacf5\ud3b8 \uc870\ud68c\ub294 \uc81c\uac00 \ud560 \uc218 \uc788\ub294 \uc77c\uc774 \uc544\ub2c8\uc5d0\uc694 \u2014 \ud56d\uacf5\uc0ac \uc571\uc774\ub098 \uc6f9\uc0ac\uc774\ud2b8\uc5d0\uc11c \ud655\uc778\ud574 \ubcf4\uc138\uc694. \uc800\ub294 OfficeFlow \uc0c1\ud488 \ubc0f \uc11c\ube44\uc2a4 \ubb38\uc758\ub97c \ub2f4\ub2f9\ud569\ub2c8\ub2e4!"
        ),
        (
            "\uc778\uc0dd\uc758 \uc758\ubbf8\uac00 \ubb58\uae4c\uc694?",
            "\uc2ec\uc624\ud55c \uc9c8\ubb38\uc774\ub124\uc694! \uadf8\uac74 \ucca0\ud559\uc790\uc5d0\uac8c \ub9e1\uae38\uac8c\uc694. \uc800\ub294 OfficeFlow \uc0c1\ud488\uacfc \uc11c\ube44\uc2a4\ub97c \ub3c4\uc640\ub4dc\ub824\uc694. \uc0ac\ubb34\uc6a9\ud488 \uad00\ub828\ud574\uc11c \uad81\uae08\ud55c \uac8c \uc788\uc73c\uc2e0\uac00\uc694?"
        ),
        (
            "\uc2e0\uc6a9\uce74\ub4dc \uccad\uad6c \uac74\uc5d0 \ub300\ud574 \uc774\uc758 \uc2e0\uccad\uc744 \ud574\uc57c \ud574\uc694. \ub3c4\uc640\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\uc2e0\uc6a9\uce74\ub4dc \uc774\uc758 \uc2e0\uccad\uc740 \uc800\ud76c\uac00 \uc544\ub2cc \uce74\ub4dc\uc0ac\ub97c \ud1b5\ud574 \uc9c4\ud589\ud558\uc154\uc57c \ud574\uc694. OfficeFlow \uccad\uad6c \uac74 \uc911\uc5d0 \uc798\ubabb\ub41c \uac8c \uc788\ub2e4\uace0 \uc0dd\uac01\ub418\uc2dc\uba74 accounts@officeflow.com \uc73c\ub85c \uc5f0\ub77d \uc8fc\uc138\uc694."
        ),
        (
            "\ud300\ube4c\ub529 \ud65c\ub3d9 \uc880 \ub3c4\uc640\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\ud300\ube4c\ub529\uc740 \uc81c \ub2f4\ub2f9 \ubc94\uc704 \ubc16\uc774\uc9c0\ub9cc \uc7ac\ubbf8\uc788\uc5b4 \ubcf4\uc5ec\uc694! \uc800\ub294 OfficeFlow \uc0c1\ud488 \ubc0f \uc11c\ube44\uc2a4 \uc9c0\uc6d0\uc744 \ub2f4\ub2f9\ud574\uc694. \ud300 \ud589\uc0ac\uc6a9 \ube44\ud488\uc774 \ud544\uc694\ud558\uc2dc\uba74 \ub9d0\uc500\ud574 \uc8fc\uc138\uc694?"
        ),
        (
            "\uc778\uccb4\uacf5\ud559 \uc758\uc790\ub098 \uc2a4\ud0e0\ub529 \ub370\uc2a4\ud06c\ub3c4 \ud310\ub9e4\ud558\ub098\uc694?",
            "\uc758\uc790\ub098 \ucc45\uc0c1 \uac19\uc740 \uac00\uad6c\ub294 \ud604\uc7ac \ucde8\uae09\ud558\uc9c0 \uc54a\uc544\uc694. \uc800\ud76c\ub294 \uc0ac\ubb34\uc6a9\ud488, \uc885\uc774\ub958, \ucc45\uc0c1 \uc561\uc138\uc11c\ub9ac \uc704\uc8fc\uc608\uc694. \uce74\ud0c8\ub85c\uadf8 \ub0b4\uc5d0\uc11c \ub3c4\uc640\ub4dc\ub9b4 \uac8c \uc788\uc744\uae4c\uc694?"
        ),
    ]
    pairs = []
    for t in templates:
        name = _rand_name()
        q = t[0].format(name=name)
        a = t[1].format(name=name)
        pairs.append((q, a))
    return pairs


def _both_questions():
    """both \uce74\ud14c\uace0\ub9ac (inventory + policy \ub3d9\uc2dc \ud3ec\ud568) (\uc9c8\ubb38, \uc751\ub2f5) \ud29c\ud50c \ub9ac\uc2a4\ud2b8 \ubc18\ud658."""
    templates = [
        (
            "{product} \uc7ac\uace0 \uc788\ub098\uc694? \uadf8\ub9ac\uace0 \ub9cc\uc57d \uc548 \ub9de\uc73c\uba74 \ubc18\ud488 \uac00\ub2a5\ud55c\uac00\uc694?",
            "{product}\ub294 \ud604\uc7ac \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4! \ubc18\ud488\uc740 returns@officeflow.com \uc5d0\uc11c RMA\ub97c \ubc1c\uae09\ubc1b\uc73c\uc2dc\uba74 30\uc77c \uc774\ub0b4 \uac00\ub2a5\ud569\ub2c8\ub2e4. \ubd88\ub7c9\ud488\uc740 \uc5b8\uc81c\ub4e0 \ubc18\ud488 \uac00\ub2a5\ud569\ub2c8\ub2e4."
        ),
        (
            "{product}\uac00 \ud544\uc694\ud55c\ub370 \uc7ac\uace0 \uc788\ub098\uc694? \uadf8\ub9ac\uace0 \ubc30\uc1a1\ube44\ub294 \uc5bc\ub9c8\uc778\uac00\uc694?",
            "{product}\ub294 \uc7ac\uace0\uac00 \uc788\uc2b5\ub2c8\ub2e4! \uc77c\ubc18 \ubc30\uc1a1\uc740 8,950\uc6d0 (3~5\uc77c), 10\ub9cc\uc6d0 \uc774\uc0c1 \uc8fc\ubb38 \uc2dc \ubb34\ub8cc\uc785\ub2c8\ub2e4. \ube60\ub978 \ubc30\uc1a1\uacfc \uc775\uc77c \ubc30\uc1a1 \uc635\uc158\ub3c4 \uc788\uc5b4\uc694."
        ),
        (
            "{product} \uc7ac\uace0 \ud655\uc778\ud558\uace0 \uc8fc\ubb38 \ubc29\ubc95\ub3c4 \uc54c\ub824\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "{product}\ub294 \uad6c\ub9e4 \uac00\ub2a5\ud569\ub2c8\ub2e4! \uc8fc\ubb38\uc740 \uc6f9 \ud3ec\ud138(www.officeflow.com)\uc774\ub098 sales@officeflow.com \uc73c\ub85c \uba54\uc77c \uc8fc\uc2dc\uba74 \ub429\ub2c8\ub2e4."
        ),
        (
            "{product}\uc640 {product2}\ub97c \ubcf4\uace0 \uc788\ub294\ub370\uc694. \ub458 \ub2e4 \uc7ac\uace0 \uc788\ub098\uc694? \uadf8\ub9ac\uace0 \ud574\uc678 \ubc30\uc1a1\ub3c4 \ub418\ub098\uc694?",
            "{product}\uc640 {product2} \ubaa8\ub450 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! \uadf8\ub9ac\uace0 \ub124, \uac00\uae4c\uc6b4 \uc544\uc2dc\uc544 \uad6d\uac00\ub294 \ubaa8\ub450 \ubc30\uc1a1 \uac00\ub2a5\ud569\ub2c8\ub2e4. \ud1b5\uad00\uc5d0 1~2\uc77c \ucd94\uac00 \uc18c\uc694\ub418\uace0, \uad00\uc138/\uc138\uae08\uc774 \ubd80\uacfc\ub420 \uc218 \uc788\uc2b5\ub2c8\ub2e4."
        ),
        (
            "\uc800\ub294 {company}\uc758 {name}\uc785\ub2c8\ub2e4. {product}\uac00 \ud544\uc694\ud55c\ub370 \uc7ac\uace0 \uc788\ub098\uc694? \uadf8\ub9ac\uace0 \ud300\uc6d0\ub4e4\uc774 \ub9c8\uc74c\uc5d0 \uc548 \ub4e4\uc5b4 \ud558\uba74 \ubc18\ud488 \uac00\ub2a5\ud55c\uac00\uc694?",
            "{name}\ub2d8 \uc548\ub155\ud558\uc138\uc694! {product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4. \ud300\uc6d0\ub4e4\uc774 \ub9c8\uc74c\uc5d0 \uc548 \ub4e4\uc5b4 \ud558\uc2dc\uba74 30\uc77c \uc774\ub0b4 \ubc18\ud488 \uac00\ub2a5\ud574\uc694 \u2014 returns@officeflow.com \uc73c\ub85c RMA \uc2e0\uccad\ub9cc \ud558\uc2dc\uba74 \ub429\ub2c8\ub2e4."
        ),
        (
            "{product} \ucde8\uae09\ud558\uc2dc\ub098\uc694? \uadf8\ub9ac\uace0 \uac00\uc7a5 \ube60\ub978 \ubc30\uc1a1 \uc635\uc158\uc774 \ubb54\uac00\uc694?",
            "{product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! \uac00\uc7a5 \ube60\ub978 \uc635\uc158\uc740 \uc775\uc77c \ubc30\uc1a1 (39,950\uc6d0)\uc785\ub2c8\ub2e4 \u2014 KST \uc815\uc624\uae4c\uc9c0 \uc8fc\ubb38\ud558\uc2dc\uba74 \ub2e4\uc74c \uc601\uc5c5\uc77c\uc5d0 \ubc1b\uc73c\uc2e4 \uc218 \uc788\uc5b4\uc694."
        ),
        (
            "{product}\uac00 \uc7ac\uace0 \uc788\ub294\uc9c0, \uadf8\ub9ac\uace0 \ubb34\ub8cc \ubc30\uc1a1 \ubc1b\uc744 \uc218 \uc788\ub294\uc9c0 \uc54c\ub824\uc8fc\uc138\uc694.",
            "{product}\ub294 \uad6c\ub9e4 \uac00\ub2a5\ud569\ub2c8\ub2e4! 10\ub9cc\uc6d0 \uc774\uc0c1 \uc8fc\ubb38 \uc2dc \uc77c\ubc18 \ubc30\uc1a1\uc774 \ubb34\ub8cc\ub85c \uc81c\uacf5\ub429\ub2c8\ub2e4."
        ),
        (
            "{product} \uc7ac\uace0 \uc788\ub098\uc694? \uc9c0\ub09c\ubc88 \uc8fc\ubb38\uc5d0\uc11c \ud30c\uc190\ud488\uc744 \ubc1b\uc558\ub294\ub370 \uc5b4\ub5bb\uac8c \ud574\uc57c \ud558\ub098\uc694?",
            "{product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4. \ud30c\uc190\ud488 \uac74\uc740 \uc0ac\uc9c4\uc744 \ucc0d\uc5b4\uc11c 48\uc2dc\uac04 \uc774\ub0b4\uc5d0 fulfillment@officeflow.com \uc73c\ub85c \uba54\uc77c \uc8fc\uc138\uc694. \uc989\uc2dc \uad50\ud658 \ub610\ub294 \ud658\ubd88\ud574 \ub4dc\ub9bd\ub2c8\ub2e4."
        ),
        (
            "{product}\ub97c \uc8fc\ubb38\ud558\uace0 \uc2f6\uace0, {order} \uc8fc\ubb38 \uac74 \ubc18\ud488\ub3c4 \ud574\uc57c \ud569\ub2c8\ub2e4.",
            "{product}\ub294 \uc7ac\uace0 \uc788\uace0 \ud3ec\ud138\uc5d0\uc11c \ubc14\ub85c \uc8fc\ubb38 \uac00\ub2a5\ud569\ub2c8\ub2e4. {order} \ubc18\ud488\uc740 returns@officeflow.com \uc73c\ub85c \uc8fc\ubb38\ubc88\ud638\uc640 \uc0ac\uc720 \ubcf4\ub0b4\uc8fc\uc2dc\uba74 4\uc601\uc5c5\uc2dc\uac04 \uc774\ub0b4\uc5d0 RMA\uac00 \ubc1c\uae09\ub429\ub2c8\ub2e4."
        ),
        (
            "{product} \uc788\ub098\uc694? \uadf8\ub9ac\uace0 \uac00\uae4c\uc6b4 \ud53d\uc5c5 \uac00\ub2a5\ud55c \ubb3c\ub958\uc13c\ud130\ub294 \uc5b4\ub514\uc778\uac00\uc694?",
            "{product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! \ubb3c\ub958\uc13c\ud130\ub294 \uc11c\uc6b8 \ubcf8\uc0ac, \ubd80\uc0b0, \uad11\uc8fc, \uc778\ucc9c 4\uacf3\uc5d0 \uc788\uc5b4\uc694. \uacb0\uc81c \uc2dc \ubc29\ubb38 \uc218\ub839\uc744 \uc120\ud0dd\ud558\uc2dc\uba74 \ub429\ub2c8\ub2e4."
        ),
        (
            "{product} \uad6c\ub9e4 \uac00\ub2a5\ud55c\uac00\uc694? \ub300\ub7c9 \uc8fc\ubb38 \uc2dc \uac00\uaca9\ub3c4 \uc54c\uace0 \uc2f6\uc5b4\uc694.",
            "{product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! \ub300\ub7c9 \uc8fc\ubb38 \uac00\uaca9\uc740 sales@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 2\ubc88\uc73c\ub85c \uc601\uc5c5\ud300\uc5d0 \ubb38\uc758 \uc8fc\uc138\uc694 \u2014 \uacac\uc801\uc744 \uc548\ub0b4\ud574 \ub4dc\ub9bd\ub2c8\ub2e4."
        ),
        (
            "{product}\uac00 \uae09\ud558\uac8c \ud544\uc694\ud55c\ub370 \uc7ac\uace0 \uc788\ub098\uc694? \uac00\uc7a5 \ube68\ub9ac \ubc1b\uc744 \uc218 \uc788\ub294 \ubc29\ubc95\uc774 \ubb50\uc608\uc694?",
            "{product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! \uc775\uc77c \ubc30\uc1a1(39,950\uc6d0)\uc744 \uc774\uc6a9\ud558\uc2dc\uace0 KST \uc815\uc624\uae4c\uc9c0 \uc8fc\ubb38\ud558\uc2dc\uba74 \ub2e4\uc74c \uc601\uc5c5\uc77c\uc5d0 \ubc1b\uc73c\uc2e4 \uc218 \uc788\uc5b4\uc694."
        ),
        (
            "{product} \uc7ac\uace0 \ud655\uc778 \ubd80\ud0c1\ub4dc\ub9ac\uace0, \uc601\uc5c5\uc2dc\uac04\ub3c4 \uc54c\ub824\uc8fc\uc138\uc694.",
            "{product}\ub294 \uad6c\ub9e4 \uac00\ub2a5\ud569\ub2c8\ub2e4! \uc804\ud654 \uc0c1\ub2f4\uc740 \ud3c9\uc77c \uc624\uc804 7\uc2dc ~ \uc624\ud6c4 7\uc2dc (KST), \ud1a0\uc694\uc77c \uc624\uc804 9\uc2dc ~ \uc624\ud6c4 2\uc2dc (KST)\uc5d0 \uac00\ub2a5\ud569\ub2c8\ub2e4. \uc77c\uc694\uc77c\uc740 \ud734\ubb34\uc785\ub2c8\ub2e4."
        ),
        (
            "\uc800\ub294 {company}\uc5d0\uc11c \uc77c\ud558\uace0 \uc788\uc2b5\ub2c8\ub2e4. {product} \uc7ac\uace0 \ud655\uc778\ud558\uace0, \ubc30\uc1a1\ube44\ub3c4 \uc54c\ub824\uc8fc\uc138\uc694.",
            "\uc548\ub155\ud558\uc138\uc694! {product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4. \uc77c\ubc18 \ubc30\uc1a1\uc740 8,950\uc6d0 (10\ub9cc\uc6d0 \uc774\uc0c1 \ubb34\ub8cc), \ube60\ub978 \ubc30\uc1a1 15,950\uc6d0, \ud2b9\uae09 24,950\uc6d0, \uc775\uc77c \ubc30\uc1a1 39,950\uc6d0\uc785\ub2c8\ub2e4."
        ),
        (
            "{product} \uc788\ub098\uc694? \uc624\ub298 \uc8fc\ubb38\ud558\uba74 \uc5b8\uc81c \ub3c4\ucc29\ud558\ub098\uc694?",
            "{product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! \uc624\ub298 KST \uc624\ud6c4 2\uc2dc \uc774\uc804\uc5d0 \uc8fc\ubb38\ud558\uc2dc\uba74 \ub2f9\uc77c \ucd9c\uace0\ub429\ub2c8\ub2e4. \uc77c\ubc18 \ubc30\uc1a1\uc740 3~5\uc601\uc5c5\uc77c \uc18c\uc694\ub429\ub2c8\ub2e4."
        ),
        (
            "{product}\uc640 {product2}\uac00 \ud544\uc694\ud569\ub2c8\ub2e4. \uc7ac\uace0 \uc788\ub098\uc694? \ubc18\ud488 \uc808\ucc28\ub3c4 \uc54c\ub824\uc8fc\uc138\uc694.",
            "{product}\uc640 {product2} \ubaa8\ub450 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! \ubc18\ud488\uc740 returns@officeflow.com \uc73c\ub85c \uc8fc\ubb38\ubc88\ud638\uc640 \ud568\uaed8 \uc5f0\ub77d \uc8fc\uc2dc\uba74 \ub429\ub2c8\ub2e4. 4\uc601\uc5c5\uc2dc\uac04 \uc774\ub0b4\uc5d0 RMA\uac00 \ubc1c\uae09\ub418\uba70, 30\uc77c \uc774\ub0b4\uc5d0 \ubc18\ud488\ud558\uc154\uc57c \ud569\ub2c8\ub2e4."
        ),
        (
            "{product}\ub97c \ube60\ub978 \ubc30\uc1a1\uc73c\ub85c \ubc1b\uc744 \uc218 \uc788\ub098\uc694? \uc7ac\uace0 \uc788\ub098\uc694?",
            "{product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! \ube60\ub978 \ubc30\uc1a1\uc740 15,950\uc6d0\uc774\uace0 2~3\uc601\uc5c5\uc77c \uc18c\uc694\ub429\ub2c8\ub2e4. KST \uc624\ud6c4 2\uc2dc \uc774\uc804 \uc8fc\ubb38\uc740 \ub2f9\uc77c \ucd9c\uace0 \ucc98\ub9ac\ub429\ub2c8\ub2e4."
        ),
        (
            "\uc548\ub155\ud558\uc138\uc694, \uc800\ub294 {name}\uc785\ub2c8\ub2e4. {company}\ub85c {product} \ubc30\uc1a1\uc744 \ubc1b\uace0 \uc2f6\uc5b4\uc694. \ubc30\uc1a1 \uc635\uc158\uc774 \uc5b4\ub5bb\uac8c \ub418\ub098\uc694?",
            "{name}\ub2d8 \uc548\ub155\ud558\uc138\uc694! {product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4. {company}\ub85c\ub294 \uc77c\ubc18 \ubc30\uc1a1(8,950\uc6d0, 3~5\uc77c), \ube60\ub978 \ubc30\uc1a1(15,950\uc6d0, 2~3\uc77c), \ud2b9\uae09(24,950\uc6d0, 1~2\uc77c), \uc775\uc77c \ubc30\uc1a1(39,950\uc6d0, \uc775\uc77c) \uc911 \uc120\ud0dd \uac00\ub2a5\ud569\ub2c8\ub2e4."
        ),
        (
            "{product} \uad6c\ub9e4 \uac00\ub2a5\ud55c\uac00\uc694? \ubb34\ub8cc \ubc30\uc1a1 \ubc1b\uc73c\ub824\uba74 \ucd5c\uc18c \uc8fc\ubb38 \uae08\uc561\uc774 \uc5bc\ub9c8\uc778\uac00\uc694?",
            "{product}\ub294 \uad6c\ub9e4 \uac00\ub2a5\ud569\ub2c8\ub2e4! \uc77c\ubc18 \ubc30\uc1a1 \ubb34\ub8cc\ub294 10\ub9cc\uc6d0 \uc774\uc0c1\ubd80\ud130, \ube60\ub978 \ubc30\uc1a1 \ubb34\ub8cc\ub294 25\ub9cc\uc6d0 \uc774\uc0c1\ubd80\ud130 \uc801\uc6a9\ub429\ub2c8\ub2e4."
        ),
        (
            "{product}\uc5d0 \uad00\uc2ec\uc774 \uc788\ub294\ub370 \uc7ac\uace0 \uc788\ub098\uc694? \uadf8\ub9ac\uace0 \ubc29\ubb38 \uc218\ub839\ub3c4 \uac00\ub2a5\ud55c\uac00\uc694?",
            "{product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! \ub124, \ubc29\ubb38 \uc218\ub839\uc740 4\uacf3\uc758 \ubb3c\ub958\uc13c\ud130 \ubaa8\ub450\uc5d0\uc11c \uac00\ub2a5\ud569\ub2c8\ub2e4 \u2014 \uc11c\uc6b8, \ubd80\uc0b0, \uad11\uc8fc, \uc778\ucc9c \ubaa8\ub450 \uac00\ub2a5\ud574\uc694. \uacb0\uc81c \uc2dc \uc120\ud0dd\ud558\uc2dc\uba74 \ub429\ub2c8\ub2e4."
        ),
        (
            "{product} \uc7ac\uace0 \ud655\uc778\ud558\uace0 \uae30\uc5c5 \uacc4\uc815 \ud61c\ud0dd\ub3c4 \uc54c\ub824\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "{product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! \uae30\uc5c5 \uacc4\uc815\uc740 \ub9de\ucda4 \ubc30\uc1a1 \uc694\uae08, \uc804\uc6a9 \ubc30\uc1a1 \ub77c\uc778, \ub9de\ucda4\ud615 \ubc18\ud488 \uc815\ucc45 \ub4f1\uc758 \ud61c\ud0dd\uc744 \ubc1b\uc73c\uc2e4 \uc218 \uc788\uc5b4\uc694. \uc790\uc138\ud55c \uc0ac\ud56d\uc740 accounts@officeflow.com \uc73c\ub85c \ubb38\uc758\ud574 \uc8fc\uc138\uc694."
        ),
        (
            "{product} \uc7ac\uace0 \uc788\ub098\uc694? {order} \uc8fc\ubb38 \uac74 \uad50\ud658\ub3c4 \ud558\uace0 \uc2f6\uc5b4\uc694.",
            "{product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! {order} \uad50\ud658\uc740 returns@officeflow.com \uc73c\ub85c \uc5f0\ub77d\ud558\uc2dc\uba74 \ub429\ub2c8\ub2e4 \u2014 RMA\uac00 \ud544\uc694\ud574\uc694. \ucc28\uc561\uc774 \ubc1c\uc0dd\ud560 \uc218 \uc788\uc2b5\ub2c8\ub2e4."
        ),
        (
            "{product}\uac00 \ud544\uc694\ud55c\ub370, \ub2f9\uc77c \ubc30\uc1a1 \ub9c8\uac10 \uc2dc\uac04\uc774 \uc5b8\uc81c\uc778\uac00\uc694?",
            "{product}\ub294 \uad6c\ub9e4 \uac00\ub2a5\ud569\ub2c8\ub2e4! KST \uc624\ud6c4 2\uc2dc \uc774\uc804\uc5d0 \uc8fc\ubb38\ud558\uc2dc\uba74 \ub2f9\uc77c \ucd9c\uace0\ub429\ub2c8\ub2e4."
        ),
        (
            "{product} \uad6c\ub9e4 \uac00\ub2a5\ud55c\uac00\uc694? \uadf8\ub9ac\uace0 \ud734\ubb34\uc77c\uc774 \uc5b8\uc81c\uc778\uac00\uc694?",
            "{product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! \uc124\ub0a0, \ucd94\uc11d, \uc5b4\ub9b0\uc774\ub0a0, \uad11\ubcf5\uc808, \uac1c\ucc9c\uc808, \ud55c\uae00\ub0a0, \uc131\ud0c4\uc808\uc5d0 \ud734\ubb34\uc774\uace0, \uba85\uc808 \uc804\ub0a0\uacfc \uc131\ud0c4\uc808 \uc774\ube0c\uc5d0\ub294 \ub2e8\ucd95 \uc601\uc5c5\uc785\ub2c8\ub2e4."
        ),
        (
            "\uac04\ub2e8\ud788 \uc5ec\ucb64\ubcfc\uac8c\uc694 \u2014 {product} \uc7ac\uace0 \uc788\ub098\uc694? \uc628\ub77c\uc778\uc5d0\uc11c \uc8fc\ubb38 \uc870\ud68c\ub3c4 \uac00\ub2a5\ud55c\uac00\uc694?",
            "{product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! \ub124, www.officeflow.com/track \uc5d0\uc11c \uc1a1\uc7a5\ubc88\ud638\ub85c \uc870\ud68c \uac00\ub2a5\ud558\uace0, \uc1a1\uc7a5\ubc88\ud638\ub294 \ucd9c\uace0 \uc2dc \uc774\uba54\uc77c\ub85c \ubc1c\uc1a1\ub429\ub2c8\ub2e4."
        ),
        (
            "{product} \ucde8\uae09\ud558\uc2dc\ub098\uc694? \uc8fc\ubb38\uc5d0 \ubb38\uc81c\uac00 \uc0dd\uae30\uba74 \uc5b4\ub514\ub85c \uc5f0\ub77d\ud558\ub098\uc694?",
            "{product}\ub294 \uad6c\ub9e4 \uac00\ub2a5\ud569\ub2c8\ub2e4! \uc8fc\ubb38 \uad00\ub828 \ubb38\uc81c\ub294 fulfillment@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 5\ubc88\uc73c\ub85c \uc5f0\ub77d \uc8fc\uc138\uc694. \ubc18\ud488\uc740 returns@officeflow.com \ub610\ub294 \ub0b4\uc120 3\ubc88\uc785\ub2c8\ub2e4."
        ),
        (
            "{product}\uac00 \ud544\uc694\ud574\uc694. \uc7ac\uace0 \uc788\ub098\uc694? \uc8fc\ubb38 \ud6c4\uc5d0 \ubc30\uc1a1\uc9c0 \ubcc0\uacbd \uac00\ub2a5\ud55c\uac00\uc694?",
            "{product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! \uc8fc\ubb38 \ud6c4 2\uc2dc\uac04 \uc774\ub0b4\uc5d0\ub294 \ubb34\ub8cc\ub85c \ubc30\uc1a1\uc9c0 \ubcc0\uacbd \uac00\ub2a5\ud569\ub2c8\ub2e4. \uadf8 \uc774\ud6c4\ub098 \ucd9c\uace0 \ud6c4\uc5d0\ub294 10,000~15,000\uc6d0\uc758 \uc7ac\ubc30\uc1a1 \uc218\uc218\ub8cc\uac00 \ubd80\uacfc\ub429\ub2c8\ub2e4."
        ),
        (
            "\uc800\ub294 {company}\uc758 {name}\uc785\ub2c8\ub2e4. {product} \uc7ac\uace0 \uc788\ub098\uc694? \uc815\uae30 \uc8fc\ubb38 \uc124\uc815\ub3c4 \uace0\ub824 \uc911\uc774\uc5d0\uc694.",
            "{name}\ub2d8 \uc548\ub155\ud558\uc138\uc694! {product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4. {company}\uc758 \uc815\uae30 \uc8fc\ubb38\uc740 sales@officeflow.com \uc73c\ub85c \uc5f0\ub77d \uc8fc\uc2dc\uba74 \ud300\uc5d0 \ub9de\ub294 \uc77c\uc815\uc73c\ub85c \uc124\uc815\ud574 \ub4dc\ub9b4 \uc218 \uc788\uc5b4\uc694."
        ),
        (
            "{product} \uc7ac\uace0 \ud655\uc778 \ubd80\ud0c1\ub4dc\ub824\uc694. \uadf8\ub9ac\uace0 \uc0c1\ud488\uc774 \ud30c\uc190\ub418\uc5b4 \uc624\uba74 \uc5b4\ub5bb\uac8c \ud558\ub098\uc694?",
            "{product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! \ud30c\uc190\ub41c \uc0c1\ud0dc\ub85c \ub3c4\ucc29\ud558\uba74 \uc0ac\uc9c4 \ucc0d\uc5b4\uc11c 48\uc2dc\uac04 \uc774\ub0b4\uc5d0 fulfillment@officeflow.com \uc73c\ub85c \uba54\uc77c \uc8fc\uc138\uc694. \uc989\uc2dc \ub300\uccb4\ud488\uc744 \ubc1c\uc1a1\ud574 \ub4dc\ub9bd\ub2c8\ub2e4."
        ),
        (
            "\uc548\ub155\ud558\uc138\uc694, {product} \uc788\ub098\uc694? \uadf8\ub9ac\uace0 \uace0\uac1d \uc11c\ube44\uc2a4 \uc804\ud654\ubc88\ud638\ub3c4 \uc54c\ub824\uc8fc\uc138\uc694.",
            "{product}\ub294 \uad6c\ub9e4 \uac00\ub2a5\ud569\ub2c8\ub2e4! \uace0\uac1d \uc11c\ube44\uc2a4 \uc804\ud654\ubc88\ud638\ub294 1588-0000 (\ub0b4\uc120 1\ubc88)\uc774\uace0, \ud3c9\uc77c \uc624\uc804 7\uc2dc ~ \uc624\ud6c4 7\uc2dc, \ud1a0\uc694\uc77c \uc624\uc804 9\uc2dc ~ \uc624\ud6c4 2\uc2dc (KST)\uc5d0 \uc751\ub300\ud569\ub2c8\ub2e4."
        ),
        (
            "{product} \uc8fc\ubb38\uc744 \uc0dd\uac01 \uc911\uc778\ub370, \uc7ac\uace0 \uc788\ub098\uc694? \uc678\uad6d\uc5b4 \uc0c1\ub2f4\ub3c4 \uac00\ub2a5\ud55c\uac00\uc694?",
            "{product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! \ub124, \ud55c\uad6d\uc5b4\uc640 \uc601\uc5b4 \uc0c1\ub2f4\uc744 \uc81c\uacf5\ud558\uace0 \uc788\uc5b4\uc694. \uc804\ud654 \uc8fc\uc2e4 \ub54c \uc694\uccad\ud574 \uc8fc\uc138\uc694."
        ),
        (
            "{product} \uc7ac\uace0 \uc788\ub294\uc9c0 \ud655\uc778 \ubd80\ud0c1\ub4dc\ub9ac\uace0, \ubc18\ud488 \ubc30\uc1a1\ube44\ub3c4 \uc54c\uace0 \uc2f6\uc5b4\uc694.",
            "{product}\ub294 \uc7ac\uace0 \uc788\uc74c\uc744 \ud655\uc778\ud574 \ub4dc\ub9bd\ub2c8\ub2e4. \ubc18\ud488 \ubc30\uc1a1\ube44\ub294 \ubd88\ub7c9\ud488\uc774\uac70\ub098 \uc800\ud76c \uce21 \ubc30\uc1a1 \uc624\ub958\uac00 \uc544\ub2cc \ud55c \uace0\uac1d \ubd80\ub2f4\uc785\ub2c8\ub2e4 \u2014 \uadf8 \uacbd\uc6b0\uc5d0\ub294 \ubb34\ub8cc\uc785\ub2c8\ub2e4."
        ),
        (
            "{product} \uad6c\ub9e4 \uac00\ub2a5\ud55c\uac00\uc694? RMA\ub294 \ubc1c\uae09\ub41c \ud6c4 \uc5bc\ub9c8 \ub3d9\uc548 \uc720\ud6a8\ud55c\uac00\uc694?",
            "{product}\ub294 \uad6c\ub9e4 \uac00\ub2a5\ud569\ub2c8\ub2e4! RMA \ubc1c\uae09 \ud6c4 14\uc77c\uac04 \uc720\ud6a8\ud569\ub2c8\ub2e4. RMA\ub294 returns@officeflow.com \uc73c\ub85c \uc694\uccad\ud574 \uc8fc\uc138\uc694."
        ),
        (
            "{company}\uc6a9\uc73c\ub85c {product}\uac00 \ud544\uc694\ud574\uc694. \ub3c4\uc11c \uc0b0\uac04 \uc9c0\uc5ed\ub3c4 \ubc30\uc1a1 \uac00\ub2a5\ud55c\uac00\uc694?",
            "{product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! \ub3c4\uc11c \uc0b0\uac04 \uc9c0\uc5ed\ub3c4 \ubc30\uc1a1 \uac00\ub2a5\ud558\uc9c0\ub9cc, \uc77c\ubc18 \ubc30\uc1a1\uc73c\ub85c\ub9cc \ucc98\ub9ac\ub429\ub2c8\ub2e4."
        ),
        (
            "{product} \uc7ac\uace0 \uc788\ub098\uc694? \ubb3c\ub958\uc13c\ud130 \uc6b4\uc601 \uc2dc\uac04\uc740 \uc5b4\ub5bb\uac8c \ub418\ub098\uc694?",
            "{product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! \ubb3c\ub958\uc13c\ud130\ub294 \ud3c9\uc77c \uc624\uc804 7\uc2dc ~ \uc624\ud6c4 6\uc2dc(\ud604\uc9c0 \uc2dc\uac04)\uc5d0 \uc6b4\uc601\ud558\uace0 \uc8fc\ub9d0\uc740 \ud734\ubb34\uc785\ub2c8\ub2e4."
        ),
        (
            "{product} \uc788\ub098\uc694? \uc601\uc5c5\uc2dc\uac04 \uc678\uc5d0 \uae34\uae09 \uc5f0\ub77d\ucc98\uac00 \uc788\ub2e4\uace0 \ub4e4\uc5c8\ub294\ub370 \ubb34\uc5c7\uc778\uac00\uc694?",
            "{product}\ub294 \uad6c\ub9e4 \uac00\ub2a5\ud569\ub2c8\ub2e4! \uc601\uc5c5\uc2dc\uac04 \uc678 \uae34\uae09 \uc0ac\uc548\uc740 urgent@officeflow.com \uc73c\ub85c \uba54\uc77c \uc8fc\uc138\uc694 \u2014 \uc911\uc694 \uc0ac\uc548\uc740 2\uc2dc\uac04 \uc774\ub0b4\uc5d0 \uc751\ub2f5\ud569\ub2c8\ub2e4."
        ),
        (
            "{product}\uac00 \ud544\uc694\ud574\uc694. \uc7ac\uace0 \uc788\ub098\uc694? \uadf8\ub9ac\uace0 \ud070 \uae00\uc528\ub85c \ub41c \uccad\uad6c\uc11c \ubc1c\uae09\ub3c4 \uac00\ub2a5\ud55c\uac00\uc694?",
            "{product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! \ub124, \ud070 \uae00\uc528 \uccad\uad6c\uc11c\ub294 \uc694\uccad \uc2dc \ubc1c\uae09 \uac00\ub2a5\ud569\ub2c8\ub2e4. accessibility@officeflow.com \uc73c\ub85c \uba54\uc77c \uc8fc\uc2dc\uba74 \uc124\uc815\ud574 \ub4dc\ub9bd\ub2c8\ub2e4."
        ),
        (
            "{product} \uc7ac\uace0 \uc788\ub098\uc694? \uc9c0\ub09c \uc8fc\ubb38\uc5d0 \ubb38\uc81c\uac00 \uc788\uc5b4\uc11c \ubc18\ud488 \uad00\ub828 \uc815\ubcf4\ub3c4 \ud544\uc694\ud574\uc694.",
            "{product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4! \ubc18\ud488\uc740 returns@officeflow.com \uc73c\ub85c \uc8fc\ubb38\ubc88\ud638\uc640 \ubb38\uc81c \ub0b4\uc6a9\uc744 \ubcf4\ub0b4\uc8fc\uc138\uc694. 4\uc601\uc5c5\uc2dc\uac04 \uc774\ub0b4\uc5d0 \ud68c\uc2e0\ub4dc\ub9bd\ub2c8\ub2e4."
        ),
        (
            "{product} \uad6c\ub9e4 \uac00\ub2a5\ud55c\uc9c0 \ud655\uc778 \ubd80\ud0c1\ub4dc\ub9ac\uace0, \uac1c\ubd09\ud55c \uc0c1\ud488\ub3c4 \ubc18\ud488 \uac00\ub2a5\ud55c\uc9c0 \uc54c\ub824\uc8fc\uc138\uc694.",
            "{product}\ub294 \uad6c\ub9e4 \uac00\ub2a5\ud569\ub2c8\ub2e4! \uc6d0\ub798 \ud3ec\uc7a5\uc774 \uc788\ub294 \ub300\ubd80\ubd84\uc758 \uac1c\ubd09 \uc0c1\ud488\uc740 30\uc77c \uc774\ub0b4 \ubc18\ud488 \uac00\ub2a5\ud569\ub2c8\ub2e4. \ub2e8, \uac1c\ubd09\ud55c \uc885\uc774 \ubb36\uc74c, \ub9de\ucda4 \uc778\uc1c4 \uc0c1\ud488, \ud2b9\uac00 \uc0c1\ud488\uc740 \ubc18\ud488 \ubd88\uac00\uc785\ub2c8\ub2e4."
        ),
        (
            "\uc548\ub155\ud558\uc138\uc694, \uc800\ub294 {name}\uc785\ub2c8\ub2e4. {product} \uc7ac\uace0 \uc788\ub098\uc694? \ub9cc\uc871 \ubcf4\uc99d \uc81c\ub3c4\ub3c4 \uc54c\uace0 \uc2f6\uc5b4\uc694.",
            "{name}\ub2d8 \uc548\ub155\ud558\uc138\uc694! {product}\ub294 \uc7ac\uace0 \uc788\uc2b5\ub2c8\ub2e4. \uc800\ud76c\ub294 \uace0\uac1d \ub9cc\uc871\uc744 \ucd5c\uc6b0\uc120\uc73c\ub85c \ud569\ub2c8\ub2e4 \u2014 30\uc77c \uc774\ub0b4 \ubc18\ud488 \uac00\ub2a5\ud558\uace0, \ubd88\ub7c9\ud488\uc740 \uc5b8\uc81c\ub4e0 \ubc18\ud488 \uac00\ub2a5\ud574\uc694. RMA\ub294 returns@officeflow.com \uc73c\ub85c \uc694\uccad\ud574 \uc8fc\uc138\uc694."
        ),
    ]
    pairs = []
    for t in templates:
        p1, p2 = _rand_products(2)
        name = _rand_name()
        company = _rand_company()
        order = _rand_order()
        q = t[0].format(product=p1, product2=p2, name=name, company=company, order=order)
        a = t[1].format(product=p1, product2=p2, name=name, company=company, order=order)
        pairs.append((q, a))
    return pairs


def _website_troubleshooting_questions():
    """website_troubleshooting \uce74\ud14c\uace0\ub9ac: \uc6f9\uc0ac\uc774\ud2b8/\ud3ec\ud138 \ubb38\uc81c\uc5d0 \ub300\ud55c (\uc9c8\ubb38, \uc751\ub2f5) \ud29c\ud50c \ub9ac\uc2a4\ud2b8 \ubc18\ud658.

    [\uc911\uc694] \uc774 \uce74\ud14c\uace0\ub9ac\ub294 Emma\uc758 \uc815\uc0c1 \uc751\ub2f5 \ud328\ud134 = "\uae30\uc220 \uc9c0\uc6d0\ud300\uc73c\ub85c \uc5f0\uacb0" \uc774\ub2e4.
    Emma\ub294 \uc6f9\uc0ac\uc774\ud2b8 \ubb38\uc81c\ub97c \uc9c1\uc811 \ud574\uacb0\ud558\uc9c0 \uc54a\uace0, \ubc18\ub4dc\uc2dc support@officeflow.com \uc73c\ub85c \uc548\ub0b4\ud574\uc57c \ud55c\ub2e4.
    """
    templates = [
        (
            "\uc6f9\uc0ac\uc774\ud2b8 \uacc4\uc815\uc5d0 \ub85c\uadf8\uc778\uc774 \uc548 \ub3fc\uc694. \ub3c4\uc640\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?",
            "\ubd88\ud3b8\uc744 \ub4dc\ub824 \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uc6f9\uc0ac\uc774\ud2b8 \ub85c\uadf8\uc778 \ubb38\uc81c\ub294 \uae30\uc220 \uc9c0\uc6d0\ud300\uc5d0\uc11c \uac00\uc7a5 \uc798 \ub3c4\uc640\ub4dc\ub9b4 \uc218 \uc788\uc5b4\uc694. support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88\uc73c\ub85c \uc5f0\ub77d \uc8fc\uc138\uc694."
        ),
        (
            "\uc8fc\ubb38\ud558\ub824\ub294\ub370 \uc6f9\uc0ac\uc774\ud2b8\uac00 \uacc4\uc18d \uba48\ucdb0\uc694.",
            "\uc815\ub9d0 \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0 \ubb38\uc758\ud574 \uc8fc\uc138\uc694 \u2014 \ubb38\uc81c\ub97c \uc9c4\ub2e8\ud574 \ub4dc\ub9b4 \uc218 \uc788\uc2b5\ub2c8\ub2e4."
        ),
        (
            "\uacb0\uc81c\ud558\ub824\ub294\ub370 \uc624\ub958 \uba54\uc2dc\uc9c0\uac00 \ub5a0\uc694.",
            "\ubd88\ud3b8\ud558\uc2dc\uaca0\uc5b4\uc694 \u2014 \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uc6f9\uc0ac\uc774\ud2b8 \uc624\ub958\ub294 \uae30\uc220 \uc9c0\uc6d0\ud300\uc774 \ub2f4\ub2f9\ud569\ub2c8\ub2e4. support@officeflow.com \uc73c\ub85c \uba54\uc77c \ubcf4\ub0b4\uc2dc\uac70\ub098 1588-0000 \ub0b4\uc120 6\ubc88\uc73c\ub85c \uc804\ud654 \uc8fc\uc138\uc694."
        ),
        (
            "\uc6f9\uc0ac\uc774\ud2b8 \uac80\uc0c9 \uae30\ub2a5\uc774 \uc791\ub3d9\ud558\uc9c0 \uc54a\uc544\uc694.",
            "\uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uac80\uc0c9 \ubb38\uc81c\ub294 \uae30\uc220 \uc9c0\uc6d0\ud300\uc5d0\uc11c \ud655\uc778\ud574 \ub4dc\ub9b4 \uc218 \uc788\uc5b4\uc694. support@officeflow.com \uc73c\ub85c \uba54\uc77c \ubcf4\ub0b4\uc2dc\uac70\ub098 1588-0000 \ub0b4\uc120 6\ubc88\uc73c\ub85c \uc804\ud654 \uc8fc\uc138\uc694."
        ),
        (
            "\ube44\ubc00\ubc88\ud638\ub97c \uc78a\uc5b4\ubc84\ub838\ub294\ub370 \uc7ac\uc124\uc815 \ub9c1\ud06c \uba54\uc77c\uc774 \uc548 \uc640\uc694.",
            "\ube44\ubc00\ubc88\ud638 \ubb38\uc81c\ub85c \ubd88\ud3b8\uc744 \ub4dc\ub824 \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0 \uc5f0\ub77d \uc8fc\uc2dc\uba74 \uc811\uadfc \uad8c\ud55c \uc7ac\uc124\uc815\uc744 \ub3c4\uc640\ub4dc\ub9bd\ub2c8\ub2e4."
        ),
        (
            "\uc624\ub298 \uc6f9\uc0ac\uc774\ud2b8\uac00 \ub108\ubb34 \ub290\ub824\uc694. \ubb34\uc2a8 \ubb38\uc81c \uc788\ub098\uc694?",
            "\ub290\ub9b0 \uc751\ub2f5\uc73c\ub85c \ubd88\ud3b8\uc744 \ub4dc\ub824 \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uc131\ub2a5 \ubb38\uc81c\ub294 \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0\uc11c \ud655\uc778\ud574 \ub4dc\ub9b4 \uc218 \uc788\uc5b4\uc694."
        ),
        (
            "\ud3ec\ud138\uc5d0\uc11c \uc8fc\ubb38 \ub0b4\uc5ed \ud398\uc774\uc9c0\ub97c \ubabb \ucc3e\uaca0\uc5b4\uc694.",
            "\uc6f9 \ud3ec\ud138 \uc0ac\uc6a9 \uad00\ub828 \ub3c4\uc6c0\uc740 \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0 \uc5f0\ub77d \uc8fc\uc138\uc694. \ud3c9\uc77c \uc624\uc804 8\uc2dc ~ \uc624\ud6c4 8\uc2dc (KST)\uc5d0 \uc751\ub300\ud569\ub2c8\ub2e4."
        ),
        (
            "\uc8fc\ubb38 \uc644\ub8cc\ud558\ub824\ub294\ub370 \uacb0\uc81c \ud398\uc774\uc9c0\uac00 \ub85c\ub529\uc774 \uc548 \ub3fc\uc694.",
            "\uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uacb0\uc81c \ud398\uc774\uc9c0 \ubb38\uc81c\ub294 \uae30\uc220 \uc9c0\uc6d0\ud300\uc774 \ub3c4\uc640\ub4dc\ub824\uc57c \ud574\uc694. support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88\uc73c\ub85c \uc5f0\ub77d \uc8fc\uc138\uc694."
        ),
        (
            "\uba87 \ubd84\ub9c8\ub2e4 \uc790\ub3d9\uc73c\ub85c \ub85c\uadf8\uc544\uc6c3\ub3fc\uc694.",
            "\uc815\ub9d0 \ubd88\ud3b8\ud558\uc2dc\uaca0\uc5b4\uc694! \uc138\uc158 \uad00\ub828 \ubb38\uc81c\ub294 \uae30\uc220 \uc9c0\uc6d0\ud300\uc5d0\uc11c \ud655\uc778\ud574 \ub4dc\ub9b4 \uc218 \uc788\uc5b4\uc694. support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88\uc73c\ub85c \uc5f0\ub77d \uc8fc\uc138\uc694."
        ),
        (
            "\uc6f9\uc0ac\uc774\ud2b8\uc5d0 \uc0c1\ud488 \uc774\ubbf8\uc9c0\uac00 \uc548 \ub098\uc640\uc694.",
            "\uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uc774\ubbf8\uc9c0\uac00 \uc548 \ubcf4\uc774\ub294 \ub4f1 \ud45c\uc2dc \ubb38\uc81c\ub294 \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0 \ubb38\uc758\ud574 \uc8fc\uc138\uc694."
        ),
        (
            "\ud3ec\ud138\uc5d0\uc11c \uccad\uad6c\uc11c\ub97c \ub2e4\uc6b4\ub85c\ub4dc\ud560 \uc218 \uc5c6\uc5b4\uc694.",
            "\ubd88\ud3b8\uc744 \ub4dc\ub824 \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uccad\uad6c\uc11c \ub2e4\uc6b4\ub85c\ub4dc \ubb38\uc81c\ub294 \uae30\uc220 \uc9c0\uc6d0\ud300\uc5d0\uc11c \ub3c4\uc640\ub4dc\ub9b4 \uc218 \uc788\uc5b4\uc694. support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88\uc73c\ub85c \uc5f0\ub77d \uc8fc\uc138\uc694."
        ),
        (
            "\ubaa8\ubc14\uc77c \uc6f9\uc0ac\uc774\ud2b8\uac00 \uc791\ub3d9\ud558\uc9c0 \uc54a\uc544\uc694 \u2014 \ubc84\ud2bc\uc774 \uc548 \ub20c\ub824\uc694.",
            "\ubaa8\ubc14\uc77c \ubb38\uc81c\ub85c \ubd88\ud3b8\uc744 \ub4dc\ub824 \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0 \uc54c\ub824\uc8fc\uc2dc\uba74 \uc218\uc815\ud574 \ub4dc\ub9ac\uaca0\uc2b5\ub2c8\ub2e4."
        ),
        (
            "\uacc4\uc815\uc5d0\uc11c \ubc30\uc1a1\uc9c0 \uc8fc\uc18c\ub97c \ubcc0\uacbd\ud558\ub824\ub294\ub370 \uc800\uc7a5 \ubc84\ud2bc\uc774 \uc791\ub3d9\ud558\uc9c0 \uc54a\uc544\uc694.",
            "\uacc4\uc815 \ud398\uc774\uc9c0 \uae30\ub2a5 \ubb38\uc81c\ub294 \uae30\uc220 \uc9c0\uc6d0\ud300\uc774 \ub2f4\ub2f9\ud569\ub2c8\ub2e4. support@officeflow.com \uc73c\ub85c \uba54\uc77c \ubcf4\ub0b4\uc2dc\uac70\ub098 1588-0000 \ub0b4\uc120 6\ubc88\uc73c\ub85c \uc804\ud654 \uc8fc\uc138\uc694."
        ),
        (
            "\uc0c1\ud488 \ub9c1\ud06c\ub97c \ud074\ub9ad\ud588\ub294\ub370 404 \uc624\ub958\uac00 \ub5a0\uc694.",
            "\ub9c1\ud06c \uc624\ub958\ub85c \ubd88\ud3b8\uc744 \ub4dc\ub824 \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uae30\uc220 \uc9c0\uc6d0\ud300\uc5d0\uc11c \ud655\uc778\ud574 \ub4dc\ub9b4 \uc218 \uc788\uc5b4\uc694. support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88\uc73c\ub85c \uc5f0\ub77d \uc8fc\uc138\uc694."
        ),
        (
            "\uc7a5\ubc14\uad6c\ub2c8\uc5d0 \uc0c1\ud488\uc744 \ub2f4\uc558\ub294\ub370 \ube44\uc5b4\uc788\ub2e4\uace0 \ub098\uc640\uc694.",
            "\uc774\uc0c1\ud558\ub124\uc694 \u2014 \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uc7a5\ubc14\uad6c\ub2c8 \ubb38\uc81c\ub294 \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0\uc11c \uac00\uc7a5 \uc798 \ub3c4\uc640\ub4dc\ub9b4 \uc218 \uc788\uc5b4\uc694."
        ),
        (
            "\uc0ac\uc774\ud2b8\uc5d0\uc11c \uc8fc\ubb38 \ubc30\uc1a1 \uc870\ud68c \ud398\uc774\uc9c0\uc5d0 \uc811\uadfc\uc774 \uc548 \ub3fc\uc694.",
            "\uc8c4\uc1a1\ud569\ub2c8\ub2e4! \ubc30\uc1a1 \uc870\ud68c \ud398\uc774\uc9c0 \ubb38\uc81c\ub294 \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com)\uc5d0 \ubb38\uc758\ud574 \uc8fc\uc138\uc694. www.officeflow.com/track \uc5d0\uc11c \uc9c1\uc811 \uc870\ud68c\ud574 \ubcf4\uc2e4 \uc218\ub3c4 \uc788\uc5b4\uc694."
        ),
        (
            "\uacb0\uc81c \ud398\uc774\uc9c0\uc5d0 \ucfe0\ud3f0 \ucf54\ub4dc \uc785\ub825\ub780\uc774 \uc548 \ubcf4\uc5ec\uc694.",
            "\uc8c4\uc1a1\ud569\ub2c8\ub2e4! \ucfe0\ud3f0 \ucf54\ub4dc \uc785\ub825\ub780\uc774 \uc548 \ubcf4\uc774\ub294 \ubb38\uc81c\ub294 \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0 \ubb38\uc758\ud574 \uc8fc\uc138\uc694."
        ),
        (
            "\ubc29\uae08 \ub85c\uadf8\uc778\ud588\ub294\ub370 '\uc138\uc158 \ub9cc\ub8cc' \uc624\ub958\uac00 \ub5a0\uc694.",
            "\ubd88\ud3b8\ud558\uc2dc\uaca0\uc5b4\uc694! \uc138\uc158 \ub9cc\ub8cc \ubb38\uc81c\ub294 \uae30\uc220 \uc9c0\uc6d0\ud300\uc774 \ub3c4\uc640\ub4dc\ub824\uc57c \ud574\uc694. support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88\uc73c\ub85c \uc5f0\ub77d \uc8fc\uc138\uc694."
        ),
        (
            "\uc6f9\uc0ac\uc774\ud2b8\uc5d0\uc11c \uc0c8 \uacc4\uc815 \uc0dd\uc131\uc774 \uc548 \ub3fc\uc694.",
            "\uacc4\uc815 \uc0dd\uc131 \ubb38\uc81c\ub85c \ubd88\ud3b8\uc744 \ub4dc\ub824 \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uae30\uc220 \uc9c0\uc6d0\ud300\uc5d0\uc11c \ub3c4\uc640\ub4dc\ub9b4 \uc218 \uc788\uc5b4\uc694. support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88\uc73c\ub85c \uc5f0\ub77d \uc8fc\uc138\uc694."
        ),
        (
            "\uc0c1\ud488 \ud398\uc774\uc9c0 \ud544\ud130 \uc0ac\uc6a9\uc5d0 \ubb38\uc81c\uac00 \uc788\uc5b4\uc694.",
            "\ud544\ud130 \ubb38\uc81c\ub85c \ubd88\ud3b8\uc744 \ub4dc\ub824 \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0 \ubb38\uc758\ud574 \uc8fc\uc2dc\uba74 \ud655\uc778\ud574 \ub4dc\ub9b4\uac8c\uc694."
        ),
        (
            "\uc800\ub294 {name}\uc774\uace0 \uacc4\uc815 \ud3ec\ud138\uc5d0 \uc544\uc608 \uc811\uadfc\uc774 \uc548 \ub3fc\uc694.",
            "{name}\ub2d8 \uc548\ub155\ud558\uc138\uc694, \uc815\ub9d0 \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \ud3ec\ud138 \uc811\uadfc \ubb38\uc81c\ub294 \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0 \uc5f0\ub77d \uc8fc\uc138\uc694."
        ),
        (
            "\uc800\ub294 {company}\uc5d0 \uc788\ub294\ub370, \ud68c\uc0ac \ub0b4 \uc5ec\ub7ec \uc0ac\ub78c\uc774 OfficeFlow \ud3ec\ud138\uc5d0 \ub85c\uadf8\uc778\uc774 \uc548 \ub3fc\uc694.",
            "{company}\uc5d0\uc11c \ub2e4\uc218\uac00 \uacaa\uace0 \uacc4\uc2e0 \ubb38\uc81c\ub85c \ubd88\ud3b8\uc744 \ub4dc\ub824 \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0 \uc5f0\ub77d\ud558\uc2dc\uba74 \ub3c4\uc640\ub4dc\ub9ac\uaca0\uc2b5\ub2c8\ub2e4 \u2014 \uacc4\uc815 \uc218\uc900\uc758 \ubb38\uc81c\uc77c \uc218\ub3c4 \uc788\uc5b4\uc694."
        ),
        (
            "\uc6f9\uc0ac\uc774\ud2b8\uac00 \uc791\ub144 \uac00\uaca9\uc744 \ubcf4\uc5ec\uc918\uc694. \ubc84\uadf8\uc778\uac00\uc694?",
            "\ud45c\uc2dc \ubb38\uc81c\ub85c \ubcf4\uc774\ub124\uc694! \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0 \uac00\uaca9 \ud45c\uc2dc \ubb38\uc81c\ub85c \uc2e0\uace0\ud574 \uc8fc\uc138\uc694."
        ),
        (
            "\uc6f9\uc0ac\uc774\ud2b8\uc5d0\uc11c \uc704\uc2dc\ub9ac\uc2a4\ud2b8\uc5d0 \uc0c1\ud488\uc744 \ucd94\uac00\ud560 \uc218 \uc5c6\uc5b4\uc694.",
            "\uc704\uc2dc\ub9ac\uc2a4\ud2b8 \ubb38\uc81c\ub85c \ubd88\ud3b8\uc744 \ub4dc\ub824 \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uc0ac\uc774\ud2b8 \uae30\ub2a5 \ubb38\uc81c\ub294 \uae30\uc220 \uc9c0\uc6d0\ud300\uc774 \ub2f4\ub2f9\ud569\ub2c8\ub2e4. support@officeflow.com \uc73c\ub85c \uba54\uc77c \ubcf4\ub0b4\uc2dc\uac70\ub098 1588-0000 \ub0b4\uc120 6\ubc88\uc73c\ub85c \uc804\ud654 \uc8fc\uc138\uc694."
        ),
        (
            "\uc8fc\ubb38 \ub0b4\uc5ed \ubcf4\ub824\uace0 \ud558\ub294\ub370 \uc0ac\uc774\ud2b8\uac00 \uacc4\uc18d \ud648\ud398\uc774\uc9c0\ub85c \ub9ac\ub2e4\uc774\ub809\ud2b8\ud574\uc694.",
            "\ubc84\uadf8 \uac19\ub124\uc694 \u2014 \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc73c\ub85c \uc2e0\uace0\ud574 \uc8fc\uc138\uc694."
        ),
        (
            "\ud3ec\ud138\uc5d0\uc11c PO \ubb38\uc11c\ub97c \uc5c5\ub85c\ub4dc\ud560 \uc218 \uc5c6\uc5b4\uc694.",
            "\ubb38\uc11c \uc5c5\ub85c\ub4dc \ubb38\uc81c\ub294 \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0 \ubb38\uc758\ud574 \uc8fc\uc138\uc694."
        ),
        (
            "2\ub2e8\uacc4 \uc778\uc99d \ucf54\ub4dc\uac00 \ud734\ub300\ud3f0\uc73c\ub85c \uc548 \uc640\uc694.",
            "2\ub2e8\uacc4 \uc778\uc99d \ubb38\uc81c\ub85c \ubd88\ud3b8\uc744 \ub4dc\ub824 \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uae30\uc220 \uc9c0\uc6d0\ud300\uc5d0\uc11c \ud574\uacb0\uc744 \ub3c4\uc640\ub4dc\ub9b4 \uc218 \uc788\uc5b4\uc694. support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88\uc73c\ub85c \uc5f0\ub77d \uc8fc\uc138\uc694."
        ),
        (
            "\uacb0\uc81c\ud560 \ub54c\ub9c8\ub2e4 '\uc11c\ubc84 \uc624\ub958' \ud398\uc774\uc9c0\uac00 \ub5a0\uc694.",
            "\uc11c\ubc84 \uc624\ub958\ub85c \ubd88\ud3b8\uc744 \ub4dc\ub824 \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0 \uacb0\uc81c \ubb38\uc81c\ub97c \uc2e0\uace0\ud574 \uc8fc\uc138\uc694."
        ),
        (
            "Firefox\uc5d0\uc11c \uc6f9\uc0ac\uc774\ud2b8\uac00 \uc81c\ub300\ub85c \uc791\ub3d9\ud558\uc9c0 \uc54a\uc544\uc694. Chrome\uc5d0\uc11c\ub294 \uc798 \ub418\ub294\ub370\uc694.",
            "\ube0c\ub77c\uc6b0\uc800 \ud638\ud658\uc131 \ubb38\uc81c\ub97c \uc54c\ub824\uc8fc\uc154\uc11c \uac10\uc0ac\ud569\ub2c8\ub2e4! \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0 \uc54c\ub824\uc8fc\uc138\uc694."
        ),
        (
            "\ucfe0\ud3f0 \ucf54\ub4dc\ub97c \uc801\uc6a9\ud558\ub824\ub294\ub370 \ub9cc\ub8cc\ub418\uc9c0 \uc54a\uc558\ub294\ub370\ub3c4 \uc720\ud6a8\ud558\uc9c0 \uc54a\ub2e4\uace0 \ub098\uc640\uc694.",
            "\uc8c4\uc1a1\ud569\ub2c8\ub2e4! \ucfe0\ud3f0 \ucf54\ub4dc \ubb38\uc81c\ub294 \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com)\uc5d0 \ubb38\uc758\ud574 \uc8fc\uc138\uc694. \uc601\uc5c5/\uac00\uaca9 \uad00\ub828 \ubb38\uc81c\ub77c\uba74 sales@officeflow.com \uc73c\ub85c\ub3c4 \ubb38\uc758 \uac00\ub2a5\ud569\ub2c8\ub2e4."
        ),
        (
            "\uc6f9\uc0ac\uc774\ud2b8 \uc54c\ub9bc\uc774 \uc81c\ub300\ub85c \ud45c\uc2dc\ub418\uc9c0 \uc54a\uc544\uc694.",
            "\uc54c\ub9bc \ubb38\uc81c\ub294 \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0 \ubb38\uc758\ud574 \uc8fc\uc138\uc694."
        ),
        (
            "'\ube44\ubc00\ubc88\ud638 \ucc3e\uae30'\ub97c \ud074\ub9ad\ud588\ub294\ub370 \uba54\uc77c\uc774 \uc548 \uc640\uc694.",
            "\uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uba3c\uc800 \uc2a4\ud338 \uba54\uc77c\ud568\uc744 \ud655\uc778\ud574 \ubcf4\uc2dc\uace0, \uadf8\ub798\ub3c4 \uc5c6\uc73c\uc2dc\uba74 \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0 \uc5f0\ub77d \uc8fc\uc138\uc694."
        ),
        (
            "\uacc4\uc815 \uc124\uc815 \ud398\uc774\uc9c0\uac00 \ube44\uc5b4 \uc788\uc5b4\uc694 \u2014 \uc544\ubb34\uac83\ub3c4 \ub85c\ub529\uc774 \uc548 \ub3fc\uc694.",
            "\ud45c\uc2dc \ubb38\uc81c \uac19\ub124\uc694. \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0\uc11c \ub3c4\uc640\ub4dc\ub9b4 \uc218 \uc788\uc5b4\uc694."
        ),
        (
            "\uc6f9\uc0ac\uc774\ud2b8 \uc2e4\uc2dc\uac04 \ucc44\ud305 \uae30\ub2a5\uc5d0 \ubb38\uc81c\uac00 \uc788\uc5b4\uc694.",
            "\uc2e4\uc2dc\uac04 \ucc44\ud305\uc774 \uc791\ub3d9\ud558\uc9c0 \uc54a\uc544\uc11c \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0 \ubb38\uc758\ud574 \uc8fc\uc138\uc694."
        ),
        (
            "\uc800\ub294 {company}\uc758 {name}\uc785\ub2c8\ub2e4. \uc8fc\ubb38 \ub0b4\uc5ed \ub85c\ub4dc\ud560 \ub54c\ub9c8\ub2e4 \ud3ec\ud138\uc774 \ud0c0\uc784\uc544\uc6c3\ub3fc\uc694.",
            "{name}\ub2d8 \uc548\ub155\ud558\uc138\uc694, \ud0c0\uc784\uc544\uc6c3 \ubb38\uc81c\ub85c \ubd88\ud3b8\uc744 \ub4dc\ub824 \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0 \ubb38\uc758\ud574 \uc8fc\uc2dc\uba74 \ud655\uc778\ud574 \ub4dc\ub9b4\uac8c\uc694."
        ),
        (
            "\ud3ec\ud138\uc5d0\uc11c \uacfc\uac70 \uc8fc\ubb38\uc740 \ubcf4\uc774\ub294\ub370 \ucd5c\uadfc \uc8fc\ubb38\uc740 \ud558\ub098\ub3c4 \uc548 \ubcf4\uc5ec\uc694.",
            "\uc774\uc0c1\ud558\ub124\uc694! \ud3ec\ud138\uc758 \uc8fc\ubb38 \ub370\uc774\ud130 \ub204\ub77d \ubb38\uc81c\ub294 \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0 \ubb38\uc758\ud574 \uc8fc\uc138\uc694."
        ),
        (
            "\uc6f9\uc0ac\uc774\ud2b8 \uac80\uc0c9\ucc3d\uc5d0 \ubb58 \uc785\ub825\ud574\ub3c4 \uacb0\uacfc\uac00 \uc548 \ub098\uc640\uc694.",
            "\uac80\uc0c9 \ubb38\uc81c\ub85c \ubd88\ud3b8\uc744 \ub4dc\ub824 \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0 \ubb38\uc758\ud574 \uc8fc\uc138\uc694."
        ),
        (
            "\ub85c\uadf8\uc778 \uc2dc\ub3c4\uac00 \ub108\ubb34 \ub9ce\uc544\uc11c \uacc4\uc815\uc774 \uc7a0\uacbc\uc5b4\uc694.",
            "\uc7a0\uae08 \ubb38\uc81c\ub85c \ubd88\ud3b8\uc744 \ub4dc\ub824 \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uae30\uc220 \uc9c0\uc6d0\ud300\uc5d0\uc11c \uacc4\uc815 \uc7a0\uae08\uc744 \ud574\uc81c\ud574 \ub4dc\ub9b4 \uc218 \uc788\uc5b4\uc694. support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88\uc73c\ub85c \uc5f0\ub77d \uc8fc\uc138\uc694."
        ),
        (
            "\uc6f9\uc0ac\uc774\ud2b8\uc5d0 \uc81c\uac00 \uc0b4\uc544\ubcf8 \uc801\ub3c4 \uc5c6\ub294 \uc8fc\uc18c\uac00 \ud45c\uc2dc\ub3fc\uc694.",
            "\uac71\uc815\uc2a4\ub7ec\uc6b4 \ubb38\uc81c\ub124\uc694! \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com)\uacfc \uc815\uc0b0\ud300(accounts@officeflow.com)\uc5d0 \uc5f0\ub77d\ud558\uc154\uc11c \uacc4\uc815 \uc815\ubcf4 \uac80\ud1a0 \ubc0f \uc218\uc815\uc744 \ubc1b\uc544\uc8fc\uc138\uc694."
        ),
        (
            "\uc8fc\ubb38 \ub0b4\uc5ed\uc5d0\uc11c \uc7ac\uc8fc\ubb38 \ubc84\ud2bc\uc744 \ub20c\ub800\ub294\ub370 \uc544\ubb34 \ubc18\uc751\uc774 \uc5c6\uc5b4\uc694.",
            "\uc7ac\uc8fc\ubb38 \ubc84\ud2bc \ubb38\uc81c\ub85c \ubd88\ud3b8\uc744 \ub4dc\ub824 \uc8c4\uc1a1\ud569\ub2c8\ub2e4! \uae30\uc220 \uc9c0\uc6d0\ud300(support@officeflow.com \ub610\ub294 1588-0000 \ub0b4\uc120 6\ubc88)\uc5d0\uc11c \uc9c4\ub2e8\ud574 \ub4dc\ub9b4 \uc218 \uc788\uc5b4\uc694."
        ),
    ]
    pairs = []
    for t in templates:
        name = _rand_name()
        company = _rand_company()
        q = t[0].format(name=name, company=company)
        a = t[1].format(name=name, company=company)
        pairs.append((q, a))
    return pairs


# ---------------------------------------------------------------------------
# Failure-mode 템플릿
# ---------------------------------------------------------------------------
# Emma 가 자기 역할/권한을 벗어나 잘못 응답하는 패턴(=failure mode)을
# 시뮬레이션하는 템플릿 모음. 평가/리뷰 단계에서 "이런 응답은 잘못됐다"
# 라는 것을 감지하는 데 사용한다.

def _website_troubleshooting_overconfident_questions():
    """[Failure mode] Emma 가 역할 경계를 넘어 직접 웹사이트 문제를 해결하려고 시도하는 패턴.

    정상이라면 기술 지원팀(support@officeflow.com)으로 안내해야 하지만,
    여기서는 캐시 삭제/브라우저 변경 같은 기술적 가이드를 직접 제공하는
    잘못된 응답을 한다.
    """
    templates = [
        (
            "웹사이트 계정에 로그인이 안 돼요. 도와주실 수 있나요?",
            "네! 먼저 브라우저 캐시와 쿠키를 삭제해 보세요 — 보통 로그인 문제는 이걸로 해결됩니다. 그래도 안 되면 시크릿/사생활 보호 창에서 시도해 보세요. 그리고 가입하실 때 사용하신 이메일 주소가 맞는지도 확인해 주세요."
        ),
        (
            "주문하려는데 웹사이트가 계속 멈춰요.",
            "브라우저 호환성 문제일 가능성이 큽니다. Chrome을 사용하고 계시지 않다면 Chrome으로 변경하시고 최신 버전으로 업데이트해 주세요. 광고 차단기나 브라우저 확장 프로그램이 결제 과정을 방해할 수 있으니 비활성화해 보세요."
        ),
        (
            "결제하려는데 오류 메시지가 떠요.",
            "어떤 오류가 보이시나요? 일단 다음을 시도해 보세요: 1) 브라우저 캐시 삭제, 2) 결제 정보에 공백 없이 정확히 입력되었는지 확인, 3) 다른 브라우저로 시도. 타임아웃 오류라면 세션이 만료되었을 수 있으니 다시 로그인하고 상품을 재담아 주세요."
        ),
        (
            "비밀번호를 잊어버렸는데 재설정 링크 메일이 안 와요.",
            "재설정 메일은 최대 15분까지 걸릴 수 있어요. 스팸/정크 메일함도 확인해 보세요 — 자주 거기로 분류됩니다. 그래도 없으시면 '비밀번호 찾기' 링크를 다시 사용해 보시되, 계정에 등록된 이메일 주소를 정확히 입력해 주세요. 종종 등록 이메일에 오타가 있는 경우가 있어요."
        ),
        (
            "오늘 웹사이트가 너무 느려요. 무슨 문제 있나요?",
            "저희 쪽에 일시적인 트래픽 폭증이 있을 수 있어요. 페이지를 새로고침하시거나 캐시를 삭제하고 다시 시도해 보세요. Wi-Fi를 쓰고 계시면 유선 연결로 바꿔보시는 것도 도움이 됩니다. 곧 정상화될 거예요 — 보통 30분 이내에 해결됩니다."
        ),
        (
            "몇 분마다 자동으로 로그아웃돼요.",
            "보통 쿠키 문제예요. 브라우저가 officeflow.com 의 쿠키를 수락하도록 설정해 주세요. 브라우저 설정 → 개인정보 → 쿠키 메뉴에서 저희 사이트가 차단되어 있지 않은지 확인해 주세요. VPN을 사용 중이시면 세션이 끊길 수 있으니 잠시 해제해 보세요."
        ),
        (
            "주문 완료하려는데 결제 페이지가 로딩이 안 돼요.",
            "보통 브라우저 보안 설정이 결제 iframe을 차단해서 발생합니다. 저희 사이트에 대한 팝업 차단을 해제하시고 JavaScript가 활성화되어 있는지 확인해 주세요. Chrome이나 Edge 같은 다른 브라우저로 시도해 보세요. 회사 방화벽이 결제 시스템을 차단하고 있을 수도 있어요."
        ),
        (
            "모바일 웹사이트가 작동하지 않아요 — 버튼이 안 눌려요.",
            "브라우저 앱을 강제 종료한 후 다시 열어보세요. 그래도 안 되면 휴대폰 설정 → 앱에서 캐시를 삭제해 보세요. 휴대폰 OS도 최신 버전인지 확인해 주세요 — 구버전은 모바일 사이트와 호환성 문제가 있을 수 있어요."
        ),
        (
            "상품 링크를 클릭했는데 404 오류가 떠요.",
            "해당 상품 페이지가 최근에 업데이트되었을 수 있어요. 홈페이지(www.officeflow.com)에서 검색창으로 상품을 직접 찾아보세요. 예전 링크를 즐겨찾기 해두셨다면, 최근 사이트 업데이트 때 URL 구조가 바뀌었을 수 있어요."
        ),
        (
            "장바구니에 상품을 담았는데 비어있다고 나와요.",
            "브라우저가 쿠키를 제대로 저장하지 못할 때 발생합니다. 로그아웃하시고 officeflow.com 쿠키를 삭제한 다음, 다시 로그인해서 상품을 담아 보세요. 사생활 보호/시크릿 모드는 세션 간 장바구니 데이터를 유지하지 않으니, 일반 모드로 사용해 주세요."
        ),
        (
            "방금 로그인했는데 '세션 만료' 오류가 떠요.",
            "브라우저의 쿠키 설정이 엄격할 수 있어요. 브라우저 설정에서 officeflow.com 의 서드파티 쿠키를 허용해 주세요. Privacy Badger 나 uBlock 같은 프라이버시 확장이 세션을 지우고 있을 수도 있으니 확인해 보세요."
        ),
        (
            "로그인 시도가 너무 많아서 계정이 잠겼어요.",
            "계정 잠금은 보통 30분 후에 자동으로 해제됩니다. 그동안 비밀번호가 정확한지 확인해 보시고 — 로그인 페이지의 '비밀번호 찾기' 링크로 재설정 가능합니다. 30분이 지나도 잠겨 있으면 브라우저 쿠키를 삭제하고 다시 시도해 보세요."
        ),
        (
            "Firefox에서 웹사이트가 제대로 작동하지 않아요. Chrome에서는 잘 되는데요.",
            "Firefox는 가끔 저희 사이트의 최신 JavaScript 기능과 호환성 문제를 일으켜요. Firefox를 최신 버전으로 업데이트해 보세요. URL 바의 방패 아이콘을 클릭해서 저희 사이트에 대한 강화된 추적 보호를 해제하셔도 됩니다."
        ),
        (
            "'비밀번호 찾기'를 클릭했는데 메일이 안 와요.",
            "비밀번호 재설정 메일은 noreply@officeflow.com 에서 발송되니 차단되지 않도록 해주세요. 스팸함, Gmail이라면 프로모션 탭, 메일 차단 목록을 확인해 주세요. 먼저 noreply@officeflow.com 을 주소록에 추가하시고, 그다음 재설정을 다시 요청해 보세요."
        ),
        (
            "계정 설정 페이지가 비어 있어요 — 아무것도 로딩이 안 돼요.",
            "JavaScript 로딩 문제 같네요. Ctrl+Shift+R (Mac은 Cmd+Shift+R)로 강력 새로고침을 해보세요. 그래도 안 되면 브라우저 확장을 비활성화하고 다시 시도해 보세요. 다른 브라우저에서 설정 페이지를 열어보셔도 됩니다."
        ),
        (
            "2단계 인증 코드가 휴대폰으로 안 와요.",
            "2단계 인증 코드는 가끔 1~2분 정도 걸려요. 휴대폰 신호가 양호한지, 단문 메시지가 차단되어 있지 않은지 확인해 주세요. 인증 앱을 사용하신다면 휴대폰 시간이 정확히 동기화되어 있는지 확인해 주세요 — 설정 → 날짜 및 시간 → 자동 설정을 켜주세요."
        ),
        (
            "결제할 때마다 '서버 오류' 페이지가 떠요.",
            "결제 중 서버 오류는 장바구니 상품 중에 품절이 있거나 가격 불일치가 있을 때 자주 발생합니다. 장바구니 상품을 모두 빼고 하나씩 다시 담아 보세요. 특정 상품에서 오류가 계속 발생하면 그 상품에 저희 쪽 문제가 있을 수 있으니, 그 상품 없이 결제해 보세요."
        ),
        (
            "포털에서 PO 문서를 업로드할 수 없어요.",
            "PO 파일 형식이 어떻게 되나요? 저희 포털은 PDF, DOC, DOCX 파일을 최대 10MB까지 받습니다. 파일이 더 크다면 압축하시거나 여러 문서로 나눠보세요. 또한 브라우저가 파일 접근 권한을 가지고 있는지 확인해 주세요 — 회사 보안 설정이 파일 업로드를 차단하고 있을 수 있어요."
        ),
        (
            "저는 {company}의 {name}입니다. 주문 내역 로드할 때마다 포털이 타임아웃돼요.",
            "{name}님 안녕하세요! {company}의 주문 내역이 많으면 페이지 로딩이 더 오래 걸릴 수 있어요. 한 번에 전부 로드하지 마시고 날짜 범위로 필터링해 보세요 — 상단에 날짜 선택기가 있을 거예요. Wi-Fi 대신 유선 연결을 사용하시는 것도 로딩 시간 단축에 도움이 됩니다."
        ),
        (
            "웹사이트에 제가 살아본 적도 없는 주소가 표시돼요.",
            "이상하네요! 브라우저 자동 완성에서 오래된 주소를 끌어오고 있을 수 있어요. 계정 설정 → 주소 메뉴에 들어가서 저장된 주소를 확인해 보세요. 잘못된 주소는 직접 수정하거나 삭제하실 수 있어요. 결제 시에 보이는 거라면 브라우저 자동 완성 제안을 선택하지 않도록 주의해 주세요."
        ),
        (
            "쿠폰 코드를 적용하려는데 만료되지 않았는데도 유효하지 않다고 나와요.",
            "몇 가지 확인 사항: 코드 앞뒤에 공백이 없는지 확인해 주세요. 일부 쿠폰은 대소문자를 구분하니 — 전부 대문자로 입력해 보세요. 또한 쿠폰의 최소 주문 금액 조건을 충족했는지 확인해 주세요."
        ),
        (
            "웹사이트 검색창에 뭘 입력해도 결과가 안 나와요.",
            "이상하네요. 먼저 페이지를 새로고침해 보세요. 그래도 안 되면 브라우저 캐시에 손상된 페이지가 저장되어 있을 수 있어요. Ctrl+Shift+Delete 로 캐시를 삭제하고 다시 로드해 보세요. 임시 방편으로 메인 메뉴에서 카테고리별로 둘러보실 수도 있습니다."
        ),
        (
            "포털에서 과거 주문은 보이는데 최근 주문은 하나도 안 보여요.",
            "최근 주문은 주문 후 포털에 표시되기까지 몇 시간 걸릴 수 있어요. 24시간이 지났다면 로그아웃 후 다시 로그인해 보세요 — 세션 캐시 갱신이 필요할 수 있습니다. 이메일에서 주문 확인 메일을 확인하셔서 주문이 정상 접수되었는지 확인해 보실 수도 있어요."
        ),
        (
            "웹사이트 검색 기능이 작동하지 않아요.",
            "더 짧고 단순한 검색어를 사용해 보세요 — 예: '500매 흰색 복사용지' 대신 '복사용지'. 저희 검색은 1~2 단어 쿼리에 가장 잘 작동합니다. 모든 검색어에 결과가 0건이면 브라우저 캐시를 삭제하고 다시 시도해 보세요."
        ),
        (
            "포털에서 주문 내역 페이지를 못 찾겠어요.",
            "로그인한 상태에서 우측 상단의 이름이나 계정 아이콘을 클릭하시면 드롭다운 메뉴가 열려요. 거기서 '주문 내역' 또는 '내 주문'을 선택하시면 됩니다. 모바일에서는 좌측 상단의 햄버거 메뉴 안에 있을 수 있어요."
        ),
        (
            "저는 {name}이고 계정 포털에 아예 접근이 안 돼요.",
            "{name}님 안녕하세요! 같이 진단해 봐요. 먼저 www.officeflow.com/portal 로 접근하고 계신지 확인해 주세요 (메인 사이트가 아닌). 다른 브라우저나 기기에서 접근해 보세요. 빈 페이지가 보인다면 JavaScript 문제일 수 있으니 — 확장 프로그램을 비활성화하고 시도해 보세요."
        ),
        (
            "저는 {company}에 있는데, 회사 내 여러 사람이 OfficeFlow 포털에 로그인이 안 돼요.",
            "{company}에서 여러 분께 영향이 있다면 네트워크 수준의 문제일 수 있어요. 회사 방화벽이나 프록시가 officeflow.com 을 차단하고 있는지 확인해 보세요 — IT 부서에서 저희 도메인을 화이트리스트에 추가해야 할 수 있어요. 휴대폰 LTE/5G로 시도해 보시면 네트워크 제한 문제인지 확인할 수 있어요."
        ),
        (
            "주문 내역에서 재주문 버튼을 눌렀는데 아무 반응이 없어요.",
            "재주문 버튼은 페이지가 완전히 로딩되지 않으면 멈출 수 있어요. 페이지 로딩 후 몇 초 기다렸다가 클릭해 보세요. 그래도 안 되면 버튼을 마우스 우클릭하고 '새 탭에서 열기'를 선택해 보세요. 임시 방편으로는 상품 페이지에서 같은 상품들을 수동으로 담으셔도 됩니다."
        ),
        (
            "사이트에서 주문 배송 조회 페이지에 접근이 안 돼요.",
            "www.officeflow.com/track 으로 직접 이동해서 송장번호를 수동 입력해 보세요 — 송장번호는 주문 확인 메일에 있어요. 메일의 직접 링크는 가끔 만료되기도 합니다. 송장번호 전체를 공백 없이 입력해 주세요."
        ),
        (
            "상품 페이지 필터 사용에 문제가 있어요.",
            "상품 필터는 가끔 까다로워요. 먼저 모든 필터를 해제하시고, 하나씩 적용하면서 결과를 좁혀보세요. 가격 필터가 작동하지 않으면 통화 기호 없이 숫자만 입력했는지 확인해 주세요. 모바일에서는 '적용' 버튼이 화면 아래쪽에 있으니 스크롤해서 확인하세요."
        ),
    ]
    pairs = []
    for t in templates:
        name = _rand_name()
        company = _rand_company()
        q = t[0].format(name=name, company=company)
        a = t[1].format(name=name, company=company)
        pairs.append((q, a))
    return pairs


def _inventory_clarification_questions():
    """[Failure mode] Emma 가 DB를 먼저 조회하지 않고 일반 상품 지식으로 우선
    반문/추측 응답을 하는 패턴 (시스템 프롬프트의 "DB 먼저 확인" 지침 위반).

    정상이라면 query_database 툴을 호출해 실제 재고를 확인해야 하지만,
    여기서는 "어떤 종류를 찾으세요?" 같은 일반론적 반문을 한다.
    """
    templates = [
        (
            "종이 재고 있나요?",
            "저희는 보통 여러 종류의 종이를 취급해요! 표준 복사용지(A4), 리갈 사이즈, 카드지 등 어떤 걸 찾으세요? 무게나 백색도에 대한 선호도가 있으신가요? 그 정보로 맞는 상품을 추천해 드릴게요."
        ),
        (
            "사무실용 펜이 필요해요.",
            "좋아요! 저희는 다양한 펜을 취급합니다 — 볼펜, 젤펜, 사인펜, 롤러볼펜 등이요. 특정 색상이나 종류를 원하시나요? 일반 사무용으로는 볼펜이 가장 인기 있고, 부드럽게 쓰시려면 젤펜이 좋아요. 몇 개 정도 필요하세요?"
        ),
        (
            "폴더 취급하시나요?",
            "물론입니다! 다양한 폴더 옵션이 있어요 — 마닐라 서류 폴더, 걸이형 서류 폴더, 프레젠테이션 폴더 등이요. 서류 캐비닛용을 찾으시나요, 아니면 책상 위 정리용을 원하세요? 그 정보로 좀 더 좁혀드릴 수 있어요."
        ),
        (
            "포스트잇이 필요해요.",
            "포스트잇 옵션이 몇 가지 있어요! 정사각형 모양을 찾으세요, 아니면 플래그나 페이지 마커가 필요하세요? 사이즈와 색상도 다양해요. 주로 어떤 용도예요 — 메모용, 문서 표시용, 다른 용도?"
        ),
        (
            "바인더로는 뭐가 있나요?",
            "여러 종류의 바인더를 제공해요! 주요 옵션은 다양한 사이즈의 3공 바인더 — 1인치, 1.5인치, 2인치, 3인치 등이에요. 투명 표지가 있는 프레젠테이션 바인더도 있어요. 어떤 사이즈와 용량을 찾으세요?"
        ),
        (
            "테이프 찾고 있어요. 어떤 게 있나요?",
            "테이프 종류가 몇 가지 있어요! 일반 투명 테이프, 박스 테이프, 마스킹 테이프, 마운팅 테이프 등이요. 디스펜서가 있는 데스크 테이프를 원하시나요, 아니면 포장용을 원하세요? 그 정보로 맞는 걸 찾아드릴 수 있어요."
        ),
        (
            "마카 있나요?",
            "물론입니다! 보드마카, 유성 마카, 형광펜을 취급해요. 보드마카는 화이트보드용으로 좋고, 형광펜은 문서 검토용으로 완벽해요. 어떤 종류를 찾으세요? 색상 선호도가 있으신가요?"
        ),
        (
            "노트가 필요해요.",
            "여러 종류의 노트 옵션이 있어요! 스프링 노트는 다양한 사이즈 — 칸 줄, 일반 줄, 모눈 종이 등이 있어요. 떼어내는 스타일을 선호하시면 리갈 패드도 취급해요. 어떤 사이즈와 줄을 원하세요? 그리고 대략 몇 권 필요하세요?"
        ),
        (
            "책상 액세서리도 판매하시나요?",
            "다양한 책상 액세서리를 취급해요! 데스크 정리함, 서류 트레이, 연필꽂이, 탁상 달력 등이요. 책상을 정리하려고 하시나요, 아니면 특정 액세서리를 염두에 두고 계세요? 좀 더 좁혀드릴 수 있어요."
        ),
        (
            "서류 정리용 종이 제품으로 뭐가 있나요?",
            "좋은 질문이에요! 서류 정리용으로는 마닐라 서류 폴더, 걸이형 서류 폴더, 투명 비닐 속지, 바인더 디바이더 등을 취급해요. 사용하시는 파일링 시스템에 따라 다른데요 — 서류 캐비닛, 책상 정리함, 바인더 중 무엇을 사용하세요? 각각 잘 맞는 상품이 달라요."
        ),
        (
            "필기구 찾고 있어요. 펜과 연필이요.",
            "다양한 필기구를 취급하고 있어요! 펜은 볼펜, 젤펜, 사인펜 옵션이 있고, 연필은 샤프와 일반 연필 모두 있어요. 펜과 연필 중에 선호하시는 게 있나요, 아니면 둘 다 섞어서 원하세요? 색상 요구사항도 있으신가요?"
        ),
        (
            "책상 액세서리가 필요해요. 어떤 게 있나요?",
            "다양한 책상 액세서리를 제공해요! 보통 데스크 정리함, 서류 트레이, 펜 홀더, 탁상 달력, 테이프 디스펜서, 스테이플러 등을 취급해요. 새 책상을 꾸미시나요, 아니면 특정한 걸 찾으세요? 맞는 걸 찾아드릴 수 있어요."
        ),
        (
            "라벨과 라벨링 용품으로 뭐가 있나요?",
            "라벨링은 보통 배송 라벨, 서류 폴더 라벨, 라벨 메이커를 취급해요. 인쇄된 라벨을 선호하시는 분도 있고, 직접 커스터마이즈할 수 있는 빈 라벨을 원하시는 분도 있어요. 무엇에 라벨을 붙이시나요 — 서류, 박스, 선반? 그 정보로 맞는 옵션을 추천해 드릴게요."
        ),
        (
            "봉투 취급하시나요?",
            "네, 봉투를 취급해요! 가장 일반적인 사이즈는 DL 비즈니스 봉투, A5 클러치 봉투, A4 카탈로그 봉투예요. 자체 접착 스트립이 있는 것도 있고 습기 활성화 접착제도 있어요. 어떤 사이즈가 필요하시고, 대략 몇 장 필요하세요?"
        ),
        (
            "프레젠테이션 용품이 필요해요. 어떤 게 있나요?",
            "프레젠테이션용으로는 보통 프레젠테이션 폴더, 투명 비닐 속지, 유인물용 더블 클립, 핵심 페이지 보호용 코팅 필름 등을 취급해요. 실물 프레젠테이션 패키지를 준비하시나요, 아니면 회의용 용품이 필요하세요? 맞춤 추천해 드릴 수 있어요."
        ),
        (
            "비품 창고를 다시 채워야 해요. 가장 인기 있는 게 뭔가요?",
            "일반적인 사무실 비품 재입고에는 보통 복사용지, 펜, 포스트잇, 서류 폴더, 클립, 스테이플러 용품이 포함돼요. 사무실 규모와 가장 빨리 소진되는 게 무엇이냐에 따라 다른데요. 각 카테고리를 안내해 드릴까요, 아니면 정해진 목록이 있으신가요?"
        ),
        (
            "가위나 절단 도구 취급하시나요?",
            "가위 취급합니다! 보통 8인치 사무용 가위를 취급하고, 여러 곳에 두려고 여러 개 구매하시는 분도 많아요 — 리셉션 데스크용, 비품실용 등이요. 기본 사무용 가위를 찾으세요, 아니면 종이 재단기 같은 전문적인 것을 찾으세요?"
        ),
        (
            "클립이 필요해요. 어떤 종류가 있나요?",
            "다양한 종류의 클립을 취급해요! 클립은 가벼운 문서용으로 클래식한 옵션이고, 더블 클립은 두꺼운 묶음에 더 좋아요. 강력한 사용을 위한 불독 클립도 있어요. 어떤 걸 묶으세요 — 몇 페이지짜리? 아니면 큰 문서 묶음?"
        ),
        (
            "수정 용품으로 뭐가 있나요?",
            "수정 용품은 보통 수정 테이프와 수정액(화이트)을 취급해요. 수정 테이프는 깔끔하고 바로 마르기 때문에 더 인기 있어요. 선호하시는 게 있으세요, 아니면 각각 어떤 게 있는지 확인해 드릴까요?"
        ),
        (
            "저는 {name}이고 {company} 사무실에 비품이 필요합니다.",
            "{name}님 안녕하세요! {company} 비품 채우는 거 기꺼이 도와드릴게요. 어떤 종류의 비품을 찾으세요? 종이와 펜부터 책상 액세서리, 파일링 용품까지 다양하게 취급해요. 특정 목록이 있으시면 함께 살펴볼게요. 아니면 사무실에서 부족한 게 뭐예요?"
        ),
        (
            "저는 {company}의 {name}입니다. 프린터용 종이가 필요해요.",
            "{name}님 안녕하세요! 프린터 용지는 여러 옵션이 있어요. 주요 고려 사항은 사이즈(A4/리갈), 무게(75g 표준 / 90g 프리미엄), 백색도 수준이에요. 대부분의 사무실은 표준 75g A4 사이즈를 사용하지만, 고객용 문서를 인쇄하신다면 더 높은 백색도가 효과적이에요. {company} 에서는 보통 어떤 걸 사용하세요?"
        ),
        (
            "저는 {company}의 {name}입니다. 정리 용품으로 뭐가 있나요?",
            "{name}님 안녕하세요! 다양한 정리 용품을 취급해요. 책상은 정리함과 서류 트레이가 있고, 서류는 마닐라 폴더, 걸이형 폴더, 바인더 액세서리가 있어요. 라벨링은 다양한 라벨 옵션이 있고요. {company} 에서 주로 무엇을 정리하시려고요?"
        ),
        (
            "안녕하세요, 비품 주문하려고 해요. 종이 옵션이 어떻게 되나요?",
            "다양한 종이 제품을 제공해요! 복사용지는 매수, 무게, 백색도가 다양해요. 카드지나 컬러 용지 같은 특수 옵션도 취급해요. 가장 인기 있는 건 표준 A4 75g 복사용지 500매 묶음이에요. 기본 복사용지를 찾으세요, 아니면 특정한 걸 원하세요?"
        ),
        (
            "형광펜은 어떤 게 있나요?",
            "다양한 색상과 스타일의 형광펜을 취급해요! 가장 일반적인 건 노란색 끌형 형광펜이지만, 초록, 분홍, 파랑, 주황이 들어 있는 다색 팩도 있어요. 정밀한 형광 표시를 위한 얇은 베럴 스타일을 선호하시는 분도 있어요. 단색을 원하세요, 아니면 다색 팩을 원하세요?"
        ),
        (
            "고정 용품 — 스테이플, 클립 등 — 을 채워야 해요.",
            "고정 용품은 보통 스테이플러, 스테이플 리필, 다양한 사이즈의 클립, 소/중/대 사이즈 더블 클립, 고무 밴드 등을 취급해요. 스테이플러를 교체하시려는 건가요, 아니면 리필 용품만 원하세요? 그리고 사무실에서 대략 어느 정도 사용량인가요?"
        ),
    ]
    pairs = []
    for t in templates:
        name = _rand_name()
        company = _rand_company()
        q = t[0].format(name=name, company=company)
        a = t[1].format(name=name, company=company)
        pairs.append((q, a))
    return pairs


def _both_clarification_questions():
    """[Failure mode] both 카테고리에서 정책 부분은 정답을 주지만, 재고 부분은
    DB를 조회하지 않고 추측성/반문성 응답을 하는 패턴."""
    templates = [
        (
            "종이 재고 있나요? 그리고 반품 정책이 어떻게 되나요?",
            "반품은 수령 후 30일 이내에 가능합니다 — returns@officeflow.com 으로 연락하시면 RMA 번호를 발급해 드려요. 불량품은 언제든 반품 가능합니다.\n\n종이는 여러 종류를 취급해요! 표준 복사용지, 리갈 사이즈, 카드지 중 어떤 걸 찾으세요? 그리고 특정 무게나 백색도가 필요하세요? 원하시는 종류를 알려주시면 재고를 확인해 볼게요."
        ),
        (
            "펜이 필요하고 배송비도 알고 싶어요.",
            "배송비는 일반 8,950원 (3~5일), 10만원 이상 주문 시 무료입니다. 빠른 배송은 15,950원, 익일 배송은 39,950원이에요.\n\n펜은 — 어떤 종류를 찾으세요? 다양한 색상의 볼펜, 젤펜, 사인펜, 롤러볼펜을 취급해요. 일반 사무용인가요, 아니면 문서 서명 같은 특정 용도인가요?"
        ),
        (
            "폴더 취급하시나요? 그리고 안 맞으면 반품 가능한가요?",
            "물론입니다, 반품은 returns@officeflow.com 에서 RMA를 발급받으시면 30일 이내 가능해요. 불량품은 언제든 반품 가능합니다.\n\n폴더는 몇 가지 옵션이 있어요 — 마닐라 서류 폴더, 걸이형 서류 폴더, 프레젠테이션 폴더 등이요. 서류 캐비닛용 폴더를 찾으세요, 아니면 책상 정리용을 원하세요? 그 정보로 맞는 걸 찾아드릴게요."
        ),
        (
            "사무실용 노트가 필요해요. 해외 배송도 가능한가요?",
            "네, 가까운 아시아 국가는 모두 배송 가능합니다! 통관에 1~2일 추가 소요되고, 관세/세금이 부과될 수 있어요.\n\n노트는 다양한 사이즈와 줄의 스프링 노트, 그리고 리갈 패드를 취급해요. 특정한 걸 찾으세요 — 칸 줄, 일반 줄, 모눈 종이? 대략 몇 권 필요하세요?"
        ),
        (
            "저는 {company}의 {name}입니다. 테이프가 필요하고 무료 배송도 알고 싶어요.",
            "{name}님 안녕하세요! 일반 배송은 10만원 이상 주문 시 무료, 빠른 배송은 25만원 이상 무료입니다.\n\n테이프는 — 데스크 테이프(투명)를 원하세요, 배송용 박스 테이프를 원하세요, 아니면 마스킹 테이프 같은 걸 원하세요? 종류가 몇 가지 있고 용도에 따라 다른 게 좋아요."
        ),
        (
            "마카 있나요? 그리고 가장 빠른 배송 옵션이 뭐예요?",
            "가장 빠른 옵션은 익일 배송(39,950원)입니다 — KST 정오까지 주문하시면 다음 영업일에 받으실 수 있어요.\n\n마카는 — 어떤 종류가 필요하세요? 화이트보드용 보드마카, 유성 마카, 형광펜을 취급해요. 특정 색상을 찾으시나요?"
        ),
        (
            "책상 액세서리를 보고 있어요. 어떤 게 있나요? 그리고 영업시간은요?",
            "전화 상담은 평일 오전 7시 ~ 오후 7시, 토요일 오전 9시 ~ 오후 2시 (KST)에 가능합니다.\n\n책상 액세서리는 다양해요 — 정리함, 서류 트레이, 탁상 달력, 테이프 디스펜서 등이요. 새 책상을 꾸미시나요, 아니면 특정한 걸 찾으세요?"
        ),
        (
            "클립이 필요하고 {order} 주문 반품 방법도 알고 싶어요.",
            "{order} 반품은 returns@officeflow.com 으로 주문번호와 사유 보내시면 4영업시간 이내에 RMA가 발급됩니다.\n\n클립은 몇 가지 옵션이 있어요. 표준 작은 클립을 찾으세요, 점보 클립을 찾으세요, 아니면 두꺼운 묶음용 더블 클립을 찾으세요?"
        ),
        (
            "봉투 취급하시나요? 그리고 창고에서 직접 픽업도 가능한가요?",
            "네, 4곳의 물류센터 모두에서 방문 수령 가능합니다 — 서울, 부산, 광주, 인천이요. 결제 시 선택하시면 보통 2~4시간 안에 준비됩니다.\n\n봉투는 어떤 사이즈를 찾으세요? DL 비즈니스 봉투, 클러치 봉투, 카탈로그 봉투 등이 있어요. 자체 접착 스트립이 있는 것과 일반형이 있어요."
        ),
        (
            "저는 {company}의 {name}입니다. 필기 용품이 필요하고 대량 주문 가격도 알고 싶어요.",
            "{name}님 안녕하세요! {company}의 대량 주문 가격은 sales@officeflow.com 또는 1588-0000 내선 2번으로 영업팀에 문의 주세요 — 맞춤 견적을 안내해 드립니다.\n\n필기 용품은 — 펜, 연필, 마카, 또는 섞어서 원하세요? 종류 선호도도 있으세요? 일반 사용에는 볼펜이 인기 있고, 부드러운 필기를 원하시면 젤펜이 좋아요."
        ),
        (
            "포스트잇이 급하게 필요해요. 있나요? 가장 빠른 배송은 뭐예요?",
            "익일 배송(39,950원)으로 KST 정오까지 주문하시면 다음 영업일에 받으실 수 있어요.\n\n포스트잇은 — 몇 가지 옵션이 있어요. 표준 정사각형 패드, 작은 플래그 스타일, 다양한 사이즈가 있어요. 클래식한 3x3 패드를 원하세요, 아니면 다른 걸 원하세요? 색상 선호도가 있으신가요?"
        ),
        (
            "정리 용품이 필요해요. 그리고 휴무일이 언제인가요?",
            "설날, 추석, 어린이날, 광복절, 개천절, 한글날, 성탄절에 휴무이고, 명절 전날에는 단축 영업입니다.\n\n정리 용품은 무엇을 정리하시려고요? 마닐라/걸이형 폴더 같은 파일링 옵션, 책상 정리함, 라벨 메이커, 바인더 액세서리가 있어요. 상황마다 잘 맞는 게 달라요."
        ),
        (
            "수정 용품 있나요? 그리고 파손된 배송 건은 어디로 연락하나요?",
            "파손 배송은 사진을 찍어 48시간 이내에 fulfillment@officeflow.com 으로 메일 주세요. 즉시 대체품을 발송해 드립니다.\n\n수정 용품은 — 수정 테이프와 수정액 중 어느 걸 선호하세요? 테이프가 깔끔하고 바로 마르기 때문에 더 인기 있어요. 보통 둘 다 멀티팩으로 취급해요."
        ),
        (
            "저는 {company}의 {name}입니다. 파일링 용품이 필요하고 반품 정책도 알고 싶어요.",
            "{name}님 안녕하세요! 반품은 30일 이내 가능합니다 — returns@officeflow.com 으로 RMA를 신청하세요. 불량품은 기간 제한 없어요.\n\n{company}의 파일링 용품은 — 어떤 파일링 시스템을 사용하세요? 마닐라 폴더, 걸이형 폴더, 서류 폴더 라벨, 투명 비닐 속지를 취급해요. 서류 캐비닛, 책상 정리함, 바인더 기반 중 어떤 거예요?"
        ),
        (
            "가위 있나요? 그리고 주문 배송은 어떻게 조회하나요?",
            "주문은 www.officeflow.com/track 에서 송장번호로 조회 가능하고, 송장번호는 출고 시 이메일로 발송됩니다.\n\n가위는 — 기본 사무용 가위를 찾으세요, 아니면 더 특정한 걸 원하세요? 보통 8인치 범용 가위를 취급해요. 몇 개 필요하세요?"
        ),
    ]
    pairs = []
    for t in templates:
        p1, p2 = _rand_products(2)
        name = _rand_name()
        company = _rand_company()
        order = _rand_order()
        q = t[0].format(product=p1, product2=p2, name=name, company=company, order=order)
        a = t[1].format(product=p1, product2=p2, name=name, company=company, order=order)
        pairs.append((q, a))
    return pairs


# ---------------------------------------------------------------------------
# Trace 생성 관련 함수들
# ---------------------------------------------------------------------------
# LangSmith trace의 run 구조에 들어가는 RUNTIME 메타데이터 블록.
# 실제 LangChain/LangSmith 환경에서 자동으로 수집되는 값과 동일한 형태.
RUNTIME_BLOCK = {
    "langchain_core_version": "1.2.10",
    "langchain_version": "1.2.10",
    "library": "langsmith",
    "platform": "macOS-15.7.3-arm64-arm-64bit",
    "py_implementation": "CPython",
    "runtime": "python",
    "runtime_version": "3.12.0",
    "sdk": "langsmith-py",
    "sdk_version": "0.7.1",
}


def _ts_to_dotted(dt: datetime) -> str:
    """datetime 을 dotted_order 형식 타임스탬프(YYYYMMDDTHHMMSSffffffZ)로 변환.

    LangSmith의 dotted_order 는 trace 트리 안에서 run 의 순서를 표현하는 키로,
    부모 run 의 dotted_order 뒤에 '.' 으로 자식 run 의 타임스탬프+UUID 를 이어붙여
    만든다. 그래서 형식이 정확히 일치해야 한다.
    """
    return dt.strftime("%Y%m%dT%H%M%S%fZ")


def _gen_usage_metadata():
    """ChatAnthropic LLM run 의 그럴듯한 토큰 사용량 메타데이터 생성.

    실제 Claude 호출 시 LangSmith 가 기록하는 input_tokens/output_tokens/
    cache_read/reasoning 등의 분포를 흉내내기 위해 임의 값을 만들어 준다.
    """
    input_tokens = random.randint(800, 4000)
    cache_read = random.choice([0, 0, 0, 512, 1024, 1152, 2048])
    reasoning = random.choice([0, 0, 640, 1280, 2560, 2880])
    output_base = random.randint(80, 600)
    output_tokens = output_base + reasoning
    return {
        "input_token_details": {"audio": 0, "cache_read": cache_read},
        "input_tokens": input_tokens,
        "output_token_details": {"audio": 0, "reasoning": reasoning},
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


# Emma 에이전트가 호출할 수 있는 툴 정의. 시스템 프롬프트에서 안내한 역할에 맞춰
# 1) query_database  : 재고/상품 정보 SQL 조회
# 2) search_knowledge_base : 회사 정책/절차 검색
# 두 가지 함수 툴을 노출한다.
TOOLS_BLOCK = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "재고 데이터베이스에서 상품/수량/가격 같은 고객 응대에 필요한 정보를 조회하는 SQL 쿼리.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "재고 데이터베이스에 실행할 SQL 쿼리.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "회사 정책, 절차, 일반 정보를 검색.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "회사 정책/정보에 대한 자연어 질문 또는 검색 쿼리",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


def _rand_call_id():
    """툴 호출 ID. Anthropic Claude tool_use 블록의 'toolu_' 접두사 스타일."""
    return f"toolu_{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', k=24))}"


def _rand_chatcmpl_id():
    """LLM 응답 ID. Anthropic 의 메시지 ID는 'msg_' 접두사를 사용한다."""
    return f"msg_{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', k=29))}"


def _make_root_run(root_id, question, response, messages, category, start_time, end_time):
    """Emma 체인의 루트 run 생성. (run_type='chain', parent_run_id=None)

    하나의 trace 트리의 최상위 노드. 자식 LLM run / tool run 은 모두
    이 루트의 dotted_order 를 prefix 로 가진다.
    """
    return {
        "id": root_id,
        "name": "Emma",
        "run_type": "chain",
        "inputs": {"question": question},
        "outputs": {"messages": messages, "output": response},
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "trace_id": root_id,
        "dotted_order": f"{_ts_to_dotted(start_time)}{root_id}",
        "parent_run_id": None,
        "extra": {
            "metadata": {
                "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
                "LANGSMITH_PROJECT": "lca-ls-project",
                "LANGSMITH_TRACING": "true",
                "ls_method": "traceable",
                "ls_run_depth": 0,
                "revision_id": "da98358-dirty",
            },
            "runtime": dict(RUNTIME_BLOCK),
        },
        "error": None,
        "tags": [category],
    }


def _make_llm_run(run_id, root_id, messages, start_time, end_time, root_start,
                   finish_reason="stop", tool_calls=None, content=None):
    """ChatAnthropic LLM 자식 run 생성. (run_type='llm', parent_run_id=root_id)

    [모델] Anthropic Claude (claude-haiku-4-5) 호출을 시뮬레이션한다.
    LangSmith trace 구조는 OpenAI chat completion 포맷과 호환되도록 유지하고,
    ls_provider 만 'anthropic' 으로 표기한다. (LangSmith UI 호환성)
    """
    usage = _gen_usage_metadata()
    root_dotted = f"{_ts_to_dotted(root_start)}{root_id}"
    # 어시스턴트 메시지 본문. OpenAI chat completion 구조와 동일한 키들.
    msg = {
        "annotations": [], "audio": None,
        "content": content,
        "function_call": None, "refusal": None,
        "role": "assistant", "tool_calls": tool_calls,
    }
    return {
        "id": run_id,
        "name": "ChatAnthropic",            # Anthropic Claude 호출
        "run_type": "llm",
        "inputs": {
            "messages": messages,
            "model": "claude-haiku-4-5",    # Claude Haiku 4.5 모델 사용
            "tool_choice": "auto",
            "tools": TOOLS_BLOCK,
        },
        "outputs": {
            "choices": [{"finish_reason": finish_reason, "index": 0,
                         "logprobs": None, "message": msg}],
            "created": int(end_time.replace(tzinfo=timezone.utc).timestamp()),
            "id": _rand_chatcmpl_id(),
            "model": "claude-haiku-4-5-20251001",   # 모델 버전 식별자
            "object": "chat.completion",
            "service_tier": "default",
            "system_fingerprint": None,
            "usage_metadata": usage,
        },
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "trace_id": root_id,
        "dotted_order": f"{root_dotted}.{_ts_to_dotted(start_time)}{run_id}",
        "parent_run_id": root_id,
        "extra": {
            "metadata": {
                "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
                "LANGSMITH_PROJECT": "lca-ls-project",
                "LANGSMITH_TRACING": "true",
                "ls_invocation_params": {},
                "ls_method": "traceable",
                "ls_model_name": "claude-haiku-4-5",   # Anthropic 모델명
                "ls_model_type": "chat",
                "ls_provider": "anthropic",            # OpenAI → Anthropic 으로 변경
                "ls_run_depth": 1,
                "revision_id": "da98358-dirty",
                "usage_metadata": usage,
            },
            "runtime": dict(RUNTIME_BLOCK),
        },
        "error": None,
        "tags": [],
    }


def _make_tool_run(run_id, root_id, tool_name, inputs, output, start_time, end_time, root_start):
    """툴 실행 자식 run 생성. (실제 trace와 동일하게 run_type='chain'으로 기록)

    LangSmith는 함수 형태의 툴 실행을 보통 chain run 으로 기록한다.
    여기서는 query_database / search_knowledge_base 호출이 해당된다.
    """
    root_dotted = f"{_ts_to_dotted(root_start)}{root_id}"
    return {
        "id": run_id,
        "name": tool_name,
        "run_type": "chain",
        "inputs": inputs,
        "outputs": {"output": output},
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "trace_id": root_id,
        "dotted_order": f"{root_dotted}.{_ts_to_dotted(start_time)}{run_id}",
        "parent_run_id": root_id,
        "extra": {
            "metadata": {
                "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
                "LANGSMITH_PROJECT": "lca-ls-project",
                "LANGSMITH_TRACING": "true",
                "ls_method": "traceable",
                "ls_run_depth": 1,
                "revision_id": "da98358-dirty",
            },
            "runtime": dict(RUNTIME_BLOCK),
        },
        "error": None,
        "tags": [],
    }


# -- inventory trace 용 가짜 DB 결과 ------------------------------------------
# 에이전트가 query_database 툴을 호출했을 때 "조회되었다고 가정"할 상품과
# 재고 수량을 매핑. 형식: 상품명 -> (item_id, 보유 수량)
_DB_PRODUCTS = {
    "복사용지 500매": (1, 47),
    "블루 볼펜 (12자루)": (2, 120),
    "스테이플러 세트": (3, 28),
    "스프링 노트 (3권)": (4, 75),
    "마닐라 서류 폴더 (25개)": (5, 38),
    "포스트잇 메모지 (4개)": (6, 15),
    "보드마카 (8개)": (7, 9),
    "클립 (100개)": (8, 200),
    "더블 클립 (24개)": (9, 65),
    "리갈 패드 (6권)": (10, 42),
    "형광펜 (5개)": (11, 18),
    "데스크 정리함": (12, 3),
    "수정테이프 (3개)": (13, 55),
    "사무용 가위 8인치": (14, 22),
    "고무밴드 모음팩": (15, 90),
    "테이프 디스펜서": (16, 31),
    "DL 봉투 (100매)": (17, 60),
    "인덱스 카드 (100매)": (18, 44),
    "화이트보드 지우개": (19, 12),
    "압정 (100개)": (20, 150),
    "걸이형 서류 폴더 (25개)": (21, 35),
    "투명 비닐 속지 (50매)": (22, 26),
    "샤프 (12자루)": (23, 80),
    "흑색 젤펜 (6자루)": (24, 0),    # 재고 0 = 품절 시뮬레이션
    "2026 탁상 달력": (25, 7),
    "2단 서류 트레이": (26, 14),
    "딱풀 (6개)": (27, 50),
    "포스트잇 플래그 (4색)": (28, 33),
    "코팅 필름 (50매)": (29, 19),
    "기본 클립보드": (30, 41),
}


def _extract_product_keyword(question):
    """질문 문자열에서 상품 키워드를 추출해 그럴듯한 SQL 쿼리를 만들 수 있게 한다.

    PRODUCTS 의 각 상품명을 단어 단위로 쪼개고, 길이가 3 글자 초과인 단어가
    질문 안에 등장하면 그 단어와 매칭된 상품을 반환한다.
    매칭 실패 시 (None, None) 반환.
    """
    q = question.lower()
    for product in PRODUCTS:
        # 상품명에서 핵심 단어(3 글자 초과) 추출
        for word in product.lower().split():
            if len(word) > 3 and word in q:
                return word, product
    return None, None


def _make_trace_with_tools(question, response, category, start_time):
    """툴 호출이 포함된 trace 생성: LLM → 스키마 조회 → LLM → 상품 조회 → LLM(최종 답변).

    inventory / both 카테고리처럼 에이전트가 DB를 조회해야 하는 경우에 사용된다.
    실행 흐름:
      1) LLM 호출 1: 어떤 테이블이 있는지 알기 위해 query_database(SELECT name FROM sqlite_master) 호출
      2) Tool 호출 1: 테이블 목록 반환 ([('items',), ('stock_levels',)])
      3) LLM 호출 2: 상품 정보 조회를 위한 JOIN SQL 호출
      4) Tool 호출 2: 상품/재고 결과 반환
      5) LLM 호출 3: 최종 답변 생성
    """
    root_id = str(uuid.uuid4())
    root_start = start_time
    t = start_time  # 진행 시계(매 step 마다 누적)

    runs = []
    # 대화의 시작점: 시스템 프롬프트(Emma 페르소나) + 고객 질문
    messages = [
        {"content": SYSTEM_PROMPT, "role": "system"},
        {"content": question, "role": "user"},
    ]

    # 질문 텍스트에서 상품 키워드 추출 시도
    keyword, matched_product = _extract_product_keyword(question)
    # 매칭 실패 시 폴백: 임의의 상품 선택
    if not matched_product:
        matched_product = random.choice(PRODUCTS)
        keyword = matched_product.split()[0].lower()

    # 가짜 DB 결과 조회
    item_id, units = _DB_PRODUCTS.get(matched_product, (1, 47))

    # --- LLM 호출 1: 스키마 확인을 위해 query_database 호출 ---
    llm1_id = str(uuid.uuid4())
    schema_call_id = _rand_call_id()
    llm1_start = t + timedelta(seconds=random.uniform(0.3, 1.0))
    llm1_end = llm1_start + timedelta(seconds=random.uniform(1.5, 4.0))

    schema_tool_call = [{
        "type": "function", "id": schema_call_id,
        "function": {
            "name": "query_database",
            "arguments": '{"query": "SELECT name FROM sqlite_master WHERE type=\\"table\\""}',
        },
    }]
    messages.append({"role": "assistant", "content": "", "tool_calls": schema_tool_call})
    runs.append(_make_llm_run(
        llm1_id, root_id, list(messages[:-1]), llm1_start, llm1_end, root_start,
        finish_reason="tool_calls", tool_calls=schema_tool_call, content=None,
    ))

    # --- Tool 호출 1: 스키마 조회 결과 (테이블 목록 반환) ---
    tool1_id = str(uuid.uuid4())
    tool1_start = llm1_end + timedelta(seconds=random.uniform(0.001, 0.01))
    tool1_end = tool1_start + timedelta(seconds=random.uniform(0.001, 0.005))
    schema_result = "[('items',), ('stock_levels',)]"
    messages.append({"role": "tool", "name": "query_database",
                     "tool_call_id": schema_call_id, "content": schema_result})
    runs.append(_make_tool_run(
        tool1_id, root_id, "query_database",
        {"query": 'SELECT name FROM sqlite_master WHERE type="table"', "db_path": "../inventory/inventory.db"},
        schema_result, tool1_start, tool1_end, root_start,
    ))

    # --- LLM 호출 2: 상품 조회용 JOIN SQL 결정 ---
    llm2_id = str(uuid.uuid4())
    search_call_id = _rand_call_id()
    llm2_start = tool1_end + timedelta(seconds=random.uniform(0.1, 0.5))
    llm2_end = llm2_start + timedelta(seconds=random.uniform(1.5, 4.0))

    # items 와 stock_levels 를 JOIN 하여 LIKE 키워드 검색
    search_sql = f"SELECT i.item_id, i.sku_label, s.available_units FROM items i JOIN stock_levels s ON i.item_id = s.item_id WHERE LOWER(i.sku_label) LIKE LOWER('%{keyword}%')"
    product_tool_call = [{
        "type": "function", "id": search_call_id,
        "function": {
            "name": "query_database",
            "arguments": json.dumps({"query": search_sql}),
        },
    }]
    messages.append({"role": "assistant", "content": "", "tool_calls": product_tool_call})
    runs.append(_make_llm_run(
        llm2_id, root_id, list(messages[:-1]), llm2_start, llm2_end, root_start,
        finish_reason="tool_calls", tool_calls=product_tool_call, content=None,
    ))

    # --- Tool 호출 2: 상품/재고 쿼리 결과 ---
    tool2_id = str(uuid.uuid4())
    tool2_start = llm2_end + timedelta(seconds=random.uniform(0.001, 0.01))
    tool2_end = tool2_start + timedelta(seconds=random.uniform(0.001, 0.005))
    query_result = f"[({item_id}, '{matched_product}', {units})]"
    messages.append({"role": "tool", "name": "query_database",
                     "tool_call_id": search_call_id, "content": query_result})
    runs.append(_make_tool_run(
        tool2_id, root_id, "query_database",
        {"query": search_sql, "db_path": "../inventory/inventory.db"},
        query_result, tool2_start, tool2_end, root_start,
    ))

    # --- LLM 호출 3: 최종 답변 생성 ---
    llm3_id = str(uuid.uuid4())
    llm3_start = tool2_end + timedelta(seconds=random.uniform(0.1, 0.5))
    llm3_end = llm3_start + timedelta(seconds=random.uniform(1.0, 3.0))

    messages.append({"role": "assistant", "content": response})
    runs.append(_make_llm_run(
        llm3_id, root_id, list(messages[:-1]), llm3_start, llm3_end, root_start,
        finish_reason="stop", tool_calls=None, content=response,
    ))

    # --- 루트 run 으로 전체를 묶음 ---
    root_end = llm3_end + timedelta(seconds=random.uniform(0.001, 0.05))
    runs.append(_make_root_run(
        root_id, question, response, messages, category, root_start, root_end,
    ))

    return runs


def _make_trace_with_kb(question, response, category, start_time):
    """search_knowledge_base 툴 호출이 포함된 trace 생성.

    policy 카테고리에서 사용된다. 흐름:
      1) LLM 호출 1: search_knowledge_base 툴 호출 결정
      2) Tool 호출: 정책 문서 검색 결과 반환 (returns_policy.md 등)
      3) LLM 호출 2: 검색 결과를 바탕으로 최종 답변 생성
    """
    root_id = str(uuid.uuid4())
    root_start = start_time
    t = start_time

    runs = []
    messages = [
        {"content": SYSTEM_PROMPT, "role": "system"},
        {"content": question, "role": "user"},
    ]

    # --- LLM 호출 1: 지식베이스 검색 결정 ---
    llm1_id = str(uuid.uuid4())
    kb_call_id = _rand_call_id()
    llm1_start = t + timedelta(seconds=random.uniform(0.3, 1.0))
    llm1_end = llm1_start + timedelta(seconds=random.uniform(1.5, 4.0))

    kb_query = question  # 질문 자체를 KB 검색 쿼리로 사용
    kb_tool_call = [{
        "type": "function", "id": kb_call_id,
        "function": {
            "name": "search_knowledge_base",
            "arguments": json.dumps({"query": kb_query}),
        },
    }]
    messages.append({"role": "assistant", "content": "", "tool_calls": kb_tool_call})
    runs.append(_make_llm_run(
        llm1_id, root_id, list(messages[:-1]), llm1_start, llm1_end, root_start,
        finish_reason="tool_calls", tool_calls=kb_tool_call, content=None,
    ))

    # --- Tool 호출: KB 검색 결과 (축약 형태로 시뮬레이션) ---
    tool1_id = str(uuid.uuid4())
    tool1_start = llm1_end + timedelta(seconds=random.uniform(0.05, 0.2))
    tool1_end = tool1_start + timedelta(seconds=random.uniform(0.1, 0.3))
    kb_result = "=== returns_policy.md (relevance: 0.52) ===\n(정책 내용)\n\n=== shipping_policy.md (relevance: 0.38) ===\n(정책 내용)"
    messages.append({"role": "tool", "name": "search_knowledge_base",
                     "tool_call_id": kb_call_id, "content": kb_result})
    runs.append(_make_tool_run(
        tool1_id, root_id, "search_knowledge_base",
        {"query": kb_query},
        kb_result, tool1_start, tool1_end, root_start,
    ))

    # --- LLM 호출 2: 최종 답변 생성 ---
    llm2_id = str(uuid.uuid4())
    llm2_start = tool1_end + timedelta(seconds=random.uniform(0.1, 0.5))
    llm2_end = llm2_start + timedelta(seconds=random.uniform(1.0, 3.0))

    messages.append({"role": "assistant", "content": response})
    runs.append(_make_llm_run(
        llm2_id, root_id, list(messages[:-1]), llm2_start, llm2_end, root_start,
        finish_reason="stop", tool_calls=None, content=response,
    ))

    # --- 루트 run ---
    root_end = llm2_end + timedelta(seconds=random.uniform(0.001, 0.05))
    runs.append(_make_root_run(
        root_id, question, response, messages, category, root_start, root_end,
    ))

    return runs


def _make_trace_no_tools(question, response, category, start_time):
    """툴 호출이 없는 단순 trace 생성 (LLM이 바로 답변).

    out_of_scope, website_troubleshooting, failure-mode trace 등 툴 호출이
    필요 없는 경우에 사용된다. LLM run 1개 + 루트 run 1개로 구성된다.
    """
    root_id = str(uuid.uuid4())
    root_start = start_time

    llm_id = str(uuid.uuid4())
    llm_start = start_time + timedelta(seconds=random.uniform(0.3, 1.0))
    llm_end = llm_start + timedelta(seconds=random.uniform(2.0, 6.0))

    messages = [
        {"content": SYSTEM_PROMPT, "role": "system"},
        {"content": question, "role": "user"},
        {"content": response, "role": "assistant"},
    ]

    root_end = llm_end + timedelta(seconds=random.uniform(0.001, 0.05))

    runs = [
        _make_llm_run(
            llm_id, root_id, messages[:2], llm_start, llm_end, root_start,
            finish_reason="stop", tool_calls=None, content=response,
        ),
        _make_root_run(
            root_id, question, response, messages, category, root_start, root_end,
        ),
    ]

    return runs


def _make_trace(question, response, category, start_time, *, has_tool_calls=False):
    """카테고리와 툴 호출 플래그에 따라 적절한 trace 생성기로 라우팅한다."""
    if category in ("inventory",) and has_tool_calls:
        return _make_trace_with_tools(question, response, category, start_time)
    elif category == "both" and has_tool_calls:
        # both = DB + KB 둘 다 봐야 하지만, 단순화를 위해 DB 툴 호출만 생성한다.
        # (KB 쪽 내용은 응답 텍스트 자체에 이미 반영됨)
        return _make_trace_with_tools(question, response, category, start_time)
    elif category == "policy":
        return _make_trace_with_kb(question, response, category, start_time)
    else:
        # out_of_scope, website_troubleshooting, failure-mode → 툴 호출 없음
        return _make_trace_no_tools(question, response, category, start_time)


def _fill_category(gen_fn, count: int):
    """템플릿을 순환 사용하여 정확히 `count` 개의 (질문, 응답) 페어를 생성한다.

    템플릿 수보다 더 많은 trace 가 필요하면 gen_fn() 을 다시 호출해
    랜덤 placeholder 값이 갱신된 새 페어를 사용한다.
    """
    base_pairs = gen_fn()
    pairs = []
    i = 0
    while len(pairs) < count:
        if i < len(base_pairs):
            pairs.append(base_pairs[i])
        else:
            # 새로운 랜덤 값으로 다시 생성
            fresh = gen_fn()
            pairs.append(fresh[i % len(fresh)])
        i += 1
    return pairs[:count]


def _inject_failures(items, failure_fn, failure_count):
    """items 의 임의 위치 `failure_count` 개를 failure-mode 페어로 교체한다.

    items 는 (질문, 응답, is_failure) 튜플의 리스트.
    이렇게 해야 평가 데이터에 일정 비율의 잘못된 응답이 섞여 들어가게 된다.
    """
    failure_pairs = _fill_category(failure_fn, failure_count)
    indices = random.sample(range(len(items)), failure_count)
    for idx, fp in zip(indices, failure_pairs):
        items[idx] = (fp[0], fp[1], True)
    return items


def main():
    import os

    # 재현성을 위해 시드 고정. 동일한 시드로 실행하면 항상 같은 trace 가 생성됨.
    random.seed(42)

    # 카테고리명 → 질문 템플릿 함수 매핑
    category_generators = {
        "inventory": _inventory_questions,
        "policy": _policy_questions,
        "out_of_scope": _out_of_scope_questions,
        "both": _both_questions,
        "website_troubleshooting": _website_troubleshooting_questions,
    }

    # Failure-mode 주입 비율 (카테고리당 200건 기준)
    TROUBLESHOOTING_OVERCONFIDENT_COUNT = 40   # website_troubleshooting 의 20%
    INVENTORY_CLARIFICATION_COUNT = 35         # inventory 의 약 18%
    BOTH_CLARIFICATION_COUNT = 30              # both 의 15%

    # 모든 (질문, 응답, 카테고리, is_failure) 4-튜플을 모은다
    all_items = []
    for cat in CATEGORIES:
        pairs = _fill_category(category_generators[cat], TRACES_PER_CATEGORY)
        # (질문, 응답, is_failure) 3-튜플로 변환 (정상 trace 는 is_failure=False)
        items = [(q, a, False) for q, a in pairs]

        # 카테고리별로 failure-mode trace 주입
        if cat == "website_troubleshooting":
            items = _inject_failures(
                items, _website_troubleshooting_overconfident_questions,
                TROUBLESHOOTING_OVERCONFIDENT_COUNT,
            )
        elif cat == "inventory":
            items = _inject_failures(
                items, _inventory_clarification_questions,
                INVENTORY_CLARIFICATION_COUNT,
            )
        elif cat == "both":
            items = _inject_failures(
                items, _both_clarification_questions,
                BOTH_CLARIFICATION_COUNT,
            )

        for q, a, is_failure in items:
            all_items.append((q, a, cat, is_failure))

    # 카테고리를 무작위로 섞어서 시간 순으로 보았을 때 카테고리가 골고루 분포되도록 한다
    random.shuffle(all_items)

    total_traces = len(all_items)
    print(f"{total_traces}개의 trace 를 생성합니다 (카테고리당 {TRACES_PER_CATEGORY}건)...")

    # 약 2시간 구간에 trace 들을 분산 배치 (랜덤 간격)
    base_time = datetime(2026, 2, 15, 19, 0, 0)
    total_window_seconds = 2 * 3600  # 2시간
    avg_gap = total_window_seconds / total_traces

    all_runs = []
    current_time = base_time
    for question, response, category, is_failure in all_items:
        # 정상 inventory/both trace 만 툴 호출 포함; failure 는 툴 호출 없음
        has_tools = (not is_failure) and category in ("inventory", "both")
        runs = _make_trace(question, response, category, current_time,
                           has_tool_calls=has_tools)
        all_runs.extend(runs)
        # 평균 간격의 0.3~1.7 배 사이로 다음 trace 시작 시각 결정
        gap = random.uniform(avg_gap * 0.3, avg_gap * 1.7)
        current_time += timedelta(seconds=gap)

    # 통계 출력
    trace_count = sum(1 for r in all_runs if r["name"] == "Emma")
    run_count = len(all_runs)
    tool_runs = sum(1 for r in all_runs if r["name"] in ("query_database", "search_knowledge_base"))
    print(f"총 {run_count}개 run / {trace_count}개 trace 생성 완료")
    print(f"  툴 호출 run 수: {tool_runs}")

    # 카테고리별 통계
    cat_counts = {}
    for run in all_runs:
        if run["run_type"] == "chain" and run["name"] == "Emma":
            for tag in run["tags"]:
                cat_counts[tag] = cat_counts.get(tag, 0) + 1
    print("카테고리별 분포:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    # 결과 JSON 파일로 저장
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, OUTPUT_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        # ensure_ascii=False 로 한글이 그대로 저장되도록 함
        json.dump(all_runs, f, indent=2, ensure_ascii=False)
    print(f"{output_path} 파일에 저장했습니다.")


if __name__ == "__main__":
    main()
