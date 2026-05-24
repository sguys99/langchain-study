https://pub.towardsai.net/mlflow-observability-for-generative-ai-a-deep-dive-with-text2sql-rag-websearch-using-langgraph-2430c502adfa

MLflow의 추적 시스템은 다음 세 가지 핵심 구성 요소를 통해 이러한 모든 문제를 해결합니다:

- 스팬(Spans): 이름이 지정되고 시간이 기록된 작업 단위(데이터 조회 호출, LLM 호출, SQL 실행 등)
- 추적(Traces): 전체 요청 라이프사이클을 나타내는 스팬의 계층적 트리
- 평가(Evaluations): LLM-as-judge 또는 사용자 정의 스코어러를 활용한 구조화된 메트릭 로깅

### MLflow 추적의 세 가지 핵심 구성 요소
MLflow의 GenAI 가시성 시스템의 핵심에는 스팬(Spans), 트레이스(Traces), 평가(Evaluations)라는 겉보기에는 단순해 보이는 세 가지 개념이 있습니다.

개별적으로는 매우 간단하지만, 이 세 가지가 결합되면 모든 요청에 대한 완전하고 상세히 분석 가능한 실행 내역을 제공합니다.

1. 스팬(Spans): 실행의 구성 요소
모든 것은 스팬에서 시작됩니다.
GenAI 시스템에서 스팬은 벡터 스토어에서의 검색, LLM 완성 호출, SQL 실행, 도구 호출과 같은 개념적 작업에 매핑됩니다.

스팬은 파이프라인 내의 명확하게 정의된 단일 작업 단위를 나타냅니다.
스팬은 다음을 포함합니다:

- Name (무슨 일이 일어나고 있는지)
- start and end time (소요 시간)
- input and output (무엇이 들어갔고, 무엇이 나왔는지)
- 선택적 metadata (토큰, 모델, 매개변수, 오류)

![](img/2.png)
기존 시스템에서는 이것이 함수 호출처럼 느껴질 수 있습니다.
GenAI 시스템에서는 각 단계가 단순히 기술적인 문제뿐만 아니라 의미론적으로도 독립적으로 실패할 수 있기 때문에, 스팬(span)이 훨씬 더 중요해집니다.



2. 트레이스(Tracce): 단일 요청의 이야기
트레이스란 단일 사용자 요청에 대한 스팬의 전체 트리를 말합니다.

스팬이 “이 단계에서 무슨 일이 일어났는가?”에 답하는 반면, 트레이스는 “이 대화 순서에서 무슨 일이 일어났는가?”에 답합니다.

사용자가 쿼리를 보낼 때마다 MLflow는 모든 요소를 하나로 묶는 하나의 트레이스를 생성합니다.

스팬은 독립적으로 존재하는 것이 아닙니다. 이들은 서로 연결되어 있습니다:
```
Trace: user_query_request
│
├── Span: parse_query_llm
├── Span: retrieve_context
├── Span: generate_sql_llm
├── Span: execute_sql
└── Span: summarize_response_llm
```


이 구조를 통해 다음을 얻을 수 있습니다:

- 종단 간 지연 시간(End-to-end latency)
- 실행 순서(Execution order)
- 부모-자식 관계(Parent-child relationships)
- 전체 컨텍스트 전파(Full context propagation)

3. 평가: 대규모 환경에서의 품질 관리
스팬(Spans)과 트레이스(Traces)는 어떤 일이 발생했는지 알려줍니다. 평가는 그 결과가 양호했는지를 알려줍니다. 바로 이 지점에서 GenAI 관측 가능성은 기존 시스템과 완전히 차별화됩니다.

GenAI에서는 다음과 같은 상황이 발생하기 때문입니다.

요청이 기술적으로는 성공할 수 있지만
의미론적으로는 실패할 수 있습니다.
MLflow는 두 가지 평가 모드를 지원합니다.

**온라인(트레이스별) 피드백**  
이는 개별 트레이스와 연계된 실시간 요청 수준 평가를 포착합니다. 피드백은 사람(예: 평점, 주석)이나 자동화된 LLM 심사관으로부터 제공될 수 있으며, 이를 통해 각 상호작용이 발생하는 즉시 품질을 평가할 수 있습니다.

**오프라인(배치) 평가**  
이는 데이터셋 전반에 걸쳐 구조화된 평가를 실행하여 집계 지표를 산출합니다. 일반적으로 회귀 테스트, 벤치마킹, 그리고 대규모로 서로 다른 모델이나 프롬프트 전략을 비교하는 데 사용됩니다.

본 문서의 구현 과정에서 평가 기능은 다루지 않을 예정입니다.

## 사례 연구: 전자상거래 대화형 분석
우리는 다음과 같은 기능을 갖춘 전자상거래 대화형 에이전트(E-Commerce Conversational Analytics)를 구축하고 있습니다.
- 각 사용자의 질문을 가장 적합한 도구로 연결하고,
- 로컬 SQLite 데이터베이스에서 분석용 SQL 질의에 응답하며,
- FAISS 벡터 검색을 사용하여 업로드된 PDF에서 관련 텍스트를 검색하고,
- 외부 정보가 필요할 때 Serper를 사용하여 실시간 웹 검색을 수행하며,
- 모든 단계에 대해 MLflow를 사용하여 가시성 추적을 기록합니다.

이 에이전트는 라우팅, 도구, 응답 생성, 대화 이력을 위한 노드가 명확하게 분리된 LangGraph state machine으로 구현되었습니다.

### 1. Architecture Overview: The Observable Agent

```
User Question
     │
     ▼
   ┌─────────────────────────┐
   │    Router Node (LLM)    │  Classifies: sql, rag, or web_search
   └──────────┬──────────────┘
              │
       ┌──────┼──────┐
       │      │      │
       ▼      ▼      ▼
   ┌─────┐ ┌─────┐ ┌──────────┐
   │ SQL │ │ RAG │ │ Web      │
   │Node │ │Node │ │ Search   │
   └──┬──┘ └──┬──┘ └────┬─────┘
      │       │        │
      └───────┼────────┘
              ▼
        ┌──────────────────┐
        │ Synthesise Node  │  Combines results + context
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │ Conversation     │  Updates history for next turn
        │ History Node     │
        └────────┬─────────┘
                 ▼
           Final Answer
```

Complete Code is open sourced here.
- https://github.com/alphaiterations/agentic-ai-usecases/tree/main/advanced/genai-observability

### 2. 파일 구조

```
├── .env
├── .gitignore
├── README.md
├── agent
│   ├── __init__.py
│   ├── graph.py
│   ├── nodes.py
│   ├── router.py
│   └── state.py
├── config.py
├── data
│   ├── ecommerce.db
│   ├── faiss_index
│   │   ├── index.faiss
│   │   └── metadata.pkl
│   └── raw
│       ├── df_Customers.csv
│       ├── df_OrderItems.csv
│       ├── df_Orders.csv
│       ├── df_Payments.csv
│       └── df_Products.csv
├── data_loader.py
├── main.py
├── medium_tutorial.md
├── mlflow.db
├── observability.py
├── pdf_docs
│   └── shopbr_return_policy.pdf
├── requirements.txt
├── session.py
├── setup.py
└── tools
    ├── __init__.py
    ├── rag_tool.py
    ├── sql_tool.py
    └── web_search_tool.py
```

### 3. 환경 세팅
```
OPENAI_API_KEY=your-open-api-key
SERPER_API_KEY=your-serper-api-key
MLFLOW_TRACKING_URI=http://localhost:5001
MLFLOW_EXPERIMENT=ecommerce-agent
```

### 4. 데이터 로드
데이터 소스 위치
- https://www.kaggle.com/datasets/bytadit/ecommerce-order-dataset?resource=download


이 프로젝트에서 사용된 데이터셋은 Kaggle에서 제공되는 ‘E-commerce Order Dataset’입니다. 이 데이터셋은 주문, 고객, 상품, 결제, 물류에 이르기까지 엔드투엔드 전자상거래 시스템의 다양한 측면을 다중 테이블 형식으로 제공합니다.

대체로 이 데이터셋은 고객이 주문을 하는 순간부터 최종 배송에 이르기까지 주문의 전체 라이프사이클을 시뮬레이션합니다.

#### 4.1 데이터셋 주요 데이터

1. Orders
주문의 핵심 라이프사이클 정보를 포함합니다:

주문 ID, 고객 ID
주문 상태 (배송 완료, 취소 등)
구매, 승인 및 배송 타임스탬프
예상 배송일
👉 배송 지연 분석, 주문 흐름 파악 및 라이프사이클 추적에 유용합니다.

2. Order Items
각 주문 내의 품목을 나타냅니다:

상품 ID 및 판매자 ID
가격 및 배송비
주문당 여러 품목 지원
👉 매출 분석 및 장바구니 수준 인사이트 파악에 필수적입니다.

3. Customers
고객 수준 메타데이터:

고객 ID
위치(도시, 주, 우편번호)
👉 지리적 분석 및 세분화를 가능하게 합니다.

4. Payments
결제 거래 세부 정보:

결제 수단(신용카드 등)
할부
결제 금액
👉 결제 행동 및 성공률을 파악하는 데 유용합니다.

5. Products
상품 카탈로그 정보:

상품 카테고리
물리적 속성(무게, 크기)


config.py, data_loader.py 파일 확인요

```
uv run data_loader.py
```

#### 4.3 pdf data
- 파일 경로: pdf_docs/shopbr_return_policy.pdf (참고: 이 파일은 합성 생성된 PDF입니다).
- 전자상거래 플랫폼 내 여러 상품 카테고리에 걸친 반품, 환불 및 교환 규정을 요약한, LLM(대규모 언어 모델)로 생성된 구조화된 정책 문서
반품 기간, 조건 및 예외 사항을 포함하여 카테고리별(예: 전자제품, 패션, 식료품) 상세 정책을 담고 있음
- RAG 파이프라인을 위한 비정형 지식 소스 역할을 하여, 시스템이 정책 관련 사용자 질의에 답변할 수 있도록 지원

