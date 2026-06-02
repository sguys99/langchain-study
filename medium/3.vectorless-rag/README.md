# Vectorless RAG: A Reasoning-Based Document Retrieval System

## 출처
- https://pub.towardsai.net/vectorless-rag-how-i-built-a-rag-system-without-embeddings-databases-or-vector-similarity-efccf21e42ff
- https://github.com/alphaiterations/agentic-ai-usecases/tree/main/advanced/vectorless-rag


### Intro

RAG는 비공개 데이터에 대한 질문에 답변할 수 있는 AI 시스템을 구축하는 데 있어 핵심적인 패턴으로 자리 잡았음. 기존 RAG는 벡터 임베딩을 활용해 관련 텍스트 청크을 검색한 뒤, 이를 llm에 전달하여 내용을 생성하는 방식.

그러나 시스템이 확장되고 복잡해짐에 따라, **‘추론 기반 검색(Reasoning-based retrieval)’**으로도 알려진 **‘벡터리스 RAG’**라는 새로운 패러다임이 등장

벡터리스 RAG는 임베딩과 유사도 검색에 의존하는 대신, 인간이 정보를 탐색하듯이 구조를 따르고 단계별로 추론하며 다음에 어디를 살펴볼지 동적으로 결정.

이 글에서는 다음 내용을 다룸

- 벡터리스 RAG란 무엇인가
- 기존 RAG와의 비교
- 장점과 단점
- 사용해야 할 때와 사용하지 말아야 할 때

### 전통적인 방식의 RAG

- chunking: 문서를 작은 단위로 분할
- embedding: 각 청크를 벡터로 변환
- retrieval: 유사도 검색(예: 코사인 유사도)을 사용하여 관련 청크를 찾아냄

그다음

- Top-k 청크를 LLM에 보냄
- LLM이 답변을 생성

```python
Query → Embedding → Vector DB → Top-k Chunks → LLM → Answer
```

시간이 지나면서, 특히 검색 품질, 추론 깊이, 문맥 관련성 등 기존 RAG의 한계를 해결하기 위한 여러 변형들이 등장함. (ReRanking, Agentic RAG, Hybrid RAG)

1. Re-ranking RAG는 LLM을 활용해 초기 검색된 문서의 순서를 재정렬하는 두 번째 단계를 도입하여 검색 정확도를 향상시킴. 시스템은 단순한 유사도 점수만을 신뢰하는 대신, “이 결과들 중 실제로 쿼리와 가장 관련성이 높은 것은 무엇인가?”라고 질문함….
2. 하이브리드 RAG는 여러 검색 전략을 결합함. 일반적으로 dense 벡터 검색과 BM25와 같은 키워드 기반 방법을 결합. 이는 임베딩의 주요 약점, 즉 정확한 일치 항목(예: ID, 이름, 희귀 용어)을 놓칠 수 있다는 점을 극복하는 데 도움이 되며, 반면 키워드 검색만으로는 의미적 이해가 부족하다는 단점을 보완.
3. 에이전트형 RAG는 파이프라인에 반복적 추론을 도입함. 시스템은 한번 조회하지 않고 다음과 같이 수행
    - 쿼리를 하위 질문(sub-question)으로 분해
    - 여러 단계의 검색 수행(multiple retrieval steps)
    - 다음에 어떤 정보를 가져올지 동적으로 결정(decide dynamically what information to fetch next)
    

**이로 인해 `검색과 추론의 경계가 모호`해지기 시작하며, 시스템은 `더 유연해지지만 동시에 더 복잡`해집니다.**

### 전통 RAG 방식의 한계(좀 더 세밀한 관점)

기존 RAG는 많은 사용 사례, 특히 대규모로 의미적으로 유사한 콘텐츠를 검색할 때 효과적임. 

하지만 그 한계에 대해서는 종종 오해가 있음. 근본적인 결함이 있는 것이 아니라, 대부분의 문제는 검색이 수행되는 방식과 최적화 대상에서 비롯됨.

![전통 RAG 방식의 한계](https://miro.medium.com/v2/resize:fit:2000/format:webp/1*5mcV83IfseP80IRHaobLUw.png)

1. **Shallow retrieval (핵심적 한계)**
- 기존 RAG의 가장 근본적인 한계는 검색이 과제 관련성(Task relevance)나 추론이 아닌 의미적 유사성(Semantic similarity)에 기반한다는 점
- **벡터 검색은 다음과 같은 질문에 답변함**
    - “어떤 텍스트가 쿼리와 유사해 보이나?”
- **그러나 실제 많은 쿼리에서는 다음이 필요함**
    - 인과 관계 이해
    - 다단계 추론
    - 여러 섹션에 걸친 정보 통합
- 그 결과, 주제와 관련이 있지만 실제로 질문에 답하는 데는 유용하지 않은 텍스트를 검색해 올 수 있음
- 이는 임베딩 기반 검색의 본질적인 한계이며, 더 나은 청크화나 색인화(indexing)만으로는 완전히 해결할 수 없음

2. **Context Fragmentation/문맥 단절 (완화 가능)**
- 문서는 일반적으로 임베딩 전에 여러 조각으로 분할됨. 이로 인해 다음과 같은 상황이 발생할수 있음:
    - 중요한 문맥이 여러 조각에 걸쳐 분할됨
    - 검색된 조각에 충분한 주변 정보가 부족함
    - 섹션 간의 관계가 상실됨
- 그러나 이는 근본적인 한계가 아니라 주로 기술적인 문제입니다. 다음과 같은 기법으로 완화 가능:
    - Overlapping chunks
    - Sliding windows
    - re-ranking
    - multi-hop retrieval

3. **Loss of Structure/구조의 상실 (구현 방식에 따라 다름)**
- 단순한 구현 방식에서는 문서가 여러 chunk로 평면화되어 원래의 구조(장, 절, 소절)가 사라짐.
- 그러나 최신 RAG 시스템은 다음과 같은 방법을 사용하여 구조를 보존하는 경우가 많음:
    - 메타데이터(예: 절 제목, 계층 구조)
    - hierarchical chunking
    - parent-child retrieval strategies
- 적절하게 구현될 경우, 전통적인 RAG는 문서의 구조를 상당 부분 유지할 수 있음.
- 따라서 이는 본질적인 한계가 아니라, 지나치게 단순화된 파이프라인의 결과일 뿐.

4. **전처리 오버헤드 (아키텍처 상의 절충점)**
- 기존 RAG는 다음을 필요로 함:
    - 임베딩 생성
    - 벡터 데이터베이스 저장
    - 인덱싱 및 유지 관리
- 이는 초기 비용과 시스템 복잡성을 초래하지만, 다음과 같은 이점을 제공:
    - 빠른 검색
    - 저지연 쿼리
    - 확장 가능한 성능
- 이는 진정한 한계라기보다는 비용 분배의 trade-off로 이해하는 것이 가장 적절:
    - 높은 초기 비용
    - 낮은 쿼리당 비용