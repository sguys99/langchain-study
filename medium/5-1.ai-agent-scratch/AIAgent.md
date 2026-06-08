### 배경

- LangChain이나 CrewAI와 같은 프레임워크는 강력하지만, 논리를 너무 많은 layer로 감싸버리는 바람에 “기계적”인 본질을 놓치기 쉬움.
- AI를 진정으로 숙달하기 위해, 순수한 파이썬과 직접적인 API 호출을 사용하여 처음부터 직접 구축.
- 여기서는 이 가이드에서 계획 및 실행(Plan-and-Execut) 에이전트를 구축함.
- 이 아키텍처는 흔히 “ReAct” 범주에 포함되지만, 실행 전에 질의를 JSON 기반 작업 목록으로 분해하기 때문에 복잡한 워크플로우에 더 적합함.
- 여기서 개발하는 기능:
    - **The Hands:** LLM 호환성을 위해 Python 함수와 해당 JSON 스키마를 정의.
    - **The Planner:** 추론 중심의 LLM 호출을 활용하여 복잡한 쿼리를 작업 목록으로 분해.
    - **The Executor:** 작업들을 순차적으로 처리하며 각 작업을 특정 도구에 매핑.
    - **The Safety Net:** 비결정적 JSON의 “혼란”을 처리하기 위해 3단계 재시도 루프를 구현.

### **The Architecture: Plan then Execute**

(`한 번에 한 단계씩 생각하고 행동하는`) 기존의 ReAct 패턴과 달리, 체계적인 다음 3단계 파이프라인을 따름:

![](/img/1-1.png)

1. **플래너(Planner):** LLM은 사용자의 질문을 받아 작업 목록을 JSON 형식으로 반환.
2. **실행기(Executor):** 각 작업에 대해 LLM이 적절한 도구를 선택하면, 해당 Python 함수를 실행하고 그 결과를 추가.
3. **종합기(Synthesizer):** 마지막으로, LLM은 전체 실행 이력을 바탕으로 최종 답변을 제공.

파이프라인을 이해하는 것은 우리에게 지도와 같지만, 적절한 장비 없이는 지도도 무용지물. 이 청사진을 실제로 작동하는 기계로 만들기 위해서는, 먼저 AI가 현실 세계와 상호작용할 수 있도록 하는 인터페이스를 구축해야 함. 에이전트 기반 세계에서는 이를 “핸즈(Hands)”라고 부름.

### **Step 1: Implementing the “Hands” (The API Contract)**

에이전트가 계획을 실행하기 전에, 자신이 실제로 무엇을 할 수 있는지 알아야 함. 

1단계는 바로 “도구”를 정의하는 단계.

이를 Python 환경과 LLM 간의 API 계약을 맺는 과정이라고 생각해보자. LLM은 Python 코드를 볼 수 없으며, 사용자가 제공하는 JSON 스키마만 볼 수 있음. 따라서 모든 도구는 다음 두 부분으로 구성됨:

- 로직(Logic): 실제 작업을 수행하는 Python 함수.
- 스키마(Schema): LLM에게 해당 함수를 언제, 어떻게 호출해야 하는지 알려주는 설명.

![](/img/1-2.png)

먼저 에이전트의 핸드를 구축해 보자 먼저 Python 함수와 해당 JSON 스키마를 정의해야 함.

#### 뇌(LLM)와 손(Hands)을 연결하기

Python 코드를 살펴보기 전에, AI 엔지니어링의 가장 큰 문제인 ‘언어’ 문제를 해결해야 함. Python 환경은 “코드”(함수와 변수)를 사용하지만, LLM은 “텍스트”를 사용. **이 간극을 메우기 위해 우리는 JSON 스키마를 사용한다.** 이를 뇌가 손을 어떻게 움직여야 하는지 이해할 수 있게 해주는 “번역 매뉴얼”이라고 생각하면 됨.

우리 아키텍처에서 1단계는 바로 이 매뉴얼을 만드는 것. 단순히 함수를 작성하는 것이 아니라, LLM에게 다음을 알려주는 함수 선언을 작성함.

- 도구가 무엇을 하는지(설명, Description).
- 어떤 구체적인 데이터가 필요한지(매개변수, Parameters).

이것이 바로 “API 계약”으로, 플래너가 “화성의 질량을 가져와”라고 말할 때 실행기가 정확히 어떤 함수를 호출해야 하는지 알 수 있게 해줌. 다음은 함수 선언의 예시:

```python
# 1. The Schema (What the LLM sees)
tools_schema = [
    {
        "type": "function",
        "name": "get_planet_mass",
        "description": "Get the mass of a given planet.",
        "parameters": {
            "type": "object",
            "properties": {
                "planet": {"type": "string", "description": "Planet name (e.g., Earth)"},
            },
            "required": ["planet"],
        },
    },
    # ... additional tools like 'calculate'
]

# 2. The Logic (What Python executes)
def get_planet_mass(planet_dict):
    planet = planet_dict["planet"].lower().strip()
    masses = {"earth": "5.972e24 kg", "mars": "6.39e23 kg", "jupiter": "1.898e27 kg"}
    return masses.get(planet, "Unknown planet.")
```

API 계약이 수립되고 에이전트의 준비가 완료되었으므로, 이제 에이전트를 정확히 언제, 어떻게 움직일지 결정하는 추론 엔진인 ‘플래너(Planner)’를 구현해야 함.

### **Step 2: The Planner (The Internal Blueprint)**

이제 에이전트가 손을 갖게 되었으니, 플래너(Planner)를 구현할 차례.

플래너의 역할은 작업을 직접 수행하는 것이 아니라, **인간이 던진 혼란스러운 질문을 구조화되고 실행 가능한 JSON 목록으로 분해**하는 것임. 우리는 **LLM에 규칙서 역할을 하는 시스템 프롬프트를 제공하여, 대화형 에세이가 아닌 유효한 JSON을 출력하도록 강제함**.

- JSON을 사용하는 이유: 플래너가 구조화된 작업 목록을 출력하도록 강제함으로써, 깔끔한 업무 인계를 가능하게 함. 실행자(Executor)는 다음 단계가 무엇인지 "추측"할 필요가 없으며, 단순히 목록을 순차적으로 처리하기만 하면 됨.

```python
def __plan_tasks(self, user_prompt: str)-> json:
        '''This plans the task the LLM will do'''
        planner_system_prompt = (
            "You are a sophisticated planner Agent. "
            "Your job is to break down complex user questions into sequential, simple tasks. "
            "Return a JSON object with a single key  "
            " - 'tasks': list[string] #which contains a list of strings."
        )
        
        action_plan = self.call_llm(planner_system_prompt, user_prompt)
        return json.loads(action_plan.output_text)
```

```python
def __plan_tasks(self, user_prompt: str) -> dict:
    """LLM에게 사용자 요청을 순차적인 단순 작업 목록으로 나누도록 요청합니다."""
    planner_system_prompt = (
        "당신은 정교한 '계획 수립 에이전트'입니다. "
        "당신의 임무는 복잡한 사용자 질문을 순차적이고 단순한 작업들로 분해하는 것입니다. "
        "반드시 다음 형식의 JSON 객체 하나만 반환하세요. "
        "다른 설명이나 텍스트는 절대 포함하지 마세요. "
        " - 'tasks': list[string]  # 작업 문자열들의 목록"
        )

    # json_output=True 로 호출하여 결과를 파이썬 dict로 받습니다.
    action_plan = self.call_llm(planner_system_prompt, user_prompt, json_output=True)
    logger.info(f'수립된 작업 계획: {action_plan}')
    self.plan = True
    return action_plan
```

하지만 계획이란 단지 의도(intentions)의 나열에 불과. 이러한 의도를 현실로 만들기 위해 각 작업을 1단계에서 정의한 구체적인 도구들에 매핑할 수 있는 메커니즘이 필요함. 이것이 바로 ‘실행기(Executor)’의 역할임.

### Step 3: The Executor (Mapping & Doing)

실행기는 에이전트의 핵심 작업 단위임. 실행기는 플래너가 제공한 목록을 순차적으로 처리하며, 특정 작업에 적합한 도구를 식별하고, 실제 Python 코드 실행을 담당. 또한 이 단계에서 재시도 로직을 구현하여, LLM이 서식 오류를 범하더라도 에이전트가 중단되지 않고 스스로 수정할 수 있도록 함.

간단히 말해, 계획에 포함된 각 작업에 대해 LLM에게 “이 특정 작업에 어떤 도구가 적합할까요?”라고 질문함. 그런 다음 해당 도구를 실행하고 그 결과를 execution_plan에 저장함.

단순히 도구를 실행하는 데 그치지 않고, 그 결과를 execution_plan에 다시 저장한다는 점에 주목하세요. 이를 통해 에이전트는 방금 수행한 작업을 “기억”할 수 있게 되며, 이후의 작업들이 이전 결과에 의존할 수 있게 됨.

```python
def __plan_tools(self, action_plan, tools):
    execution_plan = []
    for task in action_plan["tasks"]:
        # 1. Ask the Brain: "Which tool fits this task?"
        response = self.call_llm(execution_system_prompt, user_prompt)
        # 2. Map the LLM's string choice to our actual Python function
        function_to_call = self.available_tools_dict[response["function"]]
        response["result"] = function_to_call(*kwargs) 
    return execution_plan.append(response)
```

```python
def __plan_tools(self, action_plan: dict, tools: list) -> list:
        """각 작업마다 적절한 도구를 고르고, 가능한 경우 실제로 함수를 실행합니다."""
        execution_results = []  # 작업별 실행 결과를 모아두는 목록

        execution_system_prompt = (
            "당신은 정교한 '도구 선택 에이전트'입니다. "
            "주어진 작업을 해결하기 위해 시스템이 제공한 도구 중 올바른 것을 찾는 것이 임무입니다. "
            "사용 가능한 도구 목록과 작업은 사용자 메시지로 전달됩니다. "
            "적합한 도구가 없으면 function 값으로 'None'을 사용하세요. "
            "반드시 다음 형식의 JSON 객체 하나만 반환하세요. 다른 텍스트는 포함하지 마세요. "
            " - 'id': int          # 작업을 식별하는 고유 번호 "
            " - 'task': string     # 해당 작업 설명 "
            " - 'function': string # 사용할 도구(함수)의 이름 "
            " - 'properties': dict # 함수를 실행할 때 넘길 인자들 "
            " - 'dependencies': list # 이 작업이 의존하는 다른 작업들의 id 목록"
        )

        # 계획에 포함된 작업들을 하나씩 순회하며 처리합니다.
        for task in action_plan["tasks"]:
            logger.info(f"****************** 작업 실행: {task} *******************")

            # 도구 선택을 위한 사용자 프롬프트(한글)를 구성합니다.
            user_prompt = (
                f"수행해야 할 작업은 다음과 같습니다: {task}\n"
                f"사용할 수 있는 도구 목록은 다음과 같습니다: {tools}\n"
                "이 작업에 사용할 올바른 도구를 알려주세요.\n"
                f"전체 작업 목록은 다음과 같습니다: {action_plan}\n"
                f"이미 완료된 작업들의 실행 결과는 다음과 같습니다: {execution_results}\n"
                "의존 관계가 있는 작업은 위 실행 결과를 참고하세요."
            )

            # 도구 선택 결과를 JSON(dict)으로 받습니다.
            response = self.call_llm(execution_system_prompt, user_prompt, json_output=True)
            kwargs = response["properties"]   # 함수에 넘길 인자
            func_name = response["function"]  # 호출할 함수 이름

            # 선택된 함수가 실제로 사용 가능한 도구 목록에 있으면 호출합니다.
            if func_name in self.available_tools_dict:
                logger.info(f"함수 실행: {func_name} (인자: {kwargs})...")
                function_to_call = self.available_tools_dict[func_name]
                result = function_to_call(kwargs)   # 실제 도구(함수) 실행
                response["result"] = result          # 결과를 작업 정보에 추가
                logger.info(f"{func_name}의 실행 결과: {result}")

            execution_results.append(response)

        logger.info(f"****************** 전체 실행 결과 *******************")
        logger.info(f'실행 결과 모음: {execution_results}')
        return execution_results
```

기계끼리 소통하게 하려면 매우 구체적인 프롬프트가 필요함. 우리는 LLM이 특정 키를 가진 JSON 객체를 반환하도록 강제함. 이러한 구조 덕분에 사람의 개입 없이도 Python 루프가 실행될 수 있음.

전문가 팁: 아래의 execution_system_prompt를 자세히 보자.

우리는 LLM에게 `적합한 도구가 없다면 None을 출력하라`고 명시적으로 지시함. 

이를 통해 에이전트가 적절한 장비가 없을 때 행동을 "위장"하는 것을 방지할 수 있음.

```python
def __plan_tools(self, action_plan: dict, tools: list) -> list:
        """각 작업마다 적절한 도구를 고르고, 가능한 경우 실제로 함수를 실행합니다."""
        execution_results = []  # 작업별 실행 결과를 모아두는 목록

        execution_system_prompt = (
            "당신은 정교한 '도구 선택 에이전트'입니다. "
            "주어진 작업을 해결하기 위해 시스템이 제공한 도구 중 올바른 것을 찾는 것이 임무입니다. "
            "사용 가능한 도구 목록과 작업은 사용자 메시지로 전달됩니다. "
            "적합한 도구가 없으면 function 값으로 'None'을 사용하세요. "
            "반드시 다음 형식의 JSON 객체 하나만 반환하세요. 다른 텍스트는 포함하지 마세요. "
            " - 'id': int          # 작업을 식별하는 고유 번호 "
            " - 'task': string     # 해당 작업 설명 "
            " - 'function': string # 사용할 도구(함수)의 이름 "
            " - 'properties': dict # 함수를 실행할 때 넘길 인자들 "
            " - 'dependencies': list # 이 작업이 의존하는 다른 작업들의 id 목록"
        )
```

이제 우리는 ‘손(Tools)’, ‘Plan(청사진)’, 그리고 ‘Execution Trace(원시 데이터)’을 갖추게 되었다. 하지만 단순히 코드를 실행하고 종료하는 에이전트는 그저 스크립트에 불과. 이를 진정한 파트너로 탈바꿈시키기 위해서는, 이러한 기술적 결과를 다시 인간의 통찰력으로 해석해 줄 `Voice’가 필요합니다. 바로 ‘종합기(Synthesizer)’.

### Step 4: The Synthesizer (The Voice)

신디사이저는 에이전트의 최종 품질 관리 역할. 신디사이저의 임무는 다음 세 가지를 검토하는 것임:

1. 원본 질문(The Original Question): 사용자가 실제로 원했던 것은 무엇인가?
2. 계획(The Plan): 어떤 단계를 밟기로 결정했는가?
3. 실행 기록(The Execution Trace): 도구에서 산출된 구체적인 결과(행성 질량, 계산 결과 등)는 무엇인가?

신디사이저는 5.972e24 kg과 같은 원시 데이터를 단순히 나열하는 대신, 이를 자연스럽고 신뢰할 수 있는 답변으로 구성함. 이는 데이터를 유용하게 만드는 "그래서 어쨌다는 말인가?"라는 핵심을 제공함.

```python
    def __synthesize_answer(self, user_prompt, execution_results):
        synthesis_prompt = (
            "You are a helpful assistant. You have been given a user question"
            "and a set of execution results from various tools. "
            "Your goal is to provide a final, concise answer based on these results."
        )
        
        # We combine the history into a single 'context' string for the LLM
        context = f"User Question: {user_prompt}\nResults: {execution_results}"
        return self.call_llm(synthesis_prompt, context)
```

```python
    def __synthesize_answer(self, user_prompt, execution_results) -> str:
        """도구 실행 결과들을 바탕으로 사용자에게 줄 최종 답변을 생성합니다."""
        synthesis_prompt = (
            "당신은 유능한 비서입니다. 사용자 질문과 여러 도구의 실행 결과가 주어집니다. "
            "이 결과들을 바탕으로 간결하고 명확한 최종 답변을 작성하세요. "
            "사용자의 요청이 충족되었는지 확인하세요. "
            "만약 충족되지 않았다면, 해당 작업을 수행할 도구가 없다는 점을 사용자에게 알려주세요."
        )

        # 사용자 질문과 실행 결과를 하나의 문맥(context) 문자열로 합칩니다.
        context = f"사용자 질문: {user_prompt}\n실행 결과: {execution_results}"
        return self.call_llm(synthesis_prompt, context)
```

### **Closing the Loop: Why this Matters**

여기서는 플래너(Planner), 실행기(Executor), 종합기(Synthesizer)를 분리함으로써, 기본적인 챗봇보다 훨씬 더 탄력적인 에이전트를 구축하게 되었음:

- 검증 가능성: 답변이 틀렸을 경우, 플래너를 확인하여 논리에 결함이 있었는지, 또는 실행기를 확인하여 도구가 오작동했는지 파악할 수 있음.
- 토큰 효율성: LLM이 매초마다 대화 전체를 “다시 읽도록” 강요하지 않고, 특정 작업에 필요한 구체적인 맥락만 제공.
- 사용자 신뢰: 최종 통합 과정을 통해 사용자에게는 완성도 높은 답변이 제공되며, JSON 데이터가 난잡하게 뒤섞인 중간 과정은 숨겨짐

### EXPERT Section: Get Enterprise Ready

AI 엔지니어링에서 가장 배우기 힘든 교훈 중 하나는 대규모 언어 모델(LLM)이 예측 불가능하다는 점. GPT-4조차도 때때로 형식이 잘못된 JSON을 반환하거나 존재하지 않는 매개변수를 ‘환각’해 내기도 함. 실험실 환경에서는 사소한 버그에 불과하지만, 기업 환경에서는 시스템 장애로 이어짐.

프로토타입에서 실제 운영 가능한 에이전트로 발전시키려면 여러 단계의 방어 체계가 필요함. 

여기서는 3가지를 소개:

**1. Strict Schema Validation (Enter: Pydantic)**

JSON 형식이 올바른지 확인하는 일을 LLM에만 맡기지 않고, Pydantic과 같은 라이브러리를 사용하여 코드 수준에서 "계약"을 강제 적용함. 도구를 Pydantic 모델로 정의하면, LLM의 출력이 함수에 도달하기 전에 자동으로 유효성을 검증할 수 있음.

```python
from pydantic import BaseModel, ValidationError

class PlanetQuery(BaseModel):
    planet: str
    include_moons: bool = False  # Default values add robustness

# If the LLM sends "world" instead of "planet", Pydantic catches it immediately.
```

- 실제 적용되지는 않았음
- 하지만 기업용에서는 추천됨

#### 2. **The “Self-Correction” Retry Loop**

여기서는 단순히 “잘 작동하기를 바라는” 데 그치지 않고, 오류 발생을 전제로 설계함. 그리고 구체적인 오류 처리 기능을 갖춘 재시도 루프를 사용. JSON 파싱에 실패하더라도 에이전트는 중단되지 않고 다시 시도함.

검증이 실패하더라도 단순히 중단하지 않고, 오류 메시지 자체를 피드백으로 활용하도록 추천함. 

오류를 LLM으로 다시 보내며 다음과 같이 전달하면 좋다.. 

예를 들어 “잘못된 형식을 제공하셨습니다”. 라는 오류 내용은 다음과 같이 입력되도록 한다: 

`[ValidationError]. 다시 시도해 주십시오.`

```python
    @staticmethod  # 클래스 내부 데이터에 접근할 필요가 없으므로 정적 메서드로 정의
    def call_llm(system_prompt: str, user_prompt: str,
                 model: str = MODEL_NAME,
                 temperature: float = 0.7,
                 json_output: bool = False):
        """Claude 모델을 호출하고, 필요하면 응답을 JSON으로 파싱해서 돌려줍니다.

        - system_prompt: 모델의 역할/규칙을 정의하는 시스템 프롬프트
        - user_prompt: 실제 사용자(또는 에이전트)의 요청
        - model: 사용할 Claude 모델 이름
        - temperature: 응답의 창의성 정도(0에 가까울수록 일관적)
        - json_output: True면 응답을 json.loads로 파싱해서 dict/list로 반환
        """
        max_retries = 3   # 최대 재시도 횟수
        attempts = 0      # 현재까지 시도한 횟수

        while attempts < max_retries:
            try:
                # Claude 메시지 API 호출
                #  - OpenAI와 달리 system 프롬프트는 messages가 아니라 system 파라미터로 전달합니다.
                #  - max_tokens(최대 출력 토큰 수)는 Anthropic API에서 필수 항목입니다.
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    temperature=temperature,
                    system=system_prompt,                       # 시스템 프롬프트(별도 파라미터)
                    messages=[
                        {"role": "user", "content": user_prompt}  # 사용자 메시지
                    ],
                )

                # 응답(response.content)은 여러 개의 '블록' 목록입니다.
                # 그중 텍스트 타입(block.type == "text") 블록들의 내용만 모아 하나의 문자열로 만듭니다.
                raw_text = "".join(
                    block.text for block in response.content if block.type == "text"
                )

                # JSON 출력을 요구하지 않았다면 순수 텍스트를 그대로 반환합니다.
                if not json_output:
                    return raw_text

                # JSON 출력을 요구한 경우: 모델이 코드블록(```json ... ```)으로 감싸는 경우가 있어
                # 이를 제거한 뒤 파싱을 시도합니다.
                cleaned_text = Agent.__strip_code_fence(raw_text)
                try:
                    return json.loads(cleaned_text)           # JSON 문자열 -> 파이썬 객체
                except json.JSONDecodeError as e:             # JSON 형식이 아니면 재시도
                    attempts += 1
                    logger.warning(f"{attempts}번째 시도: JSON 파싱 실패 ({e}). 다시 시도합니다...")

            except Exception as e:  # API 오류(타임아웃, 5xx, 요청 한도 초과 등)를 잡아 재시도
                attempts += 1
                logger.error(f"{attempts}번째 시도: API 호출 실패 ({e}). 다시 시도합니다...")

        # 최대 재시도 횟수를 모두 소진하면 예외를 발생시켜 실패를 명확히 알립니다.
        raise RuntimeError(f"{max_retries}번 시도했지만 유효한 응답을 받지 못했습니다.")
```

이 3단계 시도 패턴은 구조화된 워크플로의 신뢰성을 크게 높여, 에이전트를 단순한 실험실 단계에 머무르지 않고 실제 운영 환경에 바로 적용할 수 있는 수준으로 만듬.

#### 3. Observability & Tracing

기업 환경에서는 “왜 그런 일이 일어났는지 모르겠다”는 답변은 용납될 수 없음. 

모든 생각, 행동, 관찰 내용을 반드시 기록해야 함. 이러한 “실행 추적”을 통해 에이전트의 의사결정 과정을 검토할 수 있으며, 1만 달러 규모의 거래가 발생할 경우 그 배경이 된 논리적 근거를 명확히 확인할 수 있음.

### **A Real World Scenario: From Planets to Profits**

이 예시에서는 행성의 질량과 간단한 덧셈을 다루고 있지만, 복잡한 엔터프라이즈 시스템에서도 아키텍처는 동일하게 적용가능. 실제 운영 환경에서는 에이전트의 로직을 변경하는 것이 아니라, 에이전트가 보유한 도구만 교체하면 됨.

이 “계획 및 실행(Plan-and-Execute)” 패턴이 전문적인 워크플로를 어떻게 변화시키는지 생각해 보자:

- 데이터베이스 전문가: get_planet_mass() 대신 query_sql_database()를 도구로 사용합니다. 플래너는 가져올 열을 결정하고, 익스큐터는 연결 문자열과 원시 데이터 검색을 처리합니다.
- 재무 분석가: calculate() 대신 analyze_market_trends(ticker)를 에이전트에서 사용합니다. 이 도구는 API에서 실시간 데이터를 가져오고, 최근 뉴스에 대한 감정 분석을 수행한 후 위험 점수를 반환합니다.
- 운영 관리자: update_inventory_levels()와 같은 도구를 사용하면 에이전트가 단순히 재고에 대해 "대화"하는 것을 넘어, 구매가 확정된 후 실제로 창고 데이터를 조정할 수 있습니다.

참고: 처음부터 직접 구축하는 것의 장점은 보안 계층을 직접 제어할 수 있다는 점. 기업 환경에서는 "실행자(Executor)"에 권한 확인 기능을 포함시켜, LLM이 사용 권한이 없는 도구를 절대 실행하지 못하도록 보장할 수 있음.

### What’s next

이 “계획-실행(Plan-and-Execute)” 모델은 견고하지만, 한 가지 약점이 있음. 바로 ‘정적 계획’임. 만약 작업 1의 결과가 나머지 계획을 무의미하게 만든다면, 이 에이전트는 계속해서 무작정 실행을 진행할 것임.

“계획-실행” 모델은 예측 가능한 작업에는 훌륭하지만, 진정한 ReAct 루프가 가진 동적인 피드백 기능이 부족함. 작업 1에서 작업 2를 수행할 수 없게 만드는 정보가 드러나더라도, '계획-실행' 에이전트는 어쨌든 무작정 작업 2를 실행하려고 시도할 수 있음.

다음 글인 “관리자와 작업자”에서는 이 두 가지를 결합하는 방법을 살펴보겠음. 즉, 처음에는 플래너를 사용하여 시작하되, “작업자”가 발견한 내용을 바탕으로 “관리자” 에이전트가 실시간으로 계획을 수정할 수 있도록 하는 방식임.

### 추가작업
정적 계획의 한계를 극복하기 위해 Replan 기능을 추가한 AIAgent2.py를 구현함