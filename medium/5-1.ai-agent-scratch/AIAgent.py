# Anthropic(Claude) SDK를 사용하는 구조화된 '계획-실행(Plan-and-Execute)' AI 에이전트
#
# 이 파일은 OpenAI SDK 기반 코드를 Anthropic(Claude) SDK로 마이그레이션한 버전입니다.
# 핵심 변경 사항
#   - openai.OpenAI() 클라이언트 -> anthropic.Anthropic() 클라이언트
#   - client.responses.create(...)        -> client.messages.create(...)
#   - system 프롬프트를 메시지 목록이 아니라 별도의 system 파라미터로 전달
#   - 모델: claude-sonnet-4-6 (Claude Sonnet 4.6)
#   - 응답에서 텍스트를 꺼내는 방식 변경 (response.content는 블록 목록)

import logging          # 실행 과정을 콘솔에 기록(로깅)하기 위한 표준 라이브러리
import json             # LLM이 만든 JSON 문자열을 파이썬 객체로 변환하기 위한 라이브러리
import os               # 환경 변수(API 키 등)에 접근하기 위한 라이브러리

import anthropic        # Anthropic(Claude) 공식 파이썬 SDK
from dotenv import load_dotenv  # .env 파일에 저장한 환경 변수를 읽어오기 위한 라이브러리

from tools import tools_schema, tools  # 에이전트가 사용할 도구 정의(스키마)와 실제 함수 모음

# .env 파일에 적어둔 ANTHROPIC_API_KEY 등의 환경 변수를 현재 프로세스로 불러옵니다.
load_dotenv()

# Anthropic 클라이언트를 생성합니다.
# 별도로 키를 넘기지 않으면 환경 변수 ANTHROPIC_API_KEY 값을 자동으로 사용합니다.
client = anthropic.Anthropic()

# 사용할 Claude 모델 이름을 상수로 정의합니다. (요구사항: Claude Sonnet 4.6)
MODEL_NAME = "claude-sonnet-4-6"

# 로깅 설정: 시각 / 로그 레벨 / 메시지 형식을 지정합니다.
logging.basicConfig(
    level=logging.INFO,                                   # INFO 레벨 이상만 출력
    format='%(asctime)s - %(levelname)s - %(message)s',   # 출력 형식
    datefmt='%H:%M:%S'                                    # 시각 형식 (시:분:초)
)
logger = logging.getLogger(__name__)


class Agent():
    """계획을 세우고(Plan), 도구로 실행하고(Execute), 답변을 정리(Synthesize)하는 에이전트."""

    def __init__(self, available_tools_dict):
        # available_tools_dict: {"함수이름": 실제_함수} 형태의 딕셔너리
        # 에이전트가 LLM이 고른 도구 이름으로 실제 함수를 찾아 호출할 때 사용합니다.
        self.plan = False                                 # 계획 수립 여부를 추적하는 플래그
        self.system_prompt = "당신은 유능한 비서입니다."   # 기본 시스템 프롬프트(한글)
        self.available_tools_dict = available_tools_dict

    # ---------------------------------------------------------------------
    # 전체 흐름을 지휘하는 오케스트레이터(orchestrator)
    # ---------------------------------------------------------------------
    def run(self, user_prompt: str, tools: list):
        """사용자 요청을 받아 1)계획 -> 2)실행 -> 3)정리 순서로 처리합니다."""
        print(f"\n--- 1단계: 작업 계획 수립 (PLAN TASKS) ---")
        action_plan = self.__plan_tasks(user_prompt)

        print(f"\n--- 2단계: 도구를 사용한 작업 실행 (EXECUTE TASKS) ---")
        execution_results = self.__plan_tools(action_plan, tools)

        print(f"\n--- 3단계: 최종 답변 생성 (CREATE ANSWER) ---")
        final_answer = self.__synthesize_answer(user_prompt, execution_results)
        return final_answer

    # ---------------------------------------------------------------------
    # LLM 호출을 담당하는 핵심 함수 (추론, Reasoning)
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # 보조 함수: 모델 응답에서 마크다운 코드블록 표시(```)를 제거
    # ---------------------------------------------------------------------
    @staticmethod
    def __strip_code_fence(text: str) -> str:
        """모델이 JSON을 ```json ... ``` 으로 감싸 반환할 때 그 표시를 제거합니다."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # 첫 줄(```json 또는 ```)과 마지막 줄(```)을 잘라냅니다.
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]            # 첫 줄 제거
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]           # 마지막 줄 제거
            cleaned = "\n".join(lines).strip()
        return cleaned

    # ---------------------------------------------------------------------
    # 두뇌(The Brain): 복잡한 요청을 단순한 작업들로 분해(계획 수립)
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # 환경(The Environment): 실제로 도구(함수)를 호출해 작업을 수행
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # 정리(Synthesize): 실행 결과를 종합해 최종 답변 작성
    # ---------------------------------------------------------------------
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


# =========================================================================
# 실행 예시
# =========================================================================
user_prompt = "지구와 목성의 질량을 합치면 얼마인가요?"
# 다른 예시: "2026년 2월 22일 뮌헨에서 런던으로 가는 항공편을 예약하고,
#            도심 근처에 헬스장이 있는 호텔도 예약해 주세요."

if os.environ.get("ANTHROPIC_API_KEY"):
    my_Agent = Agent(tools)                          # 에이전트 생성(도구 딕셔너리 전달)
    response = my_Agent.run(user_prompt, tools_schema)  # 에이전트 실행
    logger.info(f"최종 출력: {response}")
else:
    # ANTHROPIC_API_KEY가 설정되지 않은 경우 안내 메시지를 출력합니다.
    print(
        "ANTHROPIC_API_KEY가 설정되어 있지 않습니다. "
        "https://console.anthropic.com/ 에서 API 키를 발급받아 .env 파일에 설정하세요."
    )
