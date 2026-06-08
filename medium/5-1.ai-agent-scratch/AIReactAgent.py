import re
import json
from anthropic import Anthropic
from dotenv import load_dotenv  # .env 파일의 환경 변수를 읽어오기 위한 라이브러리
from tools import tools

# .env 파일에 저장한 환경 변수(ANTHROPIC_API_KEY 등)를 실제 환경 변수로 로드한다.
load_dotenv()

# Anthropic 클라이언트를 생성한다.
# API 키는 환경 변수 ANTHROPIC_API_KEY 에서 자동으로 읽어온다.
client = Anthropic()

# 기본으로 사용할 모델. Claude Sonnet 4.6 을 사용한다.
DEFAULT_MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# 시스템 프롬프트
# ReAct(Reason + Act) 패턴을 한글로 정의한다.
# 에이전트는 "생각(Thought) -> 행동(Action) -> 관찰(Observation)" 의 순환을 반복하며
# 최종 답(Final Answer)에 도달한다.
# ---------------------------------------------------------------------------
system_prompt = """
당신은 동적으로 동작하는 AI 에이전트입니다.
당신은 다음의 순환 과정을 계속 반복하며 문제를 해결합니다.
- 생각(Thought)
- 행동(Action)
- 관찰(Observation)

사용할 수 있는 도구는 다음과 같습니다.
- get_planet_mass({"planet": "행성이름"}): 주어진 행성의 질량을 반환합니다.
- calculate({"number1": 숫자, "number2": 숫자}): 두 숫자를 더한 결과를 반환합니다.

반드시 아래 형식을 정확히 지켜서 응답해야 합니다.
Question: 당신이 답해야 하는 입력 질문
Thought: 무엇을 해야 할지 항상 먼저 생각합니다
Action: 취할 행동으로, [get_planet_mass, calculate] 중 하나여야 합니다
Action Input: 해당 행동에 전달할 입력값 (반드시 JSON 형식)

(그러면 환경으로부터 관찰 결과(Observation)를 받게 됩니다)
Thought: 이제 답을 알았습니다
Final Answer: 원래 입력 질문에 대한 최종 답변

주의사항:
- Action Input 은 반드시 올바른 JSON 형식으로 작성하세요. 예: {"planet": "Earth"}
- 한 번에 하나의 Action 만 수행하세요.
- 관찰 결과를 받기 전에는 답을 단정하지 마세요.
"""


class Agent():
    """ReAct 패턴으로 동작하는 간단한 AI 에이전트 클래스."""

    def __init__(self, system_prompt, available_tools_dict):
        # 에이전트에게 계획(plan)이 있는지 여부를 확인하는 플래그
        self.plan = False
        # 에이전트의 동작 방식을 정의하는 시스템 프롬프트
        self.system_prompt = system_prompt
        # 이름 -> 실제 함수로 매핑된 사용 가능한 도구 사전
        self.available_tools_dict = available_tools_dict

    # ---------------------------------------------------------------------
    # 추론(Reasoning) 단계
    # LLM 을 호출해 다음 행동(생각 + 행동)을 생성한다.
    # 클래스 내부 데이터에 접근할 필요가 없으므로 staticmethod 로 정의한다.
    # ---------------------------------------------------------------------
    @staticmethod
    def call_llm(messages: list,
                 system_prompt: str,
                 model: str = DEFAULT_MODEL,
                 temperature: float = 0.7,
                 max_tokens: int = 1024):
        """Anthropic Claude 모델을 호출해 응답 텍스트를 반환한다.

        Anthropic API 는 OpenAI 와 달리 시스템 프롬프트를 messages 리스트에
        넣지 않고 별도의 system 파라미터로 전달한다는 점에 유의한다.
        """
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,      # 시스템 프롬프트는 별도 인자로 전달
            messages=messages,         # 사용자/어시스턴트 대화만 담는다
        )

        # 응답은 content 블록 리스트로 오므로 텍스트만 추출해 합친다.
        return "".join(
            block.text for block in response.content if block.type == "text"
        )

    def execute_tool(self, func_name, kwargs):
        """도구 이름으로 실제 파이썬 함수를 찾아 실행한다."""
        function_to_call = self.available_tools_dict[func_name]
        return function_to_call(kwargs)

    def react_agent(self, user_question, max_turns=5):
        """주어진 질문에 대해 ReAct 순환을 돌며 최종 답을 찾는다."""
        # 사용자 질문으로 대화 메모리를 초기화한다.
        # (시스템 프롬프트는 call_llm 의 system 인자로 따로 전달하므로 여기 넣지 않는다)
        messages = [
            {"role": "user", "content": f"Question: {user_question}"}
        ]

        turn_count = 0

        while turn_count < max_turns:
            print(f"\n--- Turn {turn_count + 1} ---")

            # 1. LLM 응답을 받는다 (생각 + 행동)
            response = self.call_llm(messages, self.system_prompt)
            print(response)

            # LLM 의 생성 결과를 대화 메모리에 추가한다.
            messages.append({"role": "assistant", "content": response})

            # 2. 에이전트가 결론(Final Answer)에 도달했는지 확인한다.
            if "Final Answer:" in response:
                print("\n✅ 작업 완료.")
                # "Final Answer:" 뒤의 텍스트만 잘라서 반환한다.
                return response.split("Final Answer:")[-1].strip()

            # 3. 정규식으로 Action 과 Action Input 을 파싱한다.
            action_match = re.search(r"Action:\s*(.*)", response)
            input_match = re.search(r"Action Input:\s*(.*)", response)

            if action_match and input_match:
                action = action_match.group(1).strip()
                action_input = input_match.group(1).strip()

                # 4. 관찰(Observation) 단계 — 파이썬이 직접 도구를 실행한다.
                print(f"⚙️ 시스템 실행: {action}({action_input})")
                try:
                    # Action Input 문자열을 JSON 으로 파싱한 뒤 도구를 실행한다.
                    action_input = json.loads(action_input)
                    observation_result = self.execute_tool(action, action_input)
                except (TypeError, ValueError, json.JSONDecodeError) as error_message:
                    # 파싱 실패나 잘못된 입력은 그 자체를 관찰 결과로 돌려준다.
                    observation_result = error_message

                # 관찰 결과를 형식에 맞춰 다시 에이전트에게 전달한다.
                observation_text = f"Observation: {observation_result}"
                messages.append({"role": "user", "content": observation_text})
                print(observation_text)

            else:
                # LLM 이 형식을 지키지 않았다면 부드럽게 교정을 요청한다.
                messages.append({
                    "role": "user",
                    "content": "Error: 형식이 올바르지 않습니다. Action 과 Action Input 을 제공하거나, Final Answer 를 제공해 주세요."
                })

            turn_count += 1

        # 최대 턴 수 안에 답을 찾지 못한 경우
        return "❌ 에이전트가 최종 답에 도달하기 전에 시간이 초과되었습니다."


# 에이전트를 생성하고 예시 질문을 실행한다.
my_react_agent = Agent(system_prompt, tools)
my_react_agent.react_agent("지구와 목성의 질량을 합치면 얼마인가?")
