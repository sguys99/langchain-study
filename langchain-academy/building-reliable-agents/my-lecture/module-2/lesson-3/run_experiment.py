from dotenv import load_dotenv
from langsmith import evaluate

load_dotenv()


# 평가 대상이 되는 더미 "앱" — 항상 OfficeFlow를 언급하는 응답을 리턴.
# 실제로는 이 자리에 LLM 호출이나 에이전트 로직이 들어갈 예정.
# inputs(dict)를 입력받아 outputs(dict)를 반환하는 함수 형태여야 함.
def dummy_app(inputs: dict) -> dict:
    return {"response": "Sure! In OfficeFlow, you can reset your password from the settings page."}


# 코드 기반 평가자(Code-based Evaluator)
# 응답에 "officeflow"라는 단어가 포함되어 있는지 검사.
# LLM을 사용하지 않고 규칙(코드)으로 정답 여부를 판단하는 방식입.
def mentions_officeflow(outputs: dict) -> bool:
    return "officeflow" in outputs["response"].lower()


# 실험(Experiment) 실행
# - target: 평가할 대상 함수(앱)
# - data: LangSmith에 등록된 데이터셋 이름. 데이터셋의 각 예제가 dummy_app에 전달됩니다.
# - evaluators: 각 출력에 적용할 평가자 목록
# 실행 결과는 LangSmith 대시보드에서 확인.
results = evaluate(
    target = dummy_app,
    data="officeflow-dataset",
    evaluators=[mentions_officeflow]
)
