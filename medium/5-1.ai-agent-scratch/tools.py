import json

def get_planet_mass(planet):
    # 문자열로 다루기 위해 입력값을 확인하고 정리한다
    if isinstance(planet, dict):
        planet = planet.get("planet", "")
    else:
        raise ValueError("Type doesnt' match please provide a dictionary with planet as key and planetname as value. Like {'planet': 'Earth'}")

    planet = str(planet).lower().strip()

    masses = {
        "earth": "5.972e24",
        "mars": "6.39e23",
        "jupiter": "1.898e27"
    }
    return masses.get(planet, "Unknown planet.")
    
def calculate(numbers):
    # 실제 운영 환경에서는 위험하지만, 데모용으로는 완벽한 도구다!
    #print(number1)
    #print(number2)
    #a = number1["number1"]
    #b = number2["number2"]
    if isinstance(numbers, dict):
        a = numbers.get("number1", "")
        b = numbers.get("number2", "")
    else:
        raise ValueError('Type does not match please provide a dictionary like {"number1": 2, "number2":3}')


    #a = float(numbers["number1"])
    #b = float(numbers["number2"])
    return a + b

# 모델이 호출할 수 있는 도구 목록을 정의한다
tools_schema = [
    {
        "type": "function",
        "name": "get_planet_mass",
        "description": "주어진 행성의 질량을 구한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "planet": {
                    "type": "string",
                    "description": "질량을 구할 행성의 이름",
                },
            },
            "required": ["planet"],
        },
    },
    {
        "type": "function",
        "name": "calculate",
        "description": "두 숫자를 더한다",
        "parameters": {
            "type": "object",
            "properties": {
                "number1": {
                    "type": "number",
                    "description": "첫 번째 숫자",
                },
                "number2": {
                    "type": "number",
                    "description": "두 번째 숫자",
                },
            },
            "required": ["numbers"],
        },
    },
]

tools = {
    "get_planet_mass": get_planet_mass,
    "calculate": calculate
}