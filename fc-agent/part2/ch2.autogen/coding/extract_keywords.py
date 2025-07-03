# filename: extract_keywords.py
import re
from collections import Counter

# 예시 문장
sentence = "Artificial intelligence is transforming the way we live and work."

# 문장에서 단어 추출 및 소문자로 변환
words = re.findall(r'\b\w+\b', sentence.lower())

# 불용어 제거
stop_words = set(['is', 'the', 'and', 'we'])
keywords = [word for word in words if word not in stop_words]

# 키워드의 빈도수 계산
keyword_counts = Counter(keywords)

# 가장 흔한 키워드 출력
print(keyword_counts.most_common())