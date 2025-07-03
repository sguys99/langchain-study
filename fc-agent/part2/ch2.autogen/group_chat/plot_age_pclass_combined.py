# filename: plot_age_pclass_combined.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 다운로드 및 읽기
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
data = pd.read_csv(url)

# 2. 열 출력
print(data.columns)

# 3. age와 pclass 간의 관계를 시각화
plt.figure(figsize=(12, 6))

# 박스 플롯
plt.subplot(1, 2, 1)
sns.boxplot(x='pclass', y='age', data=data)
plt.title('Boxplot of Age by Pclass')

# 산점도
plt.subplot(1, 2, 2)
sns.scatterplot(x='pclass', y='age', data=data, alpha=0.5)
plt.title('Scatterplot of Age by Pclass')

# 4. 차트를 파일로 저장
plt.tight_layout()  # 서브플롯 간의 공간 조정
plt.savefig('age_vs_pclass_combined.png')
print("차트가 'age_vs_pclass_combined.png' 파일로 저장되었습니다.")