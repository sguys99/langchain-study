# filename: plot_age_pclass_updated.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 다운로드 및 읽기
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
data = pd.read_csv(url)

# 2. 열 출력
print(data.columns)

# 3. age와 pclass 간의 관계를 시각화
plt.figure(figsize=(10, 6))
sns.boxplot(x='pclass', y='age', data=data)
plt.title('Age vs Pclass')

# 4. 차트를 파일로 저장
plt.savefig('age_vs_pclass.png')
print("차트가 'age_vs_pclass.png' 파일로 저장되었습니다.")