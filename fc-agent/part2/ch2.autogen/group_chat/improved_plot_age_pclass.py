# filename: improved_plot_age_pclass.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 다운로드 및 로드
url = 'https://github.com/mwaskom/seaborn-data/raw/master/titanic.csv'
data = pd.read_csv(url)

# 2. NaN 값 처리: age 열의 NaN 값을 평균으로 대체
data['age'].fillna(data['age'].mean(), inplace=True)

# 3. 데이터셋의 열 출력
print("Columns in the dataset:")
print(data.columns)

# 4. pclass에 따른 색상 인코딩을 갖는 상자 수염 그림 생성
plt.figure(figsize=(10, 6))
sns.boxplot(x='pclass', y='age', data=data, palette='pastel')
plt.title('Age Distribution by Pclass', fontsize=16)
plt.xlabel('Pclass', fontsize=14)
plt.ylabel('Age', fontsize=14)
plt.grid(axis='y')

# 범례 추가 (필요시 추가)
plt.legend(title='Pclass', loc='upper right', labels=data['pclass'].unique())

# 차트를 파일로 저장
plt.savefig('age_pclass_relation_improved.png')
plt.close()