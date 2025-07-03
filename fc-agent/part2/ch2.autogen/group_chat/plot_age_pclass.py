# filename: plot_age_pclass.py
import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터 다운로드 및 로드
url = 'https://github.com/mwaskom/seaborn-data/raw/master/titanic.csv'
data = pd.read_csv(url)

# 2. 데이터셋의 열 출력
print("Columns in the dataset:")
print(data.columns)

# 3. age와 pclass 변수 간의 관계를 시각화하는 차트 생성
plt.figure(figsize=(10, 6))
plt.scatter(data['pclass'], data['age'], alpha=0.5)
plt.title('Relationship between Age and Pclass')
plt.xlabel('Pclass')
plt.ylabel('Age')
plt.grid(True)

# 차트를 파일로 저장
plt.savefig('age_pclass_relation.png')
plt.close()