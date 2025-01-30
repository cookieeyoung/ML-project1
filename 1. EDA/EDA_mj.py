import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
df = pd.read_csv('Churn_Modelling.csv')
print(df.info())
# print(df.describe(include='all'))
# print(df.dtypes)


# 연령 분포 확인
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# 잔고 분포 확인
sns.histplot(df['Balance'], bins=30, kde=True, color='green')
plt.title('Balance Distribution')
plt.show()

# 신용 점수 분포 확인
sns.histplot(df['CreditScore'], bins=30, kde=True, color='red')
plt.title('Credit Score Distribution')
plt.show()

# 거래 기간 분포 확인
sns.histplot(df['Tenure'], bins=11, kde=True, color='blue')  # bins 값을 11로 조정
plt.title('Tenure Distribution')
plt.xticks(range(0, 11))
plt.show()

# 고객이 이용하는 은행 제품 개수 분포 확인
sns.histplot(df['Tenure'], bins=11, kde=True, color='blue')
plt.title('Tenure Distribution')
plt.xticks(range(0, 11))
plt.show()

# 고객 예상 급여 분포 확인
sns.histplot(df['EstimatedSalary'], bins=30, kde=True, color='blue')
plt.title('EstimatedSalary Distribution')
plt.show()


def exited_distribution(df, column, bins=10):
    # 해당 컬럼을 구간별로 나누기
    df['bin'] = pd.qcut(df[column], q=bins, duplicates='drop', precision=0)

    # 이탈 여부별로 카운트
    group_data = df.groupby('bin')['Exited'].value_counts().unstack().fillna(0)

    # 막대 그래프 그리기
    ax = group_data.plot(kind='bar', stacked=True, color=['#4CAF50', '#FF6347'], figsize=(10, 6))
    plt.title(f'{column} Distribution by Exited Status')
    plt.xlabel(f'{column} Bins')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(['Not Exited', 'Exited'], title='Status')
    plt.tight_layout()
    plt.show()

for col in ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']:
    exited_distribution(df, col)


# 범주형 변수의 분포 확인
category_vars = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
for var in category_vars:
    sns.countplot(x=var, data=df)
    plt.title(f'Distribution of {var}')
    plt.show()

# 이탈률 확인
sns.countplot(x='Exited', data=df)
plt.title('Distribution of Exited')
plt.show()

# 결측치 시각화
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Value Heatmap')
plt.show()


# 범주형 변수 인코딩
# 범주형 데이터 레이블 인코딩
category_vars = ['Gender', 'HasCrCard', 'IsActiveMember']
encoder = LabelEncoder()
for var in category_vars:
    df[f'{var}_en'] = encoder.fit_transform(df[var])

# 원 핫 인코딩
geography_dummies = pd.get_dummies(df['Geography'], prefix='Geography')
df = pd.concat([df, geography_dummies], axis=1)
# print(df.dtypes)

for column in df.columns:
    if df[column].dtype == 'bool':
        df[column] = df[column].astype(int)

# 인코딩 된 변수 분포 확인
for var in category_vars:
    sns.histplot(df[f'{var}_en'], bins=30, kde=True, color='blue')
    plt.title(f'encoded_{var}')
    plt.show()


## 상관관계 분석
# 수치형 데이터만 포함된 DataFrame 생성
numeric_df = df.select_dtypes(include=[np.number])

# 이랕 고객만 보기
exited_customers = numeric_df[df['Exited'] == 1]

# 상관관계 행렬 계산
corr_matrix = numeric_df.corr()
corr_matrix2 = exited_customers.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix2, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 전체 시각화
sns.pairplot(data=numeric_df, kind='scatter', plot_kws={'s': 5}, palette='deep')
plt.show()
