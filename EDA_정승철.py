import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 불필요한 경고문 생략(선택)
import warnings
warnings.filterwarnings('ignore')

# 모든 컬럼 출력설정(선택)
pd.set_option('display.max_columns', None)

# 데이터 로드
data = pd.read_csv('C:/Users/goodm/Downloads/Churn_Modelling.csv',index_col=0)
data

# data type 확인
data.info()

# 결측치 확인 및 제거(4개 행 제거거)

data.isnull().sum()
nan_data = data.dropna()

# 중복 값 행 검색 및 행 삭제
data[data.duplicated()]

new_data = nan_data[~nan_data.duplicated()]

new_data
new_data.describe()

# RAW EDA : 각 지표 별 그래프 작성 및 확인

plt.boxplot(new_data['CreditScore'])
plt.show() #CreditScore에서 너무 작은 값 다수 존재, 해당 인원들의 이탈여부 확인 예정

# 위 그래프의 이상치들은 전부 이탈 : 신용 점수가 너무 낮은 인원은 이탈한다 볼 수 있다.
IQR = new_data['CreditScore'].quantile(q=0.75) - new_data['CreditScore'].quantile(q=0.25)
min = new_data['CreditScore'].quantile(q=0.25) - 1.5*IQR
print(f"사분위수 이상치 판독 최소값 기준 : {min}")

new_data[new_data['CreditScore'] < min]['Exited']

# 연령, 잔고를 통한 고객 상태 확인 : 연령대 그룹화 진행
new_data['Age_Group'] = (new_data['Age'] // 10) * 10
exit_data = new_data[new_data['Exited'] == 1]

sns.scatterplot(data=exit_data, x='Age_Group', y='Balance')

# 목표변수, 범주형 자료 제외하고 변수 간 상관관계 확인, heatmap 사용 : 수치형 자료 사이에는 큰 상관관계 존재 X

int_data = new_data.drop(columns=['CustomerId', 'Surname', 'Geography', 'Gender', 'Age'])
int_data.corr(method='pearson')

import seaborn as sb
plt.figure(figsize=(10,10))
sb.heatmap(int_data.corr(), annot=True, vmin=-1, vmax=1)

# 전체 데이터(수치형)에 대한 로지스틱 회귀 진행
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def get_att(x):
    # x모델 넣기
    print('클래스 종류', x.classes_)
    print('독립변수 갯수', x.n_features_in_)
    print('들어간 독립변수(x)의 이름', x.feature_names_in_)
    print('가중치', x.coef_)
    print('바이어스', x.intercept_)

def get_metrics(true, pred):
    print('정확도', accuracy_score(true, pred))
    print('f1-score', f1_score(true, pred))

model_lor = LogisticRegression()
X = int_data[['CreditScore', 'Tenure', 'Balance', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Age_Group']]
y_true = int_data[['Exited']]

model_lor.fit(X, y_true)
get_att(model_lor)

y_pred = model_lor.predict(X)
get_metrics(y_true, y_pred)