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

# 결측치 확인 및 제거(4개 행 제거)
data.isnull().sum()
nan_data = data.dropna()

# 중복 값 행 검색 및 행 삭제
nan_data[nan_data.duplicated()]

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

# 성별, 국가 범주형 변수에서 수치형 변수로 인코딩 시작
def get_gender(x):
    if x == 'Female':
        return 1
    else:
        return 0
    
def get_geography(x):
    if x == 'France':
        return 1
    elif x == 'Germany':
        return 2
    else:
        return 3

new_data['Gender'] = new_data['Gender'].apply(get_gender)
new_data['Geography'] = new_data['Geography'].apply(get_geography)

new_data

# 목표변수, 범주형 자료 제외하고 변수 간 상관관계 확인, heatmap 사용 : 수치형 자료 사이에는 큰 상관관계 존재 X
int_data = new_data.drop(columns=['CustomerId', 'Surname', 'Age'])
int_data.corr(method='pearson')

import seaborn as sb
plt.figure(figsize=(10,10))
sb.heatmap(int_data.corr(), annot=True, vmin=-1, vmax=1)

# 다중공산성 확인 : 각 변수가 다른 독립 변수 전체와의 상관 관계 강도, 10이 넘어가면 다중공산성이 높아 원하는 결과가 나오기 힘들다고 예측 가능, 제거하는 것이 이상적
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["변수명"] = int_data.columns
vif["VIF"] = [variance_inflation_factor(int_data.values, i) for i in range(int_data.shape[1])]

vif

# 로지스틱 회귀 결과 해석을 위한 함수 지정 및 변수 분류
def get_att(x):
    # x모델 넣기
    print('클래스 종류', x.classes_)
    print('독립변수 갯수', x.n_features_in_)
    print('들어간 독립변수(x)의 이름', x.feature_names_in_)
    print('가중치', x.coef_)
    print('바이어스', x.intercept_)

def get_metrics(true, pred):
    print('정확도', round(accuracy_score(true, pred), 4))
    print('f1-score', round(f1_score(true, pred), 4))


X = int_data[['CreditScore', 'Tenure', 'Balance', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Age_Group', 'Geography', 'Gender']]
y_true = int_data[['Exited']]

# 전체 데이터(수치형)에 대한 로지스틱 회귀 진행
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

model_lor = LogisticRegression()

model_lor.fit(X, y_true)
get_att(model_lor)

y_pred = model_lor.predict(X)
get_metrics(y_true, y_pred)

# 성별 간 로지스틱 회귀 분석 진행

sex_data = new_data.drop(columns=['CustomerId', 'Surname', 'Age'])
male_data = sex_data[sex_data['Gender'] == 0]
female_data = sex_data[sex_data['Gender'] == 1]

X_male = male_data[['CreditScore', 'Tenure', 'Balance', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Age_Group','Geography']]
y_true_male = male_data[['Exited']]

model_lor.fit(X_male, y_true_male)
get_att(model_lor)

y_pred_male = model_lor.predict(X_male)
get_metrics(y_true_male, y_pred_male)

X_female = female_data[['CreditScore', 'Tenure', 'Balance', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Age_Group', 'Geography']]
y_true_female = female_data[['Exited']]

model_lor.fit(X_female, y_true_female)
get_att(model_lor)

y_pred_female = model_lor.predict(X_female)
get_metrics(y_true_female, y_pred_female)

"""
- 데이터 스케일링
    - 정규화 진행 : 표준화 진행시 -1 ~ 1 사이 값을 가짐으로, 일부 범주형 변수를 처리한 0, 1과 같은 양의 정수 값들과 분석 상의 오류 발생 가능성 있음.
    - 데이터에서 평균을 뺀 후, 변수에서의 최대값에서 최소값을 뺀 값으로 나눈다
"""

# 데이터 엔지니어링(결측치 제거 후 전체 데이터셋 기준), 30:70 으로 훈련 셋과 테스트셋 분할
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

mms = MinMaxScaler()

X_data_normal = mms.fit_transform(X)
train_normal = pd.DataFrame(X_data_normal, index=X.index, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X,y_true, test_size = 0.3, random_state= 42)

# SHAP를 통한 각 항목별 이탈률 기여도 확인, 사용 모델은 XGBoost 회귀 모델
# !pip install shap
# !pip install xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import shap

model_xg = XGBClassifier(random_state = 42)
model_xg.fit(X_train, y_train)

ex = shap.Explainer(model_xg)
shap_v = ex(X_train)

shap.plots.waterfall(shap_v[0])

y_train_pred = model_xg.predict(X_train)
print(f'train 정확도 : {accuracy_score(y_train, y_train_pred):.3f}')

y_test_pred = model_xg.predict(X_test)
print(f'train 정확도 : {accuracy_score(y_test, y_test_pred):.3f}')


