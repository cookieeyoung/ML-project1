#다중 로지스틱 회귀 모델링 - 베이스 라인 
#독립변수(X) : 12개 (순번/고객명/고객id 제외 전부)
#종속변수 Y : 1개 ('Exited') 
#전처리 : 결측치/중복값/인코딩/스케일링 (-6행) 
#교차검증/최적화 전

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder #인코딩
from sklearn.preprocessing import MinMaxScaler,StandardScaler #스케일링
from sklearn.model_selection import train_test_split #테스터 분리
from sklearn.linear_model import LogisticRegression #lor 회귀 모델
from sklearn.metrics import accuracy_score,f1_score #평가 모듈

#데이터 로드 
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
data = pd.read_csv('Churn_Modelling.csv')

# 데이터 셋 분할 (train/test)
X = data.drop(columns=['RowNumber','CustomerId','Surname','Exited'])
y = data['Exited']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y) 
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#전처리 함수

#중복값 처리 함수
def get_unique (df):
    df= df.drop_duplicates()
    return df

#결측치 처리 함수
def get_non_missing(df):
    df = df.dropna(axis=0, how='any')
    return df

#스케일링 함수화 (테스터 학습하지 않도록 주의!)
def get_sc (df):
    #정규화
    mn_sc = MinMaxScaler() #모델생성
    mn_sc.fit(X_train[['EstimatedSalary','Balance']]) # 학습

    # 표준화 -나이/신용점수
    sd_sc = StandardScaler() #모델생성
    sd_sc.fit(X_train[['Age','CreditScore']]) # 학습

    df[['sal_mn_sc','bal_mn_sc']] = mn_sc.transform(df[['EstimatedSalary','Balance']]) # 학습
    df[['age_sd_sc','score_sd_sc']] = sd_sc.transform(df[['Age','CreditScore']]) # 학습
    
    return df

#인코딩 함수
def get_encoding(df):

    # 레이블인코딩 - 성별(여성0, 남성1)
    le = LabelEncoder()
    df['Gender_le'] = le.fit_transform(df['Gender'])

    #원핫인코딩 - 국가
    oe = OneHotEncoder()
    oe.fit(df[['Geography']])
    geo_csr = oe.transform(df[['Geography']])

    csr_df = pd.DataFrame(geo_csr.toarray(), columns = oe.get_feature_names_out())

    df = df.reset_index(drop=True)  # df 인덱스 초기화
    csr_df = csr_df.reset_index(drop=True)  # csr_df 인덱스 초기화

    df = pd.concat([df,csr_df],axis=1)
    return df

X_train = get_encoding(X_train)
X_train.columns

#모델링 함수 
def get_model(df,y_true):
    model_lor = LogisticRegression() #동일하게 해주는게 맞나? random_state=42 옵션?
    X = df.drop(columns=['Age','Gender','Geography','CreditScore','Balance','EstimatedSalary'])
    return model_lor.fit(X,y_true)

#최종 학습모델 저장 
model_output = get_model(X_train,y_train)


#테스터 전처리
X_test2 = get_unique(X_test) #중복
X_test2 = get_non_missing(X_test2) #결측
y_test = y_test.loc[X_test2.index]  # 결측치 삭제된 X_test2와 같은 인덱스 유지 ->추가 학습 필요요
X_test2 = get_encoding(X_test2) #인코딩
X_test2 = get_sc(X_test2) #스케일링
X_test2 = X_test2[X_train.columns]

# train / test 예측 및 평가 
#예측 
X1 = X_train.drop(columns=['Age','Gender','Geography','CreditScore','Balance','EstimatedSalary'])
y_train_pred = model_output.predict(X1)

#평가
train_ac = accuracy_score(y_train,y_train_pred)
train_f1 = f1_score(y_train,y_train_pred)

#예측 
X2 = X_test2.drop(columns=['Age','Gender','Geography','CreditScore','Balance','EstimatedSalary'])
y_test_pred = model_output.predict(X2)

#평가
test_ac = accuracy_score(y_test,y_test_pred)
test_f1 = f1_score(y_test,y_test_pred)

print(f'accuracy_train : {test_ac:.3f}, f1-score_train : {test_f1:.3f}')
print(f'accuracy_test : {train_ac:.3f}, f1-score_test : {train_f1:.3f}')
