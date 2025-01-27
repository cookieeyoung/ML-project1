import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 데이터 로드
df = pd.read_csv('Churn_Modelling.csv')

# 피처와 타겟 분리
X = df.drop('Exited', axis=1)
y = df['Exited']

# Train/Test 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


###### Preprocessing ######
# 결측치 제거
X_train_cleaned = X_train.dropna()
X_test_cleaned = X_test.dropna()
y_train_cleaned = y_train[X_train_cleaned.index]
y_test_cleaned = y_test[X_test_cleaned.index]

# 중복값 제거
X_train_unique = X_train_cleaned.drop_duplicates(subset='CustomerId', keep='first')
X_test_unique = X_test_cleaned.drop_duplicates(subset='CustomerId', keep='first')
y_train_unique = y_train_cleaned.loc[X_train_unique.index]
y_test_unique = y_test_cleaned.loc[X_test_unique.index]

# 범주형 변수 인코딩
encoder = LabelEncoder()
category_vars = ['Gender', 'HasCrCard', 'IsActiveMember']

for var in category_vars:
    X_train_unique[f'{var}_en'] = encoder.fit_transform(X_train_unique[var])
    X_test_unique[f'{var}_en'] = encoder.transform(X_test_unique[var])

# 원 핫 인코딩
geography_dummies_train = pd.get_dummies(X_train['Geography'], prefix='Geography')
geography_dummies_test = pd.get_dummies(X_test['Geography'], prefix='Geography')
X_train_unique = pd.concat([X_train_unique, geography_dummies_train.loc[X_train_unique.index]], axis=1)
X_test_unique = pd.concat([X_test_unique, geography_dummies_test.loc[X_test_unique.index]], axis=1)

# 필요없는 컬럼 드랍
columns_to_drop = ['RowNumber', 'CustomerId', 'Surname', 'Geography', 'Gender', 'IsActiveMember', 'HasCrCard']
X_train_final = X_train_unique.drop(columns=columns_to_drop)
X_test_final = X_test_unique.drop(columns=columns_to_drop)

# Scaling
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_final), columns=X_train_final.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_final), columns=X_test_final.columns)

# 최종 데이터 확인
# print(X_train_scaled.head())
# print(y_train_unique.head())
# print(X_test_scaled.head())
# print(y_test_unique.head())


###### Classification ######
# 모델 초기화
lm = LogisticRegression()

# 모델 학습
lm.fit(X_train_scaled, y_train_unique)

# 테스트 데이터에 대한 예측
y_pred = lm.predict(X_test_scaled)

# 모델 성능 평가
accuracy = accuracy_score(y_test_unique, y_pred)
conf_matrix = confusion_matrix(y_test_unique, y_pred)
class_report = classification_report(y_test_unique, y_pred)

# 결과 출력
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
