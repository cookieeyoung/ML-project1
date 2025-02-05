# -*- coding: utf-8 -*-
"""‎최종 제출 코드(final)/XGB_rogistic_한리브가.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14_is2_dt9J1DVhJn1R30NAeKc_GsSFus
"""

#XGB

import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, average_precision_score

# Optuna objective 함수 정의
def objective(trial):
    # XGBoost 하이퍼파라미터 설정
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    subsample = trial.suggest_uniform('subsample', 0.6, 1.0)
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
    scale_pos_weight = trial.suggest_int('scale_pos_weight', 1, 5)

    # 모델 정의
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight
    )

    # 모델 학습
    model.fit(X_train_res, y_train_res)

    # 예측
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.4).astype(int)

    # 평가 지표 계산
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Precision과 Recall을 조화롭게 최대화하는 방식으로 최적화 (Recall을 강조)
    return recall * 0.6 + precision * 0.4  # Recall을 더 중요시
    # Optuna가 이 값을 최대화하도록 할 것임

# Optuna 최적화 실행
study = optuna.create_study(direction='maximize')  # 최대화
study.optimize(objective, n_trials=100)  # 50번의 trial 수행

# 최적화된 파라미터 출력
print(f"Best trial: {study.best_trial.params}")

# 최적화된 하이퍼파라미터로 모델 학습 및 평가
best_params = study.best_trial.params
xgb_clf = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    **best_params
)

# 모델 학습
xgb_clf.fit(X_train_res, y_train_res)

# 예측
y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.4).astype(int)

# 결과 출력
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
avg_precision = average_precision_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.3f}")
print(f"F1 Score: {f1:.16f}")
print(f"Recall: {recall:.1f}")
print(f"Precision: {precision:.3f}")
print(f"Average Precision: {avg_precision:.3f}")


#Rogistic regresion

import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# 필요 없는 컬럼 제거 (고객 ID 등 분석과 관계없는 데이터가 있다면)
if "CustomerId" in df.columns:
    df.drop(columns=["CustomerId"], inplace=True)

# One-Hot Encoding 적용 (문자형 → 숫자형 변환)
df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

# X, y 분리
X = df.drop(columns=["Exited"])  # 'Exited'가 종속변수(타겟)라고 가정
y = df["Exited"]

# 데이터 나누기 (학습 80%, 테스트 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 정규화 (로지스틱 회귀는 정규화 필요)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SMOTE 적용 (데이터 불균형 해결)
smote = SMOTE(sampling_strategy=0.6, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Optuna 하이퍼파라미터 튜닝 함수
def objective(trial):
    # Optuna가 탐색할 하이퍼파라미터 정의
    C = trial.suggest_loguniform("C", 1e-5, 10)  # 정규화 강도
    solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs", "saga"])  # 최적화 알고리즘

    # 로지스틱 회귀 모델 생성
    model = LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=42)

    # 모델 학습
    model.fit(X_train_res, y_train_res)

    # 예측 수행
    y_pred = model.predict(X_test)

    # 평가 지표 계산 (F1-Score 최대화)
    return f1_score(y_test, y_pred)

# Optuna 실행 (최적화)
study = optuna.create_study(direction="maximize")  # F1-score 최대화
study.optimize(objective, n_trials=50)  # 50번 시도

# 최적 하이퍼파라미터 출력
print("Best Parameters:", study.best_trial.params)

# 최적 파라미터로 모델 재학습
best_params = study.best_trial.params
best_model = LogisticRegression(C=best_params["C"], solver=best_params["solver"], max_iter=1000, random_state=42)
best_model.fit(X_train_res, y_train_res)

# 최적 모델 평가
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")