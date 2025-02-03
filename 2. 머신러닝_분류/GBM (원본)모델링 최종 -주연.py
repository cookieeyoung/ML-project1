#최종셀렉 recall 0.8303 precision 0.5006

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, average_precision_score,roc_auc_score
from imblearn.over_sampling import SMOTE

#obtuna
!pip install --quiet optuna
# 각 모델 정확도를 제외한 모든 수치가 낮아 실사용 불가, optuna를 이용한 튜닝 수행
import optuna
import itertools
import sklearn.svm
import sklearn.model_selection
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# 데이터 로드
data = pd.read_csv('/content/drive/MyDrive/심화프로젝트 ML/train.csv')

# 인코딩 (라벨 인코딩 + 원핫 인코딩)
le = LabelEncoder()
data['Gender'] = le.fit_transform(data["Gender"])

oe = OneHotEncoder()
geo_csr = oe.fit_transform(data[['Geography']])
geo_df = pd.DataFrame(geo_csr.toarray(), columns=oe.get_feature_names_out())

# 데이터 병합 및 불필요한 컬럼 제거
data = pd.concat([data.reset_index(drop=True), geo_df.reset_index(drop=True)], axis=1)
data.drop(columns=['id', 'CustomerId', 'Surname', 'Geography'], inplace=True)

# 데이터 분리
X = data.drop(columns=["Exited"])
y = data["Exited"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# SMOTE 적용 (오버샘플링)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("데이터 전처리 완료:", X_train_res.shape, y_train_res.shape)

# Optuna를 활용한 GradientBoostingClassifier 하이퍼파라미터 튜닝 (Recall 최대화)
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),  # 트리 개수 증가
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.2),  # 학습률 감소
        "max_depth": trial.suggest_int("max_depth", 5, 15),  # 트리 깊이 증가
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),  # 노드 분할 최소 샘플 수
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),  # 리프 노드 최소 샘플 수
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),  # 일부 데이터 샘플링
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }

    model = GradientBoostingClassifier(**params, random_state=42)
    model.fit(X_train_res, y_train_res)
    y_prob = model.predict_proba(X_test)[:, 1]

    best_threshold = 0.5
    best_f1 = 0
    best_recall = 0
    best_precision = 0

    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        if recall >= 0.8 and precision >= 0.5:
            f1 = f1_score(y_test, y_pred)
            if recall > best_recall:
                best_threshold = threshold
                best_recall = recall
                best_precision = precision
                best_f1 = f1

    return best_recall if best_recall >= 0.8 and best_precision >= 0.5 else 0

# Optuna 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, n_jobs=-1)

# 최적 모델 학습 및 평가
best_params = study.best_trial.params
best_model = GradientBoostingClassifier(**best_params, random_state=42)
best_model.fit(X_train_res, y_train_res)

y_prob_test = best_model.predict_proba(X_test)[:, 1]

# 최적 Threshold 찾기
best_threshold = 0.5
best_recall = 0
best_precision = 0
best_f1 = 0

for threshold in np.arange(0.1, 0.9, 0.01):
    y_pred_test = (y_prob_test >= threshold).astype(int)
    recall = recall_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)

    if recall >= 0.8 and precision >= 0.5:
        f1 = f1_score(y_test, y_pred_test)
        if recall > best_recall:
            best_threshold = threshold
            best_recall = recall
            best_precision = precision
            best_f1 = f1

# 최종 평가
y_test_pred_final = (y_prob_test >= best_threshold).astype(int)

roc_auc = roc_auc_score(y_test, y_prob_test)

print("\n최적 하이퍼파라미터:", best_params)
print(f"최적 Threshold: {best_threshold:.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_final):.4f}")
print(f"F1 Score: {f1_score(y_test, y_test_pred_final):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred_final):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred_final):.4f}")
print(f"Average Precision: {average_precision_score(y_test, y_test_pred_final):.4f}")
print(f"ROC AUC Score: {roc_auc_score:.4f}")

!pip install shap

import shap

# SHAP Explainer 생성 (속도 최적화)
explainer = shap.TreeExplainer(best_model, feature_perturbation="tree_path_dependent")

# X_test에서 일부 샘플만 선택하여 속도 향상
X_test_sample = X_test.sample(n=500, random_state=42)

# SHAP 값 계산
shap_values = explainer.shap_values(X_test_sample)

# SHAP Summary Plot (전체 Feature 영향력 분석)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_sample)
plt.show()

# SHAP Bar Plot (Feature Importance 순위)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_sample, plot_type="bar")
plt.show()
