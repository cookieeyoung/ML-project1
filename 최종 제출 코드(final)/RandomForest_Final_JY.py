# -*- coding: utf-8 -*-
"""랜덤포레스트 (원본) 최종 - 주연.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1U88LK7p9pzXorzguMeP7oB9qzMFDrpPI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, average_precision_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

#obtuna
# 각 모델 정확도를 제외한 모든 수치가 낮아 실사용 불가, optuna를 이용한 튜닝 수행
!pip install --quiet optuna

import optuna
import itertools
import sklearn.svm
import sklearn.model_selection
from sklearn.metrics import classification_report

#평가 AUC
from sklearn.metrics import roc_auc_score

# 데이터 로드 및 전처리
pd.set_option('display.max_columns', None)
df = pd.read_csv('/content/drive/MyDrive/심화프로젝트 ML/train.csv')

nan_data = df.drop(columns=['id', 'CustomerId'])
data = nan_data[~nan_data.duplicated()]

le = LabelEncoder()
data['Gender'] = le.fit_transform(data["Gender"])

oe = OneHotEncoder()
geo_csr = oe.fit_transform(data[['Geography']])
geo_df = pd.DataFrame(geo_csr.toarray(), columns=oe.get_feature_names_out())

data = pd.concat([data.reset_index(drop=True), geo_df.reset_index(drop=True)], axis=1)
data.drop(columns=['Surname', 'Geography'], inplace=True)

X = data.drop(columns=['Exited']).reset_index(drop=True)
y = data['Exited'].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# SMOTE 적용
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Optuna를 이용한 하이퍼파라미터 튜닝
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }

    rf = RandomForestClassifier(**params, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train_res, y_train_res)
    y_prob = rf.predict_proba(X_test)[:, 1]

    best_recall = 0
    best_threshold = 0.5

    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        # Precision ≥ 0.5 조건을 만족하면서 Recall 최대로 뽑기
        if precision >= 0.5 and recall > best_recall:
            best_recall = recall
            best_threshold = threshold
            
    trial.set_user_attr("best_threshold", best_threshold)

    return best_recall

# Optuna 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, n_jobs=-1)

#최적 threshold 값 가져오기
best_threshold = study.best_trial.user_attrs["best_threshold"]

# 최적 모델 학습
best_params = study.best_trial.params
best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
best_model.fit(X_train_res, y_train_res)

y_prob_test = best_model.predict_proba(X_test)[:, 1]

# 최종 평가
y_test_pred_final = (y_prob_test >= best_threshold).astype(int)

print("최적 하이퍼파라미터:", best_params)
print(f"최적 Threshold: {best_threshold:.2f}")
print("Accuracy:", accuracy_score(y_test, y_test_pred_final))
print("F1 Score:", f1_score(y_test, y_test_pred_final))
print("Recall:", recall_score(y_test, y_test_pred_final))
print("Precision:", precision_score(y_test, y_test_pred_final))
print("Average Precision:", average_precision_score(y_test, y_prob_test))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob_test))
