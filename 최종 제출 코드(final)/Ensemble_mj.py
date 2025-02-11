import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import average_precision_score, roc_auc_score
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import optuna
import warnings
from tqdm.auto import tqdm

# 경고 무시 및 판다스 설정
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

print("1. 데이터 로드")
# train, test 데이터 로드
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# train 데이터 전처리
train_data = train_df.drop(columns=['id', 'CustomerId'])
train_data = train_data[~train_data.duplicated()]

# test 데이터 전처리
test_data = test_df.drop(columns=['id', 'CustomerId'])

print("2. 데이터 분할")
# train 데이터를 train과 validation으로 분할 (8:2)
train_final, valid_data = train_test_split(train_data,
                                           test_size=0.2,
                                           stratify=train_data['Exited'],
                                           random_state=42)

print("3. Random Forest 전처리")
# Random Forest를 위한 전처리
le = LabelEncoder()

# Train 데이터 전처리
train_rf = train_final.copy()
train_rf['Gender'] = le.fit_transform(train_rf["Gender"])
oe = OneHotEncoder()
geo_csr_train = oe.fit_transform(train_rf[['Geography']])
geo_df_train = pd.DataFrame(geo_csr_train.toarray(), columns=oe.get_feature_names_out())
train_rf = pd.concat([train_rf.reset_index(drop=True), geo_df_train], axis=1)
train_rf.drop(columns=['Surname', 'Geography'], inplace=True)

# Validation 데이터 전처리
valid_rf = valid_data.copy()
valid_rf['Gender'] = le.transform(valid_rf["Gender"])
geo_csr_valid = oe.transform(valid_rf[['Geography']])
geo_df_valid = pd.DataFrame(geo_csr_valid.toarray(), columns=oe.get_feature_names_out())
valid_rf = pd.concat([valid_rf.reset_index(drop=True), geo_df_valid], axis=1)
valid_rf.drop(columns=['Surname', 'Geography'], inplace=True)

# Test 데이터 전처리
test_rf = test_data.copy()
test_rf['Gender'] = le.transform(test_rf["Gender"])
geo_csr_test = oe.transform(test_rf[['Geography']])
geo_df_test = pd.DataFrame(geo_csr_test.toarray(), columns=oe.get_feature_names_out())
test_rf = pd.concat([test_rf.reset_index(drop=True), geo_df_test], axis=1)
test_rf.drop(columns=['Surname', 'Geography'], inplace=True)

print("4. CatBoost 전처리")
# CatBoost를 위한 전처리
train_cat = train_final.copy()
train_cat.drop(columns=['Surname'], inplace=True)

valid_cat = valid_data.copy()
valid_cat.drop(columns=['Surname'], inplace=True)

test_cat = test_data.copy()
test_cat.drop(columns=['Surname'], inplace=True)


print("5. 특성과 타겟 분리")
# RF 데이터 분리
X_train_rf = train_rf.drop(columns=['Exited'])
y_train_rf = train_rf['Exited']
X_valid_rf = valid_rf.drop(columns=['Exited'])
y_valid_rf = valid_rf['Exited']
X_test_rf = test_rf

# CatBoost 데이터 분리
X_train_cat = train_cat.drop(columns=['Exited'])
y_train_cat = train_cat['Exited']
X_valid_cat = valid_cat.drop(columns=['Exited'])
y_valid_cat = valid_cat['Exited']
X_test_cat = test_cat

print("6. SMOTE 적용")
# RF용 SMOTE
sm = SMOTE(random_state=42)
X_train_rf_res, y_train_rf_res = sm.fit_resample(X_train_rf, y_train_rf)

# CatBoost용 SMOTE-NC
cat_features = ['Geography', 'Gender']
smote_nc = SMOTENC(categorical_features=[X_train_cat.columns.get_loc(col) for col in cat_features],
                   random_state=42)
X_train_cat_res, y_train_cat_res = smote_nc.fit_resample(X_train_cat, y_train_cat)


def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    max_depth = trial.suggest_int("max_depth", 3, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train_rf_res, y_train_rf_res)
    y_prob = rf.predict_proba(X_valid_rf)[:, 1]

    best_f1 = 0
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        recall = recall_score(y_valid_rf, y_pred)
        precision = precision_score(y_valid_rf, y_pred)
        if recall >= 0.8 and precision >= 0.5:
            f1 = f1_score(y_valid_rf, y_pred)
            if f1 > best_f1:
                best_f1 = f1

    return best_f1 if best_f1 > 0 else 0


print("7. Random Forest 최적화")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, n_jobs=-1)

print("8. 최적 Random Forest 모델 학습")
best_params_rf = study.best_trial.params
rf_model = RandomForestClassifier(**best_params_rf, random_state=42, n_jobs=-1)
rf_model.fit(X_train_rf_res, y_train_rf_res)

print("9. CatBoost 모델 학습")
cat_params = {
    'learning_rate': 0.02678152407579277,
    'depth': 8,
    'n_estimators': 111,
    'l2_leaf_reg': 0.0008386712970316092
}

cat_model = CatBoostClassifier(
    **cat_params,
    random_state=42,
    cat_features=[X_train_cat.columns.get_loc(col) for col in cat_features],
    verbose=False,
    class_weights={0: 0.32, 1: 0.68}
)

cat_model.fit(X_train_cat_res, y_train_cat_res)


def ensemble_predict(rf_probs, cat_probs, rf_weight, cat_weight, threshold):
    weighted_probs = (rf_probs * rf_weight + cat_probs * cat_weight) / (rf_weight + cat_weight)
    return (weighted_probs >= threshold).astype(int)


print("10. 앙상블 모델 최적화")
# 검증 세트에 대한 예측 확률
rf_valid_probs = rf_model.predict_proba(X_valid_rf)[:, 1]
cat_valid_probs = cat_model.predict_proba(X_valid_cat)[:, 1]

# 최적의 가중치와 임계값 찾기
best_f1 = 0
best_weights = (0.5, 0.5)
best_threshold = 0.5

total_iterations = len(np.arange(0.1, 1.0, 0.1)) * len(np.arange(0.3, 0.7, 0.01))
pbar = tqdm(total=total_iterations, desc="앙상블 최적화")

for rf_w in np.arange(0.1, 1.0, 0.1):
    cat_w = 1 - rf_w
    for threshold in np.arange(0.3, 0.7, 0.01):
        y_pred = ensemble_predict(rf_valid_probs, cat_valid_probs, rf_w, cat_w, threshold)

        recall = recall_score(y_valid_rf, y_pred)
        precision = precision_score(y_valid_rf, y_pred)
        f1 = f1_score(y_valid_rf, y_pred)

        if recall >= 0.8 and precision >= 0.5 and f1 > best_f1:
            best_f1 = f1
            best_weights = (rf_w, cat_w)
            best_threshold = threshold

        pbar.update(1)

pbar.close()

print("11. 최종 테스트 세트 예측")
# 테스트 세트에 대한 최종 예측
rf_test_probs = rf_model.predict_proba(X_test_rf)[:, 1]
cat_test_probs = cat_model.predict_proba(X_test_cat)[:, 1]
final_test_pred = ensemble_predict(rf_test_probs, cat_test_probs,
                                 best_weights[0], best_weights[1],
                                 best_threshold)

# 결과 출력
print("\n=== Random Forest 모델 최적 파라미터 ===")
print(best_params_rf)

print("\n=== 검증 세트 결과 ===")
valid_pred = ensemble_predict(rf_valid_probs, cat_valid_probs,
                              best_weights[0], best_weights[1],
                              best_threshold)
print(f"최적 가중치 (RF:CAT) = {best_weights[0]:.2f}:{best_weights[1]:.2f}")
print(f"최적 임계값 = {best_threshold:.2f}")
print("Accuracy:", accuracy_score(y_valid_rf, valid_pred))
print("F1 Score:", f1_score(y_valid_rf, valid_pred))
print("Recall:", recall_score(y_valid_rf, valid_pred))
print("Precision:", precision_score(y_valid_rf, valid_pred))
print("ROC AUC:", roc_auc_score(y_valid_rf, valid_pred))
print("Average Precision:", average_precision_score(y_valid_rf, valid_pred))

