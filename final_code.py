import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler

#모델링 모듈
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, average_precision_score, confusion_matrix,roc_auc_score
import shap
from imblearn.over_sampling import SMOTE,SMOTENC
import optuna

#1. data_load
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
df = pd.read_csv('train.csv')
print('원본 데이터 :',df.shape)

nan_data = df.drop(columns=['id','CustomerId'])
data = nan_data[~nan_data.duplicated()]
print('중복 처리 :',data.shape)

#check
X = data.drop(columns=['Surname','Exited'],axis=1).reset_index(drop=True)
y_true = data['Exited'].reset_index(drop=True)
print('전처리 완료:',X.shape,y_true.shape)
display(X.info())
print('----------------------------')

#2.EDA

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

#encoding : gender-label ( Female = 0, male = 1) /  geography-onehot
le = LabelEncoder()
data['Gender'] = le.fit_transform(data["Gender"])

oe = OneHotEncoder()
oe.fit(data[['Geography']])
geo_csr = oe.transform(data[['Geography']])
csr_df = pd.DataFrame(geo_csr.toarray(), columns = oe.get_feature_names_out())
df = data.reset_index(drop=True)  # df 인덱스 초기화
csr_df = csr_df.reset_index(drop=True)  # csr_df 인덱스 초기화
inco_df = pd.concat([df,csr_df],axis=1)

#check
int_data = inco_df.drop(columns=['Surname','Geography'])
X = int_data.drop("Exited", axis=1)
y_true = int_data['Exited']
print('전처리 완료:',X.shape,y_true.shape)
print('----------------------------')

#4.data engineering
#tester split
X_train, X_test, y_train, y_test = train_test_split(X,y_true,stratify = y_true,test_size = 0.2, random_state= 42)
print('데이터 분리 후 크기 : ',X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#over sampling
# SMOTE 적용

# SMOTE-NC 적용
# 범주형 변수의 인덱스 설정
cat_features = ['Geography','Gender']
smote_nc = SMOTENC(categorical_features=cat_features, random_state=42)
X_train_res, y_train_res = smote_nc.fit_resample(X_train, y_train)

# 증강 후 데이터 비율 확인
print("Before SMOTE:", y_train.value_counts(normalize=True))
print("After SMOTENC:", y_train_res.value_counts())

X_train_res.info()

# 각각의 모델 최적화 수행 및 결과 출력

#XGBoost
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

# Logistic Regression

# X, y 분리
X = nan_data.drop(columns=["Exited"])  # 'Exited'가 종속변수(타겟)라고 가정
y = nan_data["Exited"]

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

#Catboost 모델
# 파라미터 튜닝 
params = {
    'learning_rate':0.02678152407579277,  
    'depth': 8,   
    'n_estimators': 111, 
    'l2_leaf_reg':0.0008386712970316092
}

model = CatBoostClassifier(**params,random_state=42,cat_features=cat_features,verbose=False,class_weights={0:0.32,1:0.68})

model.fit(X_train_res, y_train_res) #범주형 인덱스 추가
y_pred = model.predict(X_test)

#임계값 조정
y_pred_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.48  # threshold를 낮추어 recall을 높임
y_pred = (y_pred_prob >= threshold).astype(int)

print('ac:', accuracy_score(y_test, y_pred))
print('f1:', f1_score(y_test, y_pred))
print('recall:', recall_score(y_test, y_pred))
print('precision:', precision_score(y_test, y_pred))
print("AUC :", roc_auc_score(y_test, y_pred))

# SHAP 값 계산을 위한 Explainer 생성
explainer = shap.Explainer(model)  #학습 모델전달
shap_values = explainer(X_test)  # 테스트 데이터로 SHAP 값 계산

# Summary Plot (변수 중요도 시각화)
shap.summary_plot(shap_values, X_test)
shap.plots.waterfall(shap_values[0])

# Bar Plot (변수 중요도 막대 그래프) gtp 코드임 추가학습 필요
shap_values_array = shap_values.values  # SHAP 값 배열 추출
feature_names = X_test.columns  # 변수 이름 가져오기

# SHAP 평균 절대값 기준으로 변수 중요도 계산
shap_importance = np.abs(shap_values_array).mean(axis=0)
shap_importance_df = pd.DataFrame({'Feature': feature_names, 'SHAP Importance': shap_importance})
shap_importance_df = shap_importance_df.sort_values(by="SHAP Importance", ascending=False)

# Bar Plot 그리기
plt.figure(figsize=(10, 6))
plt.barh(shap_importance_df["Feature"], shap_importance_df["SHAP Importance"], color='skyblue')
plt.xlabel("SHAP Importance")
plt.ylabel("Feature")
plt.title("Feature Importance - SHAP values")
plt.gca().invert_yaxis()  # 중요도가 높은 변수를 위로 정렬
plt.show()

#Decision Tree 모델
# 4. 모델 학습 (최적화된 파라미터 적용)
dt = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=23,
    min_samples_leaf=17,
    max_features='log2',
    class_weight={0: 1.0, 1: 8.61},
    random_state=42
)
dt.fit(X_train_res, y_train_res)

# 5. 예측 및 threshold 적용
def predict_with_threshold(model, X, threshold=0.56):
    y_pred_proba = model.predict_proba(X)[:, 1]
    return (y_pred_proba >= threshold).astype(int)

# 6. 모델 평가
def evaluate_model(y_true, y_pred, model_name, threshold):
    print(f"\n{model_name} 성능 (threshold={threshold:.3f}):")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_pred):.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\nSpecificity: {tn/(tn+fp):.4f}")
    print(f"False Positive Rate: {fp/(fp+tn):.4f}")
    print(f"False Negative Rate: {fn/(fn+tp):.4f}")

# 7. 최종 예측 및 평가
threshold = 0.56
y_pred = predict_with_threshold(dt, X_test, threshold)
evaluate_model(y_test, y_pred, "Decision Tree", threshold)

#Random Forest & Catboost Ensemble(train.csv 및 test.csv 동시 사용)

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

#GBM 모델 구현 및 시각화
# Optuna 하이퍼파라미터 튜닝
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100,350),  # 트리 개수 증가
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.15),  # 학습률 감소
        "max_depth": trial.suggest_int("max_depth", 3, 10),  # 트리 깊이 증가
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),  # 노드 분할 최소 샘플 수
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 9),  # 리프 노드 최소 샘플 수
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),  # 일부 데이터 샘플링
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }

    model = GradientBoostingClassifier(**params, random_state=42)
    model.fit(X_train_res, y_train_res)
    y_prob = model.predict_proba(X_test)[:, 1]

    best_recall = 0
    best_threshold = 0.5

    for threshold in np.arange(0.1, 0.9, 0.02):
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
study.optimize(objective, n_trials=20, n_jobs=-1, timeout=300)

#최적 threshold 값 가져오기
best_threshold = study.best_trial.user_attrs["best_threshold"]

# 최적 모델 학습 및 평가
best_params = study.best_trial.params
best_model = GradientBoostingClassifier(**best_params, random_state=42)
best_model.fit(X_train_res, y_train_res)

y_prob_test = best_model.predict_proba(X_test)[:, 1]

# 최종 평가
y_test_pred_final = (y_prob_test >= best_threshold).astype(int)

print("최적 하이퍼파라미터:", best_params)
print(f"최적 Threshold: {best_threshold:.4f}")
print("Accuracy:", accuracy_score(y_test, y_test_pred_final))
print("F1 Score:", f1_score(y_test, y_test_pred_final))
print("Recall:", recall_score(y_test, y_test_pred_final))
print("Precision:", precision_score(y_test, y_test_pred_final))
print("Average Precision:", average_precision_score(y_test, y_prob_test))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob_test))

# Feature Importance 출력
importances = best_model.feature_importances_
feature_names = X.columns

# 중요도 순으로 정렬
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Feature Importance 출력
print("Feature Importance :")
print(importance_df)

# Feature Importance 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="Blues_r")
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

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

#--------------------------------------------------------------------
# Average Precision Score 사용
scores = cross_val_score(best_model, X_train_res, y_train_res, cv=5, scoring='average_precision')
print(f"교차 검증 Average Precision 점수 평균: {np.mean(scores):.4f}")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# Optuna로 찾은 최적의 하이퍼파라미터 적용
gbm = GradientBoostingClassifier(
    n_estimators=best_params["n_estimators"],
    learning_rate=best_params["learning_rate"],
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    min_samples_leaf=best_params["min_samples_leaf"],
    subsample=best_params["subsample"],
    max_features=best_params["max_features"],
    random_state=42
)

# 교차 검증을 통한 최종 평가
scores = cross_val_score(gbm, X_train, y_train, cv=5, scoring='roc_auc')

print("각 Fold의 AUC 점수:", scores)
print("평균 AUC 점수:", scores.mean())

# ---------------------------------------------------------------
# 소수 클래스 비율 시각화
sns.countplot(x=y)
plt.title("Class Distribution")
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#------------------------------------------------------------------

# 혼동 행렬 계산
cm = confusion_matrix(y_test, y_test_pred_final)

# 혼동 행렬 시각화
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()

#-----------------------------------------------------------------
# 인코딩 O, 스케일링 O, smote O (스케일링 추가) 2
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler

#모델링 모듈
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, average_precision_score, confusion_matrix
import shap
from imblearn.over_sampling import SMOTE

# 각 모델 정확도를 제외한 모든 수치가 낮아 실사용 불가, optuna를 이용한 튜닝 수행

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

#minmaxscaling - 적용유무 검토
target_features = ["CreditScore", "Age", "Balance", "EstimatedSalary"]
X_train_sc = X_train.copy()
X_test_sc = X_test.copy()

mn_sc = MinMaxScaler()
X_train_sc[target_features] = mn_sc.fit_transform(X_train[target_features])
X_test_sc[target_features] = mn_sc.transform(X_test[target_features]) #테스터는 정규화 학습 안함

#Over Sampling
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_sc, y_train)
print("SMOTE 적용 후 데이터 크기:", X_train_res.shape, y_train_res.shape)
print(f'original : {y_train.value_counts()}')
print(f'smote : {y_train_res.value_counts()}')

print('----------------------------')
# Optuna 하이퍼파라미터 튜닝
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100,350),  # 트리 개수 증가
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.15),  # 학습률 감소
        "max_depth": trial.suggest_int("max_depth", 3, 10),  # 트리 깊이 증가
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),  # 노드 분할 최소 샘플 수
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 9),  # 리프 노드 최소 샘플 수
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),  # 일부 데이터 샘플링
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }

    model = GradientBoostingClassifier(**params, random_state=42)
    model.fit(X_train_res, y_train_res)
    y_prob = model.predict_proba(X_test)[:, 1]

    best_recall = 0
    best_threshold = 0.5

    for threshold in np.arange(0.1, 0.9, 0.02):
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
study.optimize(objective, n_trials=20, n_jobs=-1, timeout=300)

#최적 threshold 값 가져오기
best_threshold = study.best_trial.user_attrs["best_threshold"]

# 최적 모델 학습 및 평가
best_params = study.best_trial.params
best_model = GradientBoostingClassifier(**best_params, random_state=42)
best_model.fit(X_train_res, y_train_res)

y_prob_test = best_model.predict_proba(X_test)[:, 1]

# 최종 평가
y_test_pred_final = (y_prob_test >= best_threshold).astype(int)

print("최적 하이퍼파라미터:", best_params)
print(f"최적 Threshold: {best_threshold:.4f}")
print("Accuracy:", accuracy_score(y_test, y_test_pred_final))
print("F1 Score:", f1_score(y_test, y_test_pred_final))
print("Recall:", recall_score(y_test, y_test_pred_final))
print("Precision:", precision_score(y_test, y_test_pred_final))
print("Average Precision:", average_precision_score(y_test, y_prob_test))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob_test))

# Feature Importance 출력
importances = best_model.feature_importances_
feature_names = X.columns

# 중요도 순으로 정렬
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Feature Importance 출력
print("Feature Importance :")
print(importance_df)

# Feature Importance 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="Blues_r")
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# !pip install shap

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

#SVM 및 KNN 모델

# SVM 모델
import sklearn.svm
clf = sklearn.svm.SVC(kernel='rbf',random_state=42)

clf.fit(X_train_res, y_train_res)

y_pred_svc = clf.predict(X_test_sc)

print("정확도 : ", accuracy_score(y_test, y_pred_svc))
print("f1-score : ", f1_score(y_test, y_pred_svc))
print("recall : ", recall_score(y_test, y_pred_svc))
print("예측도 : ", precision_score(y_test, y_pred_svc))
print("AUC :", average_precision_score(y_test, y_pred_svc))

# KNN 모델, 추가 전처리 없을 시 K=14에서 에러율 제일 작음
error_rate = []

for i in range(1, 16):
    knn_i = KNeighborsClassifier(n_neighbors=i)
    knn_i.fit(X_train_res, y_train_res)
    y_pred_i = knn_i.predict(X_test)
    error_rate.append(np.mean(y_pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 16), error_rate, marker='o', linestyle='dashed', markersize=8, markerfacecolor='red')
plt.title("Error Rate vs. K Value")
plt.xlabel("K Value")
plt.ylabel("Error Rate")
plt.show()

k = 2
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_res, y_train_res)
y_pred_knn = knn.predict(X_test_sc)

print("정확도 : ", accuracy_score(y_test, y_pred_knn))
print("f1-score : ", f1_score(y_test, y_pred_knn))
print("recall : ", recall_score(y_test, y_pred_knn))
print("예측도 : ", precision_score(y_test, y_pred_knn))
print("AUC :", average_precision_score(y_test, y_pred_knn))

# SVC optuna 수행
def objective(trial):

    svc_c = trial.suggest_float("C", 0.01, 100, log=True)
    classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto", random_state=42)

    score = sklearn.model_selection.cross_val_score(classifier_obj, X_train_res, y_train_res, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

    # 최적화된 하이퍼파라미터로 튜닝

svc_c = study.best_params['C']
model = sklearn.svm.SVC(C=svc_c, gamma="auto", random_state=42)
model.fit(X_train_res, y_train_res)

y_pred_svc_op = model.predict(X_test_sc)

print("정확도 : ", accuracy_score(y_test, y_pred_svc_op))
print("f1-score : ", f1_score(y_test, y_pred_svc_op))
print("recall : ", recall_score(y_test, y_pred_svc_op))
print("예측도 : ", precision_score(y_test, y_pred_svc_op))
print("AUC :", average_precision_score(y_test, y_pred_svc_op))

# KNN 최적화, 목적함수 정의
from sklearn.model_selection import cross_val_score

def obj_knn(trial,x,y):
    n_neighbors = trial.suggest_int('n_neighbors', 1, 30)
    metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])

    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
    score = cross_val_score(model, x, y, n_jobs=-1, cv=10, scoring='f1_macro')
    f1_macro = np.mean(score)

    return f1_macro

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(lambda trial: obj_knn(trial, X_train_res, y_train_res), n_trials = 100)

# 최적 파라미터 출력
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# 최적 파라미터로 모델 학습
best_params = study.best_trial.params
knn_model = KNeighborsClassifier(**best_params)

# bset parmeter training
knn_model.fit(X_train_res,y_train_res)
y_pred_knn_op = knn_model.predict(X_test_sc)

print("정확도 : ", accuracy_score(y_test, y_pred_knn_op))
print("f1-score : ", f1_score(y_test, y_pred_knn_op))
print("recall : ", recall_score(y_test, y_pred_knn_op))
print("예측도 : ", precision_score(y_test, y_pred_knn_op))
print("AUC :", average_precision_score(y_test, y_pred_knn_op))

# 모델과 하이퍼파라미터 범위 정의
param_grids = {    
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
    },
    'SVM': {
        'model': svm.SVC(random_state=42),
        'params': {
            'C': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100, 500],
            'kernel': ['linear', 'rbf', 'poly', 'sigmod'],
            'gamma': ['auto', 'scale']
        }
    },             
}

# 결과를 저장
results = {}

# 각 모델별 학습 및 평가
from sklearn.model_selection import cross_val_score, GridSearchCV
for name, model_info in param_grids.items():
    print(f"\n{name} 모델 학습 중...")

    # GridSearchCV를 통한 하이퍼파라미터 튜닝
    grid_search = GridSearchCV(
        model_info['model'],
        model_info['params'],
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    # 모델 학습
    grid_search.fit(X_train_res, y_train_res)

    # 5-fold cv
    cv_scores = cross_val_score(grid_search.best_estimator_, X_train_res, y_train_res, cv=5)

    # prediction
    y_pred = grid_search.predict(X_test_sc)

    # 평가 지표 계산
    results[name] = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_scores_mean': cv_scores.mean(),
        'cv_scores_std': cv_scores.std(),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'avg_precision': average_precision_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

# 결과 출력
for name, result in results.items():
    print(f"\n=== {name} 모델 결과 ===")
    print(f"Best Parameters: {result['best_params']}")
    print(f"Best CV Score: {result['best_score']:.4f}")
    print(f"5-fold CV Score: {result['cv_scores_mean']:.4f} (+/- {result['cv_scores_std'] * 2:.4f})")
    print(f"Test Accuracy: {result['accuracy']:.4f}")
    print(f"F1 Score: {result['f1']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Average Precision: {result['avg_precision']:.4f}")
    print("\nConfusion Matrix:")
    print(result['confusion_matrix'])

# LightGBM 모델

# 4. 모델 학습 (최적화된 파라미터 적용)
lgbm = LGBMClassifier(
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.01,
    num_leaves=50,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=4.0,
    reg_alpha=1e-5,
    reg_lambda=1e-5,
    random_state=42,
    boosting_type='gbdt',
    objective='binary'
)
lgbm.fit(X_train_res, y_train_res)

# 5. 예측 및 threshold 적용
def predict_with_threshold(model, X, threshold=0.69):
    y_pred_proba = model.predict_proba(X)[:, 1]
    return (y_pred_proba >= threshold).astype(int)

# 6. 모델 평가
def evaluate_model(y_true, y_pred, model_name, threshold):
    print(f"\n{model_name} 성능 (threshold={threshold:.3f}):")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_pred):.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\nSpecificity: {tn/(tn+fp):.4f}")
    print(f"False Positive Rate: {fp/(fp+tn):.4f}")
    print(f"False Negative Rate: {fn/(fn+tp):.4f}")

# 7. 최종 예측 및 평가
threshold = 0.69  # 이전 최적 threshold
y_pred = predict_with_threshold(lgbm, X_test, threshold)
evaluate_model(y_test, y_pred, "LightGBM", threshold)

# Random Forest 모델
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

# K-Means Clustering 수행 및 시각화

X_cluster = df[['Age', 'NumOfProducts', 'IsActiveMember']]

# 각 K값에 대한 오차제곱합 계산 및 시각화
SSE = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_cluster)
    SSE.append(kmeans.inertia_)

plt.plot(range(1,11), SSE, marker='o')
plt.xlabel('value of K')
plt.ylabel('inertia(SSE)')
plt.show()

diff = np.diff(SSE)
optimal_k = np.argmin(diff[1:] - diff[:1]) +2
print(optimal_k)

# 경사가 가장 완만해지는 K값 = 2이나 그룹 간 특징을 산출하기 위해 k=3 사용, 1,2번 그룹이 이탈률이 상대적으로 높음
kmeans = KMeans(n_clusters=4, random_state=42)
df_filtered['Cluster'] = kmeans.fit_predict(X_cluster)
df_filtered[['Age', 'NumOfProducts','IsActiveMember' ,'Cluster']].head()

cluster_analysis = df_filtered.groupby('Cluster')['Exited'].mean()
print(cluster_analysis)

cluster_summery = df_filtered.groupby('Cluster').mean(numeric_only=True).drop(['id', 'CustomerId'], axis=1)
cluster_summery

cluster_summery_mode = df_filtered.groupby('Cluster').agg({
    'NumOfProducts' : lambda x : x.mode()[0],
    'HasCrCard' : lambda x : x.mode()[0],
    'IsActiveMember' : lambda x : x.mode()[0]
    })
cluster_summery_mode

# 산점도를 통한 군집 시각화
selected_features = ['Age', 'NumOfProducts', 'IsActiveMember']
sns.pairplot(df_filtered, vars = selected_features ,hue="Cluster", palette="viridis", diag_kind="kde")
plt.suptitle("K-Means Clustering Results", y=1.02)
plt.show()

# 3D 산점도 시각화
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 클러스터별 색상 설정
scatter = ax.scatter(df_filtered["Age"], df_filtered["IsActiveMember"], df_filtered["NumOfProducts"],
                     c=df_filtered["Cluster"], cmap="viridis", s=50, alpha=0.7, edgecolors='k')

# 축 라벨 설정
ax.set_xlabel("Age")
ax.set_ylabel("IsActiveMember")
ax.set_zlabel("NumOfProducts")
ax.set_title("3D Scatter Plot of K-Means Clusters")

# 컬러바 추가
plt.colorbar(scatter, ax=ax, label="Cluster")

# 그래프 표시
plt.show()

# 클러스터별 Age 분포
plt.figure(figsize=(8, 5))
sns.boxplot(x="Cluster", y="Age", data=df_filtered)
plt.title("Age Distribution by Cluster")
plt.show()

# 클러스터별 NumOfProducts 분포
plt.figure(figsize=(8, 5))
sns.barplot(x="Cluster", y="NumOfProducts", data=df_filtered)
plt.title("NumOfProducts Distribution by Cluster")
plt.show()

# 클러스터별 IsActiveMember 분포
plt.figure(figsize=(8, 5))
sns.barplot(x="Cluster", y="IsActiveMember", data=df_filtered)
plt.title("IsActiveMember Distribution by Cluster")
plt.show()