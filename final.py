import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# 모델
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 평가
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, average_precision_score, \
    confusion_matrix
import shap

#1. data_load
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
data = pd.read_csv('Churn_Modelling.csv',index_col=0)

#2. data_preprocessing
# missing D(4개 행 제거)
data.isnull().sum()
nan_data = data.dropna()
print('결측 처리 :' ,nan_data.shape)

# duplicated D(2행 제거)
nan_data[nan_data.duplicated()]
new_data = nan_data[~nan_data.duplicated()]
print('중복 처리 :',new_data.shape)

#encoding : gender-label ( Female = 0, male = 1) /  geography-onehot
le = LabelEncoder()
new_data['Gender'] = le.fit_transform(new_data["Gender"])

oe = OneHotEncoder()
oe.fit(new_data[['Geography']])
geo_csr = oe.transform(new_data[['Geography']])
csr_df = pd.DataFrame(geo_csr.toarray(), columns = oe.get_feature_names_out())
df = new_data.reset_index(drop=True)  # df 인덱스 초기화
csr_df = csr_df.reset_index(drop=True)  # csr_df 인덱스 초기화
inco_df = pd.concat([df,csr_df],axis=1)

#check
int_data = inco_df.drop(columns=['CustomerId', 'Surname','Geography'])
X = int_data.drop("Exited", axis=1)
y_true = int_data['Exited']
print('전처리 완료:',X.shape,y_true.shape)
print('----------------------------')

#4.data engineering
#tester split
X_train, X_test, y_train, y_test = train_test_split(X,y_true,stratify = y_true,test_size = 0.3, random_state= 42)
print('데이터 분리 후 크기 : ',X_train.shape, X_test.shape, y_train.shape, y_test.shape)

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


# 모델과 하이퍼파라미터 범위 정의
param_grids = {
    'Logistic Regression': {
        'model': LogisticRegression(),
        'params': {
            'C': [ ],
            'penalty': [ ],
            'solver': [ ]
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'max_depth': [ ],
            'min_samples_split': [ ],
            'min_samples_leaf': [ ]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [ ],
            'max_depth': [ ],
            'min_samples_split': [ ],
            'min_samples_leaf': [ ]
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [ ],
            'weights': [ ],
            'metric': [ ]
        }
    },
    'SVM': {
        'model': SVC(random_state=42),
        'params': {
            'C': [ ],
            'kernel': [ ],
            'gamma': [ ]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [ ],
            'learning_rate': [ ],
            'max_depth': [ ]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42),
        'params': {
            'n_estimators': [ ],
            'max_depth': [ ],
            'learning_rate': [ ]
        }
    },
    'LightGBM': {
        'model': LGBMClassifier(random_state=42),
        'params': {
            'n_estimators': [ ],
            'max_depth': [ ],
            'learning_rate': [ ]
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(random_state=42, verbose=False),
        'params': {
            'learning_rate': [0.087],  
            'depth': [8],  
            'n_estimators': [145],
            'l2_leaf_reg': [0.07]
        }
    },
}

# 결과를 저장
results = {}

# 각 모델별 학습 및 평가
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