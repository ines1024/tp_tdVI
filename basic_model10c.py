import pandas as pd
from scipy.stats import uniform
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
import gc
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack


comp_data = pd.read_csv("competition_data.csv")

# Split into training and evaluation samples
train_data = comp_data[comp_data["ROW_ID"].isna()]
eval_data = comp_data[comp_data["ROW_ID"].notna()]
del comp_data
gc.collect()

# Train a random forest model on the train data
# train_data_ = train_data.sample(frac=1/3)
y_train = train_data["conversion"]
X_train = train_data.drop(columns=["conversion", "ROW_ID"])
# del train_data
# gc.collect()

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=161828)

X_train_obj = X_train.select_dtypes(include=["object"])
X_train_num = X_train.select_dtypes(include="number")


encoder = OneHotEncoder(sparse_output=True, handle_unknown="infrequent_if_exist")
X_train_ohe_obj = encoder.fit_transform(X_train_obj)
X_train_ohe = hstack([X_train_ohe_obj, X_train_num])

# Sobre validation
X_test_obj = X_test.select_dtypes(include=["object"])
X_test_num = X_test.select_dtypes(include="number")


X_test_ohe_obj = encoder.transform(X_test_obj)
X_test_ohe = hstack([X_test_ohe_obj, X_test_num])

params = {'colsample_bytree': 0.75,
        'gamma': 0.3,
        'learning_rate': 0.075,
        'max_depth': 8,
        'min_child_weight': 10,
        'n_estimators': 300,
        'reg_lambda': 0.5,
        'subsample': 0.75,
        }

clf_xgb = xgb.XGBClassifier(objective="binary:logistic", seed=160702, eval_metric="auc")
clf_xgb.fit(X_train_ohe, y_train)

#AUC
y_preds_val_prob = clf_xgb.predict_proba(X_test_ohe)[:,1]
roc_auc = roc_auc_score(y_test, y_preds_val_prob)
print("ROC AUC Score", roc_auc)

# Predict on the evaluation set


eval_data = eval_data.drop(columns=["conversion"])
eval_obj = eval_data.select_dtypes(include= ["object"])
eval_num = eval_data.select_dtypes(include='number')
eval_num_sin_RowID = eval_num.drop(columns=["ROW_ID"])
eval_ohe_obj = encoder.transform(eval_obj)

eval_ohe = hstack([eval_ohe_obj, eval_num_sin_RowID])

y_preds = clf_xgb.predict_proba(eval_ohe)[:, clf_xgb.classes_ == 1].squeeze()

# Make the submission file
submission_df = pd.DataFrame({"ROW_ID": eval_data["ROW_ID"], "conversion": y_preds})
submission_df["ROW_ID"] = submission_df["ROW_ID"].astype(int)
submission_df.to_csv("basic_model_ohe_sin_escalar.csv", sep=",", index=False)