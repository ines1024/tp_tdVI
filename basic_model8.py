# xgb con busqueda de hiperparametros y ohe agregando de a una variable
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


# Load the competition data
comp_data = pd.read_csv("competition_data.csv")

# Split into training and evaluation samples
train_data = comp_data[comp_data["ROW_ID"].isna()]
eval_data = comp_data[comp_data["ROW_ID"].notna()]
del comp_data
gc.collect()

df_categoricas = pd.DataFrame()
for columna in train_data.columns:
    if train_data[columna].dtype == 'object':
        # Mover la columna categórica al nuevo DataFrame
        df_categoricas[columna] = train_data[columna]
        # Dropear la columna del DataFrame original
        # train_data.drop(columna, axis=1, inplace=True)

label_encoder = LabelEncoder()
for columna in df_categoricas.columns:
    df_categoricas[columna] = label_encoder.fit_transform(df_categoricas[columna])

X_new_cat = SelectKBest(mutual_info_classif, k=5).fit_transform(df_categoricas, train_data["conversion"])
selected_feature_indices = SelectKBest(mutual_info_classif, k=5).fit(df_categoricas, train_data["conversion"]).get_support()

selected_column_names = df_categoricas.columns[selected_feature_indices]
selected_categorical = pd.DataFrame(X_new_cat, columns=selected_column_names)
selected_categorical.to_csv("selected_categorical_features.csv", sep=",", index=False)


# Train a random forest model on the train data
# train_data_ = train_data.sample(frac=1/3)
y_train = train_data["conversion"]
X_train = train_data.drop(columns=["conversion", "ROW_ID"])
X_train = X_train.select_dtypes(include='number')

X_train = pd.concat([X_train, selected_categorical], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=161828)

# del train_data
# gc.collect()

params = {'colsample_bytree': 0.75,
        'gamma': 0.3,
        'learning_rate': 0.075,
        'max_depth': 8,
        'min_child_weight': 10,
        'n_estimators': 300,
        'reg_lambda': 0.5,
        'subsample': 0.75,
        }

clf_xgb = xgb.XGBClassifier(objective = 'binary:logistic',
                            seed = 161828,
                            eval_metric = 'auc',
                            **params)

clf_xgb.fit(X_train, y_train, verbose = 100, eval_set = [(X_test, y_test)])
preds_test_xgb = clf_xgb.predict_proba(X_test)[:,1]
roc_auc = roc_auc_score(y_test, preds_test_xgb)

#predict on the evaluation set

eval_data = eval_data.drop(columns=["conversion"])
selected_categorical_eval_data = eval_data[selected_column_names].copy()

for columna in selected_categorical_eval_data:
    selected_categorical_eval_data[columna] = label_encoder.fit_transform(selected_categorical_eval_data[columna])

eval_data = eval_data.select_dtypes(include='number')

eval_data = pd.concat([eval_data, selected_categorical_eval_data], axis=1)

y_preds = clf_xgb.predict_proba(eval_data.drop(columns=["ROW_ID"]))[:, clf_xgb.classes_ == 1].squeeze()

# Make the submission file
submission_df = pd.DataFrame({"ROW_ID": eval_data["ROW_ID"], "conversion":y_preds})
submission_df["ROW_ID"] = submission_df["ROW_ID"].astype(int)
submission_df.to_csv("basic_model_kbest_xgb_.csv", sep=",", index=False)

#AUC test score - XGBoost con busqueda de hiperparametros: 

