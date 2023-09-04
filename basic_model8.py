# xgb con busqueda de hiperparametros y ohe agregando de a una variable

import pandas as pd
from scipy.stats import uniform
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import xgboost as xgb
from sklearn.model_selection import ParameterSampler
import gc

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
print(df_categoricas)

# # Train a random forest model on the train data
# # train_data_ = train_data.sample(frac=1/3)
# y_train = train_data["conversion"]
# X_train = train_data.drop(columns=["conversion", "ROW_ID"])
# X_train = X_train.select_dtypes(include='number')
# # del train_data
# # gc.collect()


# best_auc = 0
# best_X_train = pd.DataFrame()
# categorica_agregada = ''

# for column in df_categoricas.columns:
#     X_train_temp = pd.concat([X_train, df_categoricas[column]], axis=1)
#     pd_ohe = pd.get_dummies(X_train_temp,
#                             columns = [column],
#                             sparse = True,    # Devolver una matriz rala.
#                             dummy_na = False, # No agregar columna para NaNs.
#                             dtype = int       # XGBoost no trabaja con 'object'; necesitamos que sean numéricos.
#                        )
  
#     X_train_1, X_test, y_train, y_test = train_test_split(pd_ohe, y_train, test_size=0.3, random_state=161828)

#     params = {'colsample_bytree': 0.75,
#             'gamma': 0.3,
#             'learning_rate': 0.075,
#             'max_depth': 8,
#             'min_child_weight': 10,
#             'n_estimators': 300,
#             'reg_lambda': 0.5,
#             'subsample': 0.75,
#             }

#     clf_xgb = xgb.XGBClassifier(objective = 'binary:logistic',
#                                 seed = 161828,
#                                 eval_metric = 'auc',
#                                 **params)
    
#     clf_xgb.fit(X_train_1, y_train, verbose = 100, eval_set = [(X_test, y_test)])
#     preds_test_xgb = clf_xgb.predict_proba(X_test)[:,1]
#     roc_auc = roc_auc_score(y_test, preds_test_xgb)
#     print(f"AUC test score - XGBoost al agregar {column}: {roc_auc}") 
#     if roc_auc > best_auc:
#         # best_X_train = X_train_1
#         # best_auc = roc_auc
#         # categorica_agregada = column
#         model = clf_xgb
#     X_train_temp.drop(column, axis=1, inplace=True)

# # best_xgb =  xgb.XGBClassifier(objective = 'binary:logistic',
# #                                 seed = 161828,
# #                                 eval_metric = 'auc',
# #                                 **params)
# # best_xgb.fit(best_X_train, y_train, verbose = 100, eval_set = [(X_test, y_test)])

# #predict on the evaluation set
# eval_data = eval_data.drop(columns=["conversion"])
# eval_data = eval_data.select_dtypes(include='number')
# eval_data = pd.concat([eval_data, df_categoricas[categorica_agregada]], axis=1)
# pd_ohe = pd.get_dummies(eval_data,
#                             columns = [categorica_agregada],
#                             sparse = True,    # Devolver una matriz rala.
#                             dummy_na = False, # No agregar columna para NaNs.
#                             dtype = int       # XGBoost no trabaja con 'object'; necesitamos que sean numéricos.
#                        )
# y_preds = model.predict_proba(eval_data.drop(columns=["ROW_ID"]))[:, model.classes_ == 1].squeeze()

# # Make the submission file
# submission_df = pd.DataFrame({"ROW_ID": eval_data["ROW_ID"], "conversion":y_preds})
# submission_df["ROW_ID"] = submission_df["ROW_ID"].astype(int)
# submission_df.to_csv("basic_model_xgb_ohe_de_a_una.csv", sep=",", index=False)


# #AUC test score - XGBoost con busqueda de hiperparametros: 