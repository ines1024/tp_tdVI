#simple model

import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

# Load the competition data
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
X_train = X_train.select_dtypes(include='number')
# del train_data
# gc.collect()

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=161828)

cls = make_pipeline(SimpleImputer(), DecisionTreeClassifier(max_depth=8, random_state=161828))
cls.fit(X_train, y_train)

#AUC
y_preds_val_prob = cls.predict_proba(X_test)[:,1]
roc_auc = roc_auc_score(y_test, y_preds_val_prob)
print("ROC AUC Score", roc_auc)

# Predict on the evaluation set
eval_data = eval_data.drop(columns=["conversion"])
eval_data = eval_data.select_dtypes(include='number')
y_preds = cls.predict_proba(eval_data.drop(columns=["ROW_ID"]))[:, cls.classes_ == 1].squeeze()

# Make the submission file
submission_df = pd.DataFrame({"ROW_ID": eval_data["ROW_ID"], "conversion": y_preds})
submission_df["ROW_ID"] = submission_df["ROW_ID"].astype(int)
submission_df.to_csv("basic_model.csv", sep=",", index=False)

# #ROC AUC Score 0.7366348698176497