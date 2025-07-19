# train.py

import joblib
import numpy as np
import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier


def train_and_save_model(X_raw, y_raw, model_path="models/xgb_classifier.pkl"):


    
    # 7️⃣ Train-test split
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )

    # 8️⃣ Balance with SMOTE
    sm = SMOTE(random_state=42)
    X_train_bal_raw, y_train_bal_raw = sm.fit_resample(X_train_raw, y_train_raw)

    # 9️⃣ Train XGBoost model
    model = XGBClassifier(
        colsample_bytree=0.9779976597381381,
        learning_rate=0.22818159875692626,
        max_depth=9,
        n_estimators=299,
        subsample=0.8711331923216198,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )


    feature_names = list(X_train_bal_raw.columns)  
   

    model.fit(X_train_bal_raw, y_train_bal_raw)

    accuracy_train = accuracy_score(y_train_raw, model.predict(X_train_raw))
    accuracy_test = accuracy_score(y_test_raw, model.predict(X_test_raw))

    # save features
    joblib.dump(feature_names, os.path.join(model_path, "feature_names.pkl"))

    print(
        f"✅ features saved to {model_path} successfully"
    )

    # 🔟 Save model
    joblib.dump(model, os.path.join(model_path, "xgb_model.pkl"))

    print(
        f"✅ Model saved to {model_path} with training accuracy of {accuracy_train} and testing accuracy of {accuracy_test}"
    )

    print ("----------------------- training classification report ------------------------------------")

    print(classification_report(y_train_raw, model.predict(X_train_raw), target_names=['non filer', 'filer']))

    print ("----------------------- testing classification report ------------------------------------")

    print(classification_report(y_test_raw, model.predict(X_test_raw), target_names=['non filer', 'filer']))


