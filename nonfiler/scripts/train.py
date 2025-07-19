# train.py

import joblib
import numpy as np
import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


def train_and_save_model(data: pd.DataFrame, model_path="models/xgb_classifier.pkl"):

    targets_to_exclude = [
        "filer_status",
        "filer_status_encoded",
        "doc_no",
        "close_date",
        "reg_date",
        "year",
        "month",
        "reg_year",
        "closed_year",
        "district_label",
        "tax_type_label",
    ]

    # 1️⃣ Keep only numeric (dropping targets)
    data_numeric_raw = data.drop(
        columns=targets_to_exclude, errors="ignore"
    ).select_dtypes(include=[np.number])

    # 2️⃣ Select categoricals and drop targets
    categorical_raw = data.select_dtypes(exclude=[np.number]).drop(
        columns=targets_to_exclude, errors="ignore"
    )

    # 3️⃣ Encode categoricals
    categorical_raw = categorical_raw.astype(str)
    encoder_raw = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    categorical_encoded_raw = pd.DataFrame(
        encoder_raw.fit_transform(categorical_raw),
        columns=encoder_raw.get_feature_names_out(categorical_raw.columns),
        index=categorical_raw.index,
    )

    # 4️⃣ Combine numeric + categorical encoded
    X_raw = pd.concat([data_numeric_raw, categorical_encoded_raw], axis=1)

    # 5️⃣ Remove all columns ending with '_encoded' or '_label'
    X_raw = X_raw.loc[:, ~X_raw.columns.str.endswith(("_encoded", "_label"))]

    # 6️⃣ Prepare target
    y_raw = data["filer_status"].map({"non filer": 0, "filer": 1}).values

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

    model.fit(X_train_bal_raw, y_train_bal_raw)

    accuracy_train = accuracy_score(y_train_raw, model.predict(X_train_raw))
    accuracy_test = accuracy_score(y_test_raw, model.predict(X_test_raw))

    # 🔟 Save model
    joblib.dump(model, os.path.join(model_path, "xgb_model.pkl"))

    print(
        f"✅ Model saved to {model_path} with training accuracy of {accuracy_train} and testing accuracy of {accuracy_test}"
    )

    print ("----------------------- training classification report ------------------------------------")

    print(classification_report(y_train_raw, model.predict(X_train_raw), target_names=['non filer', 'filer']))

    print ("----------------------- testing classification report ------------------------------------")

    print(classification_report(y_test_raw, model.predict(X_test_raw), target_names=['non filer', 'filer']))


