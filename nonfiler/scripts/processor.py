import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def preprocess(df: pd.DataFrame):

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
    data_numeric_raw = df.drop(
        columns=targets_to_exclude, errors="ignore"
    ).select_dtypes(include=[np.number])

    # 2️⃣ Select categoricals and drop targets
    categorical_raw = df.select_dtypes(exclude=[np.number]).drop(
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
    y_raw = df["filer_status"].map({"non filer": 0, "filer": 1}).values


    return X_raw, y_raw
