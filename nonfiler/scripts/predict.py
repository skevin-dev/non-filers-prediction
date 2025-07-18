import pandas as pd
import numpy as np
import joblib

def predict_filer(data: pd.DataFrame) -> pd.Series:
    """
    Loads the trained model and encoder, applies them to the input data,
    and returns filer predictions (0 for non-filer, 1 for filer).
    """
    # Load model and encoder
    model = joblib.load("xgb_filer_model.joblib")
    encoder = joblib.load("encoder_filer.joblib")

    # Drop irrelevant columns
    drop_cols = [
        'filer_status', 'filer_status_encoded', 'doc_no', 'close_date',
        'reg_date', 'year', 'month', 'reg_year', 'closed_year',
        'district_label', 'tax_type_label'
    ]
    data_numeric = data.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])

  


    # Predict
    predictions = model.predict(data_numeric)
    return pd.Series(predictions, index=data.index, name="prediction")