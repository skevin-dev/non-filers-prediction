from fastapi import FastAPI, UploadFile, File
import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
import joblib
from nonfiler.scripts import feature_engineering, processor, data_cleaning 

description = """
This endpoint is intended to predict non-filing status of taxpayers 

"""
app = FastAPI(title="Backend Endpoint of non-filers prediction",description=description)

load_dotenv()

# Get model directory
MODEL_DIR = os.getenv("MODEL_DIR")

# Construct full paths
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")

# Load model and features
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read CSV
    df = pd.read_csv(file.file)


    # Keep a copy of 'tin' for output
    doc_nos = df['doc_no'].tolist()

    # Preprocess the data
    df_clean = data_cleaning.clean_data(df=df)

    df_feature = feature_engineering.engineer_features(df=df_clean)

    X_raw , _ = processor.preprocess(df=df_feature)

    # Drop irrelevant columns
    data_numeric = X_raw[features]

    # Do prediction
    preds = model.predict(data_numeric)

    label_map = {1: "filer", 0: "non-filer"}
    labeled_preds = [label_map.get(pred, "unknown") for pred in preds]

    # Combine TINs and predictions
    result = dict(zip(doc_nos, labeled_preds))

    return result