from nonfiler.scripts import load_data, feature_engineering, train, processor, data_cleaning 
import os
from dotenv import load_dotenv

def main():

     # Load env variables
    load_dotenv()

    # Get paths from .env
    filepath = os.getenv("DATA_PATH")
    model_path = os.getenv("MODEL_DIR")


    # filepath = "~/non-filers-prediction/nonfiler/data/data.csv"
    # Step 1: Load data
    df = load_data.load_data(path=filepath)

    df = data_cleaning.clean_data(df=df)

    # Step 2: Feature engineering
    df = feature_engineering.engineer_features(df=df)

    X_raw , y_raw = processor.preprocess(df=df)

 
    train.train_and_save_model(X_raw=X_raw,y_raw=y_raw,model_path=model_path)

if __name__ == "__main__":
    main()


