from nonfiler.scripts import load_data, feature_engineering, feature_selection, train, predict, processor

def main():

    filepath = "~/non-filers-prediction/nonfiler/data/data.csv"
    # Step 1: Load data
    df = load_data.load_data(path=filepath)
 
    df = processor.preprocess(df=df)

    # Step 2: Feature engineering
    df = feature_engineering.engineer_features(df=df)

    train.train_and_save_model(data=df,model_path="/home/shy2351783/non-filers-prediction/nonfiler/models/")

if __name__ == "__main__":
    main()


