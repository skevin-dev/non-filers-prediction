import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Mode Imputation for Categorical Fields
    mode_impute_cols = ['month', 'tax_centre_no', 'district_no', 'seg_no', 'tp_size']
    for col in mode_impute_cols:
        if col in df.columns:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)

    # 2. Mean Imputation for Continuous Numeric Field
    if 'tax_type_no_count' in df.columns:
        df['tax_type_no_count'] = df['tax_type_no_count'].fillna(df['tax_type_no_count'].mean())

    # 3. Float Fields: Add existence indicator + Fill missing with 0
    float_columns = df.select_dtypes(include=['float64']).columns.tolist()
    for col in float_columns:
        df[f"{col}_Exist"] = df[col].notnull().map({True: 'Yes', False: 'No'})
        df[col] = df[col].fillna(0)

    # 4. Handle `close_date`
    if 'close_date' in df.columns:
        df['closedate_Exist'] = df['close_date'].notnull().map({True: 'Yes', False: 'No'})
        tomorrow = pd.Timestamp.today() + pd.Timedelta(days=1)
        df['close_date'] = df['close_date'].fillna(tomorrow)

    # 5. Handle `reg_date`
    if 'reg_date' in df.columns:
        df['reg_date'] = pd.to_datetime(df['reg_date'], errors='coerce')
        df = df.dropna(subset=['reg_date'])

    # 6. Data type conversions
    if 'tax_type_no_count' in df.columns:
        df['tax_type_no_count'] = df['tax_type_no_count'].astype('int64')

    columns_to_object = ['month', 'seg_no', 'tax_type_no', 'district_no', 'tax_centre_no']
    for col in columns_to_object:
        if col in df.columns:
            df[col] = df[col].astype('object')

    return df