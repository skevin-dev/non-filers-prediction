import pandas as pd


def drop_highly_correlated_features(
    data, target_column=None, high_threshold=0.85
) -> pd.DataFrame:
    """
    Drop features that are highly correlated with each other.

    Args:
        data (pd.DataFrame): Input DataFrame.
        target_column (str, optional): If provided, retain features more correlated with this target.
        high_threshold (float): Threshold for high correlation.

    Returns:
        pd.DataFrame: DataFrame with dropped features removed.
        list: List of dropped features.
    """
    numerical_data = data.select_dtypes(include=["float64", "int64"])

    # Exclude target from correlation matrix if provided
    if target_column:
        corr_matrix = numerical_data.drop(
            columns=[target_column], errors="ignore"
        ).corr()
    else:
        corr_matrix = numerical_data.corr()

    corr_pairs = corr_matrix.unstack()
    corr_pairs = corr_pairs[
        corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)
    ]

    very_strong_corr = corr_pairs[abs(corr_pairs) > high_threshold].sort_values(
        ascending=False
    )

    features_to_drop_high = set()
    already_seen_high = set()

    # Correlation with target if needed
    corr_with_target = {}
    if target_column:
        corr_with_target = numerical_data.corrwith(
            numerical_data[target_column]
        ).to_dict()

    for feature1, feature2 in very_strong_corr.index:
        if feature1 not in already_seen_high and feature2 not in already_seen_high:
            corr1 = abs(corr_with_target.get(feature1, 0))
            corr2 = abs(corr_with_target.get(feature2, 0))
            if corr1 >= corr2:
                features_to_drop_high.add(feature2)
            else:
                features_to_drop_high.add(feature1)
            already_seen_high.update([feature1, feature2])

    cleaned_data = numerical_data.drop(columns=features_to_drop_high, errors="ignore")

    return cleaned_data
