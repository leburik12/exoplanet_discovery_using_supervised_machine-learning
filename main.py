import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold

from src.data.load_data import ExoplanetDataLoader
from src.data.preprocess import ExoplanetDataPreprocessor
from src.data.statisticalexplorer import ExoplanetStatisticalExplorer
from src.features.exoplanet_engineer import ExoplanetFeatureEngineer
from src.models.xgboost_trainer import XGBoostTrainer
from src.models.train import ModelEvaluator


def main():
    file_path = "data/nasa_1.csv"

    loader = ExoplanetDataLoader(file_path)
    exoplanet_df = loader.load()
    loader.summarize()

    print("\n--- Running ExoplanetStatisticalExplorer ---")
    explorer = ExoplanetStatisticalExplorer(exoplanet_df)
    explorer.explore()

    eng = ExoplanetFeatureEngineer()
    df_features = eng.transform(exoplanet_df)
    print(f"Features Column names: \n{list(df_features.columns)}")

    preprocessor = ExoplanetDataPreprocessor(df_features, target_column="is_habitable")
    processed_df = preprocessor.fit_transform(df_features)
    print(f"Processed Column names: \n{list(processed_df.columns)}")

    # Split into categorical and numerical columns
    num_cols = processed_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = processed_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Encode categorical columns
    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )
    df_cat_encoded = pd.DataFrame(
        ordinal_encoder.fit_transform(processed_df[cat_cols]), columns=cat_cols
    )
    df_num = processed_df[num_cols].copy()
    df_for_imputation = pd.concat([df_num, df_cat_encoded], axis=1)

    # Impute missing values
    imputer = IterativeImputer(
        random_state=42, max_iter=20, sample_posterior=True, initial_strategy="median"
    )
    X_imputed = imputer.fit_transform(df_for_imputation)
    X_imputed_df = pd.DataFrame(X_imputed, columns=df_for_imputation.columns)

    X = X_imputed_df.drop(columns=["is_habitable"])
    y = X_imputed_df["is_habitable"]

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Define models
    xgboost_model = XGBoostTrainer(scale_pos_weight=1, calibrator="sigmoid", cv=5)
    # knn_model = KNNTrainer()

    models = {
        "xgboost": xgboost_model,
        # 'knn': knn_model
    }

    evaluator = ModelEvaluator(models=models, X=X, y=y)
    evaluator.evaluate_all(cv)
    evaluator.plot_roc_curves()
    evaluator.plot_precision_recall_curves()


if __name__ == "__main__":
    main()
