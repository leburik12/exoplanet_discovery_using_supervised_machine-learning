# preprocessor = ExoplanetDataPreprocessor(exoplanet_df, target_column="habitable")
# processed_df = preprocessor.fit_transform(exoplanet_df)

# org_df, feature_pairs = get_corresponding_original_features(processed_df, exoplanet_df)
# plot_transformation_effects(org_df, processed_df, feature_pairs)
# updated_df = preprocessor.reintegrate_processed_features_into_original_df(
#     processed_df, exoplanet_df
# )

# eng = ExoplanetFeatureEngineer()
# eng.pipeline = eng._create_sk_pipeline()
# df_features = eng.pipeline.transform(updated_df)


# # Initialize IterativeImputer (MICE) with BayesianRidge as default estimator
# imputer = IterativeImputer(
#     random_state=42,
#     max_iter=20,
#     sample_posterior=True,  # Important for MICE to sample from posterior
#     initial_strategy="median",
#     imputation_order="ascending",  # Order of feature imputation, can be 'ascending', 'descending', or 'random'
# )

# num_cols = df_features.select_dtypes(include=["float64", "int64"]).columns.tolist()
# cat_cols = df_features.select_dtypes(include=["object", "category"]).columns.tolist()

# # Fit OrdinalEncoder on categorical columns (handle unknown categories gracefully)
# ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

# df_cat_encoded = pd.DataFrame(
#     ordinal_encoder.fit_transform(df_features[cat_cols]), columns=cat_cols
# )

# df_num = df_features[num_cols].copy()
# df_for_imputation = pd.concat([df_num, df_cat_encoded], axis=1)

# # Fit imputer on df_for_imputation
# X_imputed = imputer.fit_transform(df_for_imputation)

# X_imputed_df = pd.DataFrame(X_imputed, columns=df_for_imputation.columns)

# X = X_imputed_df.drop(columns=["is_habitable"])
# y = X_imputed_df["is_habitable"]

# # Set up stratified k-fold cross-validation
# cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# xgboost_model = XGBoostTrainer(scale_pos_weight=1, calibrator="sigmoid", cv=5)

# # Initialize your KNN model
# knn_model = KNNTrainer()
# # xgboost_model = XGBoostTrainer(calibrated=True, calibrator='sigmoid', cv=5)

# # Create a models dictionary (required by ModelEvaluator)
# models = {
#     # 'knn': knn_model,
#     "xgboost": xgboost_model
# }

# # Initialize ModelEvaluator with your data and models
# evaluator = ModelEvaluator(models=models, X=X, y=y)

# # Run the evaluation
# evaluator.evaluate_all(cv)


# print("\n--- Running ExoplanetStatisticalExplorer ---")
# if exoplanet_df is not None:
#     explorer = ExoplanetStatisticalExplorer(exoplanet_df)
#     # runs _distribution_analysis + _correlation_structure
#     explorer.explore()

#     # # transform_config = explorer.preprocessing_recommendations

#     # print("\n--- Statistical Exploration Results ---")
#     # print("Distribution Analysis Results (partial):")
#     # for col, res in explorer.dist_results.items():
#     #   print(f" {col}: Skew={res['skewness']:.2f}, Kurtosis={res['kurtosis']:.2f}")

#     # print("\nCorrelation Results:")
#     # print("Pearson Correlation (Exerpt):\n", explorer.correlation_results['pearson'].head())

#     preprocessor = ExoplanetDataPreprocessor(exoplanet_df, target_column="habitable")
#     processed_df = preprocessor.fit_transform(exoplanet_df)

#     print("Processed shape:", processed_df.shape)
#     print("Sample rows:")
#     display(processed_df.head())

#     print(f"Data-Type - {type(processed_df)}")
#     print(f"Processed-DF-Columns -  {processed_df.columns}")
#     print(f"Data-Type for the column - {processed_df['log_features__pl_rade'].dtype}")

#     # Feature schema
#     # for col in processed_df.columns:
#     #   print(f"{col} -> dtype: {processed_df[col].dtype},"
#     #         f"min: {processed_df[col].min():.3f}, max: {processed_df[col].max():.3f}")

#     sns.boxplot(data=processed_df)
#     plt.xticks(rotation=90)
#     plt.title("Distribution of Processed Features")
#     plt.show()

# org_df, feature_pairs = get_corresponding_original_features(processed_df, exoplanet_df)
# plot_transformation_effects(org_df, processed_df, feature_pairs)
# updated_df = preprocessor.reintegrate_processed_features_into_original_df(
#     processed_df, exoplanet_df
# )
# updated_df.head(20)
# updated_df.columns


# eng = ExoplanetFeatureEngineer()
# eng.pipeline = eng._create_sk_pipeline()
# df_features = eng.pipeline.transform(updated_df)

# print(f"\n{df_features.columns.tolist()}")
# df_features.head(9)
# df_features["is_habitable"] = (
#     (df_features["in_hz"] == 1) & (df_features["is_terrestrial"] == 1)
# ).astype(int)
# num_in_hz = ((df_features["in_hz"] == 1) & (df_features["is_terrestrial"] == 1)).sum()

# X = df_features.drop(columns=["is_habitable"])
# y = df_features["is_habitable"]
# print(f"Habitable -- {(y == 1).sum()}")
# print(f"Number of exoplanets in the habitable zone : {num_in_hz}")
