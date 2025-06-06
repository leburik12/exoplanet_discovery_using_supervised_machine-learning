import numpy as np
import pandas as pd

from typing import Optional, Dict, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    FunctionTransformer,
    PowerTransformer,
    StandardScaler,
)
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize


class NamedFunctionTransformer(FunctionTransformer):
    def get_feature_names_out(self, input_features=None):
        return input_features


class ExoplanetDataPreprocessor:
    def __init__(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        transform_config: Optional[Dict[str, List[str]]] = None,
    ):
        self.df = df.copy()
        self.target_column = target_column
        self.preprocessing_pipeline: Optional[Pipeline] = None
        self.processed_df: Optional[pd.DataFrame] = None

        self.solar_properties = {
            "teff": 5778,  # K
            "metallicity": 0.0,  # [Fe/H]
            "age": 4.6,  # Gyr
        }

        # Create mapping from processed to original column names
        self.processed_to_original = {
            "log_features__pl_rade": "pl_rade",
            "log_features__pl_radj": "pl_radj",
            "log_features__pl_bmasse": "pl_bmasse",
            "log_features__pl_bmassj": "pl_bmassj",
            "log_features__pl_orbper": "pl_orbper",
            "log_features__pl_orbsmax": "pl_orbsmax",
            "log_features__pl_insol": "pl_insol",
            "log_features__st_teff": "st_teff",
            "log_features__st_rad": "st_rad",
            "log_features__st_mass": "st_mass",
            "log_features__sy_dist": "sy_dist",
            "power_features__pl_eqt": "pl_eqt",
            "power_features__st_met": "st_met",
            "power_features__st_logg": "st_logg",
            "power_features__sy_vmag": "sy_vmag",
            "power_features__sy_kmag": "sy_kmag",
            "power_features__sy_gaiamag": "sy_gaiamag",
        }

        self.log_transform_cols = (
            transform_config.get("log_transform")
            if transform_config and transform_config.get("log_transform") is not None
            else [
                "pl_rade",
                "pl_radj",
                "pl_bmasse",
                "pl_bmassj",
                "pl_orbper",
                "pl_orbsmax",
                "pl_insol",
                "st_teff",
                "st_rad",
                "st_mass",
                "sy_dist",
            ]
        )

        self.power_transform_cols = (
            transform_config.get("power_transform_cols")
            if transform_config
            and transform_config.get("power_transform_cols") is not None
            else [
                "pl_eqt",
                "st_met",
                "st_logg",
                "sy_vmag",
                "sy_kmag",
                "sy_gaiamag",
            ]
        )

        # For winsorization, identify features with extreme kurtosis (from EDA)
        self.winsorize_cols = (
            transform_config.get("winsorize_cols")
            if transform_config and transform_config.get("winsorize_cols") is not None
            else [
                "pl_rade",
                "pl_radj",
                "pl_bmasse",
                "pl_bmassj",
                "pl_orbper",
                "pl_orbsmax",
                "pl_insol",
                "st_teff",
                "st_rad",
                "st_mass",
                "st_met",
                "st_logg",
                "sy_dist",
            ]
        )

        # Filter columns to only include those present in the actual DataFrame
        self.log_transform_cols = [
            c for c in self.log_transform_cols if c in self.df.columns
        ]
        self.power_transform_cols = [
            c for c in self.power_transform_cols if c in self.df.columns
        ]
        self.winsorize_cols = [c for c in self.winsorize_cols if c in self.df.columns]

        # All numerical features that will be processed
        self.numerical_features = list(
            set(
                self.log_transform_cols
                + self.power_transform_cols
                + self.winsorize_cols
            )
        )

        # Remove target column if it's a feature
        if self.target_column and self.target_column in self.numerical_features:
            self.numerical_features.remove(self.target_column)

    def _clean_data(self, df):
        """Astrophysical data cleaning with uncertainty propagation"""
        df_cleaned = df.copy()  # Work on a copy

        # Replace unphysical values and handle extreme outliers
        df_cleaned["pl_rade"] = df_cleaned["pl_rade"].clip(lower=0.1, upper=30)
        df_cleaned["pl_insol"] = df_cleaned["pl_insol"].clip(
            lower=1e-6
        )  # Insolation must be positive

        # Handle missing metallicities with solar metallicity as a domain-informed imputation
        df_cleaned["st_met"] = df_cleaned["st_met"].fillna(
            self.solar_properties["metallicity"]
        )

        # Impute other critical numerical columns with median or a domain-specific value
        # This is a robust approach for a professional pipeline to avoid NaNs downstream
        for col in [
            "pl_bmasse",
            "pl_eqt",
            "pl_orbper",
            "st_mass",
        ]:  # Add more as needed
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

        # Ensure 'hostname' and 'pl_name' are handled for system_complexity
        df_cleaned["hostname"] = df_cleaned["hostname"].fillna("Unknown_System")
        df_cleaned["pl_name"] = df_cleaned["pl_name"].fillna("Unknown_Planet")

        return df_cleaned

    def _winsorize_transformer(
        self, X: np.ndarray, lower_limit: float = 0.01, upper_limit: float = 0.01
    ) -> np.ndarray:
        """
        Applies winsorization column-wise.
        """
        X_winsorized = np.empty_like(X, dtype=float)
        for i in range(X.shape[1]):
            X_winsorized[:, i] = winsorize(
                X[:, i], limits=(lower_limit, upper_limit)
            ).data
        return X_winsorized

    def build_pipeline(self) -> Pipeline:
        # Imputation (Median is robust to outliers/skewness)
        imputer = SimpleImputer(strategy="median")

        # Define transformers for different groups of features
        transformers = []

        # Log Transform + Winsorize + Scale
        if self.log_transform_cols:
            log_pipeline = Pipeline(
                steps=[
                    (
                        "impute",
                        SimpleImputer(strategy="median"),
                    ),  # Impute first if not done globally
                    (
                        "log",
                        NamedFunctionTransformer(np.log1p, validate=False),
                    ),  # log(1+x)
                    (
                        "winsorize",
                        NamedFunctionTransformer(
                            self._winsorize_transformer,
                            validate=False,
                            kw_args={"lower_limit": 0.001, "upper_limit": 0.001},
                        ),
                    ),
                    ("scaler", StandardScaler()),
                ]
            )

            transformers.append(("log_features", log_pipeline, self.log_transform_cols))

        # Power Transform (Yeo-Johnson) + Winsorize + Scale
        if self.power_transform_cols:
            power_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "power",
                        PowerTransformer(method="yeo-johnson", standardize=False),
                    ),
                    (
                        "winsorize",
                        NamedFunctionTransformer(
                            self._winsorize_transformer,
                            validate=False,
                            kw_args={"lower_limit": 0.001, "upper_limit": 0.001},
                        ),
                    ),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(
                ("power_features", power_pipeline, self.power_transform_cols)
            )

        all_processed_cols = set(self.log_transform_cols + self.power_transform_cols)
        remaining_numerical_cols = [
            c for c in self.numerical_features if c not in all_processed_cols
        ]

        if remaining_numerical_cols:
            remainder_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            transformers.append(
                ("remainder_features", remainder_pipeline, remaining_numerical_cols)
            )

        # Combile all transformers using ColumnTransformer
        # `n-jobs=-1` for parallel processing if applicable
        self.preprocessing_pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    ColumnTransformer(
                        transformers=transformers, remainder="passthrough"
                    ),
                )
            ]
        )

        return self.preprocessing_pipeline

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the preprocessing pipeline and transforms the data.
        Returns a processed DataFrame with proper feature names.
        """
        print(f"Cleaning data before fitting the pipeline...")
        X_cleaned = self._clean_data(X)

        pipeline = self.build_pipeline()
        print(f"Fitting and transforming data with pipeline...")

        # Fit-transform the data
        X_processed_array = pipeline.fit_transform(X_cleaned)

        try:
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out(
                X_cleaned.columns
            )
            cleaned_feature_names = [name.split("__")[-1] for name in feature_names]
            print("[INFO] Feature names cleaned.")
        except Exception as e:
            print(f"[WARNING] Could not extract feature names: {e}")
            feature_names = [f"x{i}" for i in range(X_processed_array.shape[1])]

        self.processed_df = pd.DataFrame(
            X_processed_array, columns=cleaned_feature_names, index=X_cleaned.index
        )

        print("[SUCCESS] Preprocessing complete with high-fidelity feature tracking.")
        return self.processed_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms new data using the fitted pipeline."""
        if self.preprocessing_pipeline is None:
            raise ValueError("Pipeline not fitted.Call fit_transform first.")

        print("Cleaning data before transforming...")
        X_cleaned = self._clean_data(X)

        print("Transforming new data with fitted pipeline...")
        X_processed_array = self.preprocessing_pipeline.transform(X_cleaned)

        try:
            feature_names = self.preprocessing_pipeline.get_feature_names_out()
            clean_feature_names = [
              name.split("__")[-1] for name in feature_names
            ]
        except AttributeError:
            clean_feature_names = [f"feature_{i}" for i in range(X_processed_array.shape[1])]

        self.processed_df = pd.DataFrame(
            X_processed_array, columns=clean_feature_names, index=X_cleaned.index
        )
        print("Transformation complete")
        return self.processed_df

    def get_corresponding_original_features(processed_df, original_df):
        """
        Extracts only the original features that have processed counterparts,
        maintaining the same row order as processed_df.
        """
        # Get only processed columns that exist in our mapping
        valid_processed_cols = [
            col for col in processed_df.columns if col in self.processed_to_original
        ]

        # Get corresponding original columns that exist in original_df
        feature_pairs = []
        for proc_col in valid_processed_cols:
            orig_col = self.processed_to_original[proc_col]
            if orig_col in original_df.columns:
                feature_pairs.append((orig_col, proc_col))

        # Create aligned subset of original_df
        original_subset = original_df[
            [orig_col for orig_col, _ in feature_pairs]
        ].copy()

        # Ensure same row order as processed_df (critical if preprocessing shuffled data)
        original_subset = original_subset.loc[processed_df.index]

        return original_subset, feature_pairs

    def reintegrate_processed_features_into_original_df(
        self, processed_df: pd.DataFrame, original_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Replaces original features in `original_df` with their corresponding processed features from `processed_df`.
        """

        # Track columns that were replaced
        updated_df = original_df.copy()

        # Track columns that were replaced
        replaced_columns = []

        for processed_col, original_col in self.processed_to_original.items():
            if (
                processed_col in processed_df.columns
                and original_col in updated_df.columns
            ):
                updated_df[original_col] = processed_df[processed_col]
                replaced_columns.append(original_col)

        print(
            f"[INFO] Replaced {len(replaced_columns)} original columns with processed versions:"
        )
        print(f"       {replaced_columns}")

        return updated_df
