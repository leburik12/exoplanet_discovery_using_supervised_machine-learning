import numpy as np
import pandas as pd
from scipy import stats
import dcor

from src.constants.physical_constants import PHYSICAL_CONSTANTS
from src.utils.helpers import suggest_transformation
from src.visualization.plot_results import plot_distributions, plot_correlation_matrix


class ExoplanetStatisticalExplorer:
    def __init__(self, df):
        self.df = df.copy()
        self.constants = PHYSICAL_CONSTANTS
        self.completeness_report = {}
        self.dist_results: Dict[str, Dict[str, Any]] = {}
        self.correlation_results: Dict[str, pd.DataFrame] = {}
        self.pca_results = {}
        self.preprocessing_recommendations: Dict[str, List[str]] = {
            "log_transform": [],
            "power_transform_yeo_johnson": [],
            "winsorize": [],
            "scale": [],
        }

    def _distribution_analysis(self):
        print("\n🔭 Distribution Analysis")
        dist_cols = [
            col
            for col in [
                "pl_rade",
                "pl_radj",
                "pl_bmasse",
                "pl_bmassj",
                "pl_orbper",
                "pl_orbsmax",
                "pl_eqt",
                "pl_insol",
                "st_teff",
                "st_rad",
                "st_mass",
                "st_met",
                "st_logg",
                "sy_dist",
                "sy_vmag",
                "sy_kmag",
                "sy_gaiamag",
            ]
            if col in self.df.columns
        ]

        for col in dist_cols:
            col_data = self.df[col].dropna()
            if col_data.empty or len(col_data) < 2:
                print(f"Skipping distribution analysis for {col}: Insufficient data.")
                continue
            try:
                # Calculate fundamental statistics
                skewness = stats.skew(col_data)
                kurtosis = stats.kurtosis(
                    col_data, fisher=True
                )  # Fisher's definition: Normal = 0

                transform_type = "None"
                if (
                    abs(skewness) > 1.0 or abs(kurtosis) > 10.0
                ):  # High skew or extreme leptokurtosis
                    if (
                        skewness > 1.0 and (col_data > 0).all()
                    ):  # Highly positive skew, only positive values
                        transform_type = "log_transform"
                        self.preprocessing_recommendations["log_transform"].append(col)
                    else:
                        transform_type = "Power Transform (Yeo-Johnson)"
                        self.preprocessing_recommendations[
                            "power_transform_yeo_johnson"
                        ].append(col)
                elif (
                    abs(skewness) > 0.5 or abs(kurtosis) > 3.0
                ):  # Moderate skew or kurtosis
                    transform_type = "power_transform_yeo_johnson"
                    self.preprocessing_recommendations[
                        "power_transform_yeo_johnson"
                    ].append(col)

                # Always consider winsorization for highly kurtotic features
                if (
                    abs(kurtosis) > 5.0
                ):  # A heuristic threshold for suggesting winsorization
                    self.preprocessing_recommendations["winsorize"].append(col)

                self.dist_results[col] = {
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                    "transform_suggested": transform_type,
                }

                print(
                    f" - {col}: Skew={skewness:.2f}, Kurtosis={kurtosis:.2f}, Suggested: {transform_type}"
                )
                # transform_suggested, pval = suggest_transformation(col_data)
                # self.dist_results[col] = {
                #     'skewness': stats.skew(col_data) if len(col_data) > 1 else np.nan,
                #     'kurtosis': stats.kurtosis(col_data) if len(col_data) > 1 else np.nan,
                #     'normality_p': pval,
                #     'transform_suggested': transform_suggested
                # }
            except Exception as e:
                print(f"Error in distribution for {col}: {e}")

        # All numerical features should eventually be scaled
        self.preprocessing_recommendations["scale"].extend(dist_cols)
        self.preprocessing_recommendations["scale"] = list(
            set(self.preprocessing_recommendations["scale"])
        )

        plot_distributions(self.df, self.dist_results, dist_cols)

    def _correlation_structure(self):
        corrs_cols = [
            col
            for col in [
                "pl_rade",
                "pl_insol",
                "pl_eqt",
                "st_teff",
                "st_met",
                "st_mass",
                "pl_orbeccen",
                "sy_pnum",
            ]
            if col in self.df.columns
        ]

        df_sub = self.df[corrs_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if df_sub.empty or len(df_sub.columns) < 2:
            return

        pearson_corr = df_sub.corr(method="pearson")
        spearman_corr = df_sub.corr(method="spearman")

        distance_corr = pd.DataFrame(index=corrs_cols, columns=corrs_cols, dtype=float)
        for i in corrs_cols:
            for j in corrs_cols:
                xi, xj = self.df[i].dropna(), self.df[j].dropna()
                idx = xi.index.intersection(xj.index)
                if len(idx) > 1:
                    xi_aligned, xj_aligned = xi.loc[idx].values, xj.loc[idx].values
                    distance_corr.loc[i, j] = dcor.distance_correlation(
                        xi_aligned, xj_aligned
                    )

        self.correlation_results = {
            "pearson": pearson_corr,
            "spearman": spearman_corr,
            "distance": distance_corr,
        }

        print("\nCorrelation Matrices (first few rows):")
        print("Pearson:\n", pearson_corr.head())
        print("Spearman:\n", spearman_corr.head())
        print("Distance:\n", distance_corr.head())

        plot_correlation_matrix(
            pearson_corr,
            "Pearson Correlation Matrix",
            "correlation_pearson.png",
            cmap="icefire",
        )
        plot_correlation_matrix(
            spearman_corr,
            "Spearman Correlation Matrix",
            "correlation_spearman.png",
            cmap="viridis",
        )
        plot_correlation_matrix(
            distance_corr,
            "Distance Correlation Matrix",
            "correlation_distance.png",
            cmap="magma",
        )

    def _report_preprocessing_recommendations(self):
        print("\n=== Preprocessing Recommendations Based on EDA ===")
        print("\nFeatures for Log Transform (e.g., np.log1p):")
        print(self.preprocessing_recommendations["log_transform"] or "None")

        print("\nFeatures for Power Transform (Yeo-Johnson):")
        print(
            self.preprocessing_recommendations["power_transform_yeo_johnson"] or "None"
        )

        print("\nFeatures Recommended for Winsorization (Outlier Capping):")
        print(self.preprocessing_recommendations["winsorize"] or "None")

        print("\nAll Numerical Features for Standard Scaling (after transformation):")
        print(self.preprocessing_recommendations["scale"] or "None")

    def explore(self):
        self._distribution_analysis()
        self._correlation_structure()
