import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from astropy.constants import M_earth, R_earth


class ExoplanetFeatureEngineer:
    def __init__(self):
        self.physical_constants = {
            'G': 6.67430e-11,  # m^3 kg^-1 s^-2, Gravitational Constant
            'k_B': 1.380649e-23,  # J/K , Boltzmann Constant
            'sigma_SB': 5.670374e-8  # W/m^2/K^4 , Stefan–Boltzmann Constant
        }

        self.solar_properties = {
            'teff': 5778,  # K
            'metallicity': 0.0,  # [Fe/H]
            'age': 4.6  # Gyr
        }
        # Placeholder for the scikit-learn pipeline, to be created in generate_features
        self.pipeline = None

    def _validate_inputs(self, df):
        """Ensure required columns exist with proper units and non-null values for core features."""
        required_cols = {
            'pl_rade': 'Earth radii',
            'pl_insol': 'Earth flux',
            'st_teff': 'K',
            'st_met': 'dex',
            'pl_eqt': 'K', # Added as it's used in _core_habitability_features
            'pl_orbper': 'days', # Added for tidal locking
            'st_mass': 'solar units', # Added for MS lifetime, tidal locking
            'pl_bmasse': 'Earth masses', # Added for Jeans parameter
            'hostname': 'string', # Added for system complexity
            'pl_name': 'string' # Added for system complexity
        }

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing critical columns: {missing}")

        # Basic check for essential feature calculation inputs (beyond just existence)
        essential_numerical_cols = ['pl_rade', 'pl_insol', 'st_teff', 'pl_eqt', 'pl_orbper', 'st_mass', 'pl_bmasse']
        for col in essential_numerical_cols:
            if df[col].isnull().all():
                raise ValueError(f"Essential column '{col}' is entirely null. Cannot proceed with feature engineering.")

        return df.copy() # Return a copy to avoid modifying the original DataFrame unexpectedly

    def _core_habitability_features(self, df):
        """Primary habitability indicators"""
        df_features = df.copy()
        # Radius valley classification (Fulton et al. 2017)
        df_features['is_terrestrial'] = (df_features['pl_rade'] <= 1.6).astype(int)

        # Conservative habitable zone (Kopparapu et al. 2013)
        # Assuming Earth flux units for pl_insol.
        df_features['in_hz'] = ((df_features['pl_insol'] >= 0.36) & (df_features['pl_insol'] <= 1.1)).astype(int)

        # Temperature bounds for liquid water (assumes pl_eqt is in K)
        df_features['temp_habitable'] = ((df_features['pl_eqt'] >= 273) & (df_features['pl_eqt'] <= 373)).astype(int)

        return df_features

    def _stellar_context_features(self, df):
        """Stellar environment features"""
        df_features = df.copy()

        # Spectral classification (Pecaut & Mamajek 2013, simplified)
        def classify_spectral_type(teff):
            if pd.isna(teff): return 'Unknown'
            if teff >= 7500: return 'A'
            if teff >= 6000: return 'F'
            if teff >= 5200: return 'G' # Solar-type
            if teff >= 3700: return 'K'
            if teff >= 2400: return 'M'
            return 'Unknown' # For stars cooler than M-dwarfs or invalid teff

        df_features['spectral_type'] = df_features['st_teff'].apply(classify_spectral_type)

        # Main sequence lifetime (Gough 1981, simplified power law approximation)
        # Ensure st_mass is not zero or negative to avoid division by zero or errors
        df_features['ms_lifetime_gyr'] = 10 * (df_features['st_mass'].replace(0, np.nan) ** -2.5)
        df_features['ms_lifetime_gyr'] = df_features['ms_lifetime_gyr'].fillna(df_features['ms_lifetime_gyr'].median()) # Impute any NaNs after calculation

        # Metallicity impact (Buchhave et al. 2012, simplified bins)
        metallicity_bins = [
            (-np.inf, -0.5, 'low_occurrence'),
            (-0.5, 0, 'moderate'),
            (0, 0.3, 'sweet_spot'),
            (0.3, np.inf, 'super_metallic')
        ]
        # Use pd.cut with proper intervals and labels
        df_features['metallicity_class'] = pd.cut(
            df_features['st_met'],
            bins=[b[0] for b in metallicity_bins] + [np.inf],
            labels=[b[2] for b in metallicity_bins],
            right=False # Interval includes the left edge, excludes the right
        )

        # Core accretion probability (Ida & Lin 2004, simplified sigmoid approximation)
        df_features['core_accretion_prob'] = 1 / (1 + np.exp(-10*(df_features['st_met'] + 0.3)))

        return df_features

    def _advanced_astrophysical_features(self, df):
        """Cutting-edge derived features"""
        df_features = df.copy()

        # UV habitable zone (Rugheimer et al. 2015, simplified scaling)
        # Ensure st_teff and pl_insol are not zero/negative and solar_properties['teff'] is not zero
        teff_ratio = df_features['st_teff'] / self.solar_properties['teff']
        df_features['uv_flux'] = (teff_ratio**4) * df_features['pl_insol']
        df_features['uv_flux'] = df_features['uv_flux'].fillna(df_features['uv_flux'].median()) # Impute NaNs

        # Tidal locking timescale (Barnes 2017, highly simplified and requires more precise inputs typically)
        # pl_orbper in days, st_mass in solar masses. Conversion factors for consistency.
        # This is a very rough proxy. A full calculation is complex.
        # Avoid division by zero or negative values.
        st_mass_clamped = df_features['st_mass'].clip(lower=0.1) # Clamp to avoid near-zero/negative mass
        orbital_factor = (0.1 * st_mass_clamped**(-1.5))
        # Handle cases where orbital_factor might be zero or tiny, leading to large values in denominator
        denominator = (df_features['pl_orbper'] / (orbital_factor + 1e-9))**2 # Add epsilon
        df_features['tidal_lock_prob'] = 1 / (1 + denominator)
        df_features['tidal_lock_prob'] = df_features['tidal_lock_prob'].fillna(df_features['tidal_lock_prob'].median()) # Impute NaNs


        # Atmospheric escape (Jeans parameter, simplified)
        # Ensure pl_bmasse, pl_rade, pl_eqt are non-zero/negative
        # Using astropy.units for constants
        G = self.physical_constants['G']
        k_B = self.physical_constants['k_B']
        pl_bmasse_kg = df_features['pl_bmasse'] * M_earth.value
        pl_rade_m = df_features['pl_rade'] * R_earth.value

        # Clamp denominators to avoid division by zero
        denominator = (pl_rade_m * k_B * df_features['pl_eqt']).replace(0, np.nan)
        df_features['jeans_parameter'] = (G * pl_bmasse_kg) / denominator
        df_features['jeans_parameter'] = df_features['jeans_parameter'].fillna(df_features['jeans_parameter'].median()) # Impute NaNs

        return df_features

    def _xai_optimized_features(self, df):
        """Features designed for interpretable ML"""
        df_features = df.copy()
        # Earth Similarity Index (Schulze-Makuch 2011, simplified)
        # Note: Original ESI has 4 components. This is a simplified 2-component version.
        df_features['esi'] = 1 - np.sqrt(
            0.5*((df_features['pl_rade']-1)/1)**2 +
            0.5*((df_features['pl_insol']-1)/1)**2
        )
        df_features['esi'] = df_features['esi'].clip(lower=0) # ESI is typically [0,1]

        # System architecture complexity (log-transformed count of planets in system)
        df_features['system_complexity'] = np.log1p(df_features.groupby('hostname')['pl_name'].transform('count'))

        return df_features

    def _create_sk_pipeline(self):
        """Creates and returns a scikit-learn Pipeline for feature engineering."""
        return Pipeline([
            ('data_validation', FunctionTransformer(self._validate_inputs, validate=False)), # Validation done separately as it raises errors
            ('core_features', FunctionTransformer(self._core_habitability_features, validate=False)),
            ('stellar_features', FunctionTransformer(self._stellar_context_features, validate=False)),
            ('advanced_physics', FunctionTransformer(self._advanced_astrophysical_features, validate=False)),
            ('xai_features', FunctionTransformer(self._xai_optimized_features, validate=False))
        ])

    def _classify_habitability(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Internal method to classify exoplanets as 'habitable' (1) or 'non-habitable' (0)
            based on a set of astrobiological and astrophysical criteria.
            This method assumes that the input DataFrame 'df' has already gone through
            _validate_inputs and _clean_data from the FeatureEngineer.
         """
        df_classified = df.copy()

        # Re-using the already cleaned and potentially imputed columns from the FeatureEngineer pipeline
        # No need for separate 'st_mass_clean' as 'st_mass' is already processed in _clean_data
        # 'ms_lifetime_gyr' is already calculated in _stellar_context_features
        # Ensure 'ms_lifetime_gyr' is available, if not, calculate it (should be present if _stellar_context_features runs)
        if 'ms_lifetime_gyr' not in df_classified.columns:
            # This should ideally not happen if the pipeline is structured correctly,
            # but acts as a safeguard.
            df_classified['ms_lifetime_gyr'] = 10 * (df_classified['st_mass'] ** -2.5)
            df_classified['ms_lifetime_gyr'] = df_classified['ms_lifetime_gyr'].fillna(df_classified['ms_lifetime_gyr'].median())
            print("Warning: ms_lifetime_gyr was not pre-calculated; calculating within _classify_habitability.")

        # --- Step 1: Define the Size/Composition Criterion ---
        df_classified['size_criterion_met'] = False
        df_classified.loc[(df_classified['pl_rade'] >= 0.5) & (df_classified['pl_rade'] <= 2.0), 'size_criterion_met'] = True
        df_classified.loc[df_classified['pl_rade'].isna() &
                          (df_classified['pl_bmasse'] >= 0.1) & (df_classified['pl_bmasse'] <= 10.0),
                          'size_criterion_met'] = True

        # --- Step 2: Define the Temperature/Insolation Criterion ---
        df_classified['temp_criterion_met'] = False
        df_classified.loc[(df_classified['pl_insol'] >= 0.25) & (df_classified['pl_insol'] <= 2.0), 'temp_criterion_met'] = True
        df_classified.loc[df_classified['pl_insol'].isna() &
                          (df_classified['pl_eqt'] >= 200) & (df_classified['pl_eqt'] <= 400),
                          'temp_criterion_met'] = True

        # --- Step 3: Define Stellar Longevity/Stability Criterion ---
        df_classified['stellar_stability_criterion_met'] = False
        df_classified.loc[df_classified['ms_lifetime_gyr'] >= 1.0, 'stellar_stability_criterion_met'] = True

        # --- Step 4: Combine All Criteria to Create the Binary Target ---
        df_classified['is_habitable'] = (
            (df_classified['size_criterion_met'] == True) &
            (df_classified['temp_criterion_met'] == True) &
            (df_classified['stellar_stability_criterion_met'] == True)
        ).astype(int)

        # Clean up intermediate columns used for classification
        df_classified = df_classified.drop(columns=[
            'size_criterion_met',
            'temp_criterion_met',
            'stellar_stability_criterion_met'
        ])

        return df_classified

    def generate_features(self, df):
        """Execute full feature engineering pipeline"""
        # It's good practice to run validation before starting the pipeline
        validated_df = self._validate_inputs(df)

        # Initialize pipeline if not already done
        if self.pipeline is None:
            self.pipeline = self._create_sk_pipeline()

        # Execute pipeline
        features = self.pipeline.fit_transform(validated_df)

        # Topological data analysis (TDA) - this needs careful integration
        # TDA can be computationally intensive and may not scale well for very large datasets
        # It also generates complex outputs (persistence diagrams) which need careful summarization.
        # For a production pipeline, consider a more robust, scalable TDA library or pre-computed features.
        # Here, we'll store a simple summary.

        # Ensure points has enough data and no NaNs for TDA
        tda_cols = ['pl_orbsmax', 'pl_rade', 'pl_insol']
        tda_data = df[tda_cols].dropna()

        if not tda_data.empty:
            rips = Rips()
            points = tda_data.values
            # Fit and transform can be very slow for large datasets
            homology_features = rips.fit_transform(points)

            # Store summary statistic (e.g., number of H0/H1 loops) instead of raw diagrams
            # Check if homology_features has elements before accessing
            features['num_H0_features'] = len(homology_features[0]) if len(homology_features) > 0 else 0 # 0D connected components
            features['num_H1_features'] = len(homology_features[1]) if len(homology_features) > 1 else 0 # 1D loops
        else:
            print("Warning: Skipping TDA as input data for TDA features contains NaNs or is empty.")
            features['num_H0_features'] = 0
            features['num_H1_features'] = 0

        return features

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
      """ Main method to generate engineered features and label the data."""
      self.pipeline = self._create_sk_pipeline()
      df_transformed = self.pipeline.fit_transform(df)
      df_final = self._classify_habitability(df_transformed)
      return df_final
