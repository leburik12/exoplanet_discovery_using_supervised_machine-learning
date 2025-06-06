# Exoplanet Habitability Classification Pipeline

This repository provides an elite-level, production-grade pipeline for classifying exoplanet habitability using data from the NASA Exoplanet Archive. The architecture combines astrophysical rigor with statistical excellence, modular software design, and explainable ML workflows.

---

## Phase 1: Data Loading and Summary

**Module**: `ExoplanetDataLoader` (located in `src/data/load_data.py`)

This component ensures data ingestion and basic summary analysis with scientific integrity and fault-tolerant validation.

### ✅ Responsibilities

- Validates the input file path and ensures the file format is `.csv`.
- Loads datasets while optionally skipping commented metadata lines common in astronomy.
- Summarizes the dataset:
  - Data types and non-null counts
  - Missing value counts per column
  - Basic descriptive statistics (mean, std, min, max, etc.)

### 🔍 Class: `ExoplanetDataLoader`

#### Methods:

- `load()`: Reads the dataset and returns a `pandas.DataFrame`.
- `summarize()`: Prints data type info, null counts, and descriptive statistics.
- `get_feature_target_split(target_column: str)`: Returns a `(X, y)` tuple for ML training.

---

## Phase 2: Statistical Exploration

**Module**: `ExoplanetStatisticalExplorer`

This component conducts a detailed statistical examination of the dataset to drive preprocessing strategy.

### 🧠 Class Overview: `ExoplanetStatisticalExplorer`

- Distributional diagnostics of astrophysical features
- Multi-level correlation structure mapping
- Preprocessing strategy generation

### 1. Distributional Diagnostics

**Features analyzed include:**

- Planet radius: `pl_rade`, `pl_radj`
- Planet mass: `pl_bmasse`, `pl_bmassj`
- Orbital properties: `pl_orbper`, `pl_orbsmax`
- Stellar properties: `st_teff`, `st_mass`, `st_rad`, `st_met`
- Temperature & Insolation: `pl_eqt`, `pl_insol`

**Transformation Rules:**

- **Log Transform**: For highly skewed, strictly positive features.
- **Yeo-Johnson**: For moderate skewness and signed values.
- **Winsorization**: For extreme kurtosis.
- **StandardScaler**: Applied to all numerical features.

Visualizations are auto-generated (histograms, KDEs) and saved.

### 2. Multi-Level Correlation Structure

| Correlation Type | Method                            | Use Case               |
|------------------|-----------------------------------|------------------------|
| Pearson          | `df.corr('pearson')`              | Linear dependencies    |
| Spearman         | `df.corr('spearman')`             | Monotonic trends       |
| Distance         | `dcor.distance_correlation(x, y)` | Nonlinear associations |

**Example Pairs:**

- `st_teff` vs `pl_insol`
- `pl_eqt` vs `pl_orbper`
- `st_met` vs `pl_bmasse`

### 3. Preprocessing Recommendation Report

- A human-readable summary of preprocessing strategy is generated.
- Outputs:
  - Dict of transformation suggestions
  - Pearson, Spearman, Distance correlation matrices
  - Distribution plots and correlation heatmaps

---

## Phase 3: Feature Engineering

**Module**: `ExoplanetFeatureEngineer`

This is the scientific core of the pipeline. It translates astrophysical theory into machine-learnable signals.

### 🔬 Scientific Principles

- **Astrophysical Rigor**: Features based on domain knowledge (Kopparapu 2013, Barnes 2017).
- **Explainability-by-Construction**: Features are physically meaningful.
- **Modular Pipelines**: Uses `sklearn.Pipeline` for transformation logic.
- **Validation**: Enforces scientific sanity and schema compliance.

### 🧱 Modules

#### 1. `_validate_inputs(df)`
- Enforces presence of required columns.
- Sanity-checks physical plausibility.

#### 2. `_core_habitability_features(df)`
- `is_terrestrial`: Radius ≤ 1.6 Earth radii
- `in_hz`: Stellar flux in [0.36, 1.1] Earth units
- `temp_habitable`: Temperature in [273K, 373K]

#### 3. `_stellar_context_features(df)`
- `spectral_type`: Inferred from `st_teff`
- `ms_lifetime_gyr`: Lifetime via mass power law
- `metallicity_class`: Binned metallicity
- `core_accretion_prob`: From sigmoid on [Ida & Lin 2004]

#### 4. `_advanced_astrophysical_features(df)`
- `uv_flux`: Stefan-Boltzmann-derived estimate
- `tidal_lock_prob`: Orbital period / `st_mass` heuristic
- `jeans_parameter`: Atmospheric escape proxy

#### 5. `_xai_optimized_features(df)`
- `esi`: Earth Similarity Index
- `system_complexity`: Log planet count per system

### ⚙️ Integration

- All transformations embedded in a `sklearn.Pipeline` via `_create_sk_pipeline()`
- Internal `_classify_habitability(df)` function available for rule-based tagging.

---

## Phase 4: Data Preprocessing

**Module**: `ExoplanetDataPreprocessor`

Handles final input conditioning and pipeline assembly before modeling.

### 📐 Scientific Methods

#### 1. Domain-Aware Cleaning
- Clips invalid values (e.g., negative radius).
- Imputes `st_met` with solar metallicity (0.0).
- Uses median for imputation over mean to minimize skew.

#### 2. Transformation Logic
- `log1p` for scale-compressed features
- Yeo-Johnson for mixed-sign distributions
- Winsorization for outlier-tolerant modeling
- `StandardScaler` for SVM/XGBoost compatibility

#### 3. Feature Grouping
- Uses `ColumnTransformer` for modular pipelines
- Named transformers for interpretability
- Parallelizable for future mission scalability

#### 4. Traceable Transformation
- Subclassed `FunctionTransformer` tracks feature lineage
- Ensures full transparency for audits and scientific publications

### ⚙️ Execution Flow

1. `fit_transform(df)`
   - Cleans raw input
   - Builds & fits pipeline
   - Transforms dataset
   - Returns a clean, ready-for-ML `DataFrame`

---

## 🔗 References

- Kopparapu et al. (2013): Habitable Zones  
- Fulton et al. (2017): Radius Valley  
- Ida & Lin (2004): Core Accretion Theory  
- Barnes (2017): Tidal Locking Dynamics  
- Schulze-Makuch et al. (2011): Earth Similarity Index  
- Rugheimer et al. (2015): UV Habitability Zones  
- Gough (1981): Stellar Lifetimes  
- Pecaut & Mamajek (2013): Spectral Classification  

---

## 🧪 Output & Next Phase

- Transformed DataFrame for modeling  
- Correlation insights and preprocessing maps  
- Astrophysically and statistically robust feature set  

> Ideal for downstream use with dimensionality reduction (PCA), classifiers (SVM/XGBoost), and interpretability tools (SHAP/XAI).

---

For scientific inquiries, reach out to the author or open an issue.

> Designed for elite-level scientists in exoplanetary science, machine learning, and astrobiology.



Supervised learning target split


Stratified cross-validation planning
We deploy a 10-fold Stratified K-Fold CV for robust generalization assessment:
Phase 5: Model Training, Evaluation & Superiority of XGBoost 
XGBoostTrainer is a high-performance gradient boosting model using the XGBoost engine — designed to handle tabular data with nonlinear relationships, missing values, and imbalanced classes effectively.
scale_pos_weight=1 reflects a balanced label distribution, but this parameter remains tunable based on class skew.
calibrator="sigmoid" activates probability calibration via Platt scaling, ensuring well-calibrated confidence scores for metrics like AUC and Brier loss.
cv=5 enables internal cross-validation within the model training loop for optimal hyperparameter tuning or early stopping.
KNN, a distance-based non-parametric classifier, was tested but ultimately excluded from production due to inferior generalization performance on the habitable planet detection task.
The ModelEvaluator abstraction handles all evaluation loops across the cross-validation splits defined earlier.
It computes metrics like accuracy, precision, recall, F1, AUROC, and AUPRC per fold, then averages them for robust comparison.

| 🔍 Aspect               | ⚡ XGBoost                                            | ❄️ KNN                                               |
|------------------------|------------------------------------------------------|-----------------------------------------------------|
| **Handling Feature Interactions** | Learns nonlinear combinations via gradient trees   | No learned interactions; purely distance-based       |
| **Scalability**         | Highly optimized, GPU-accelerated                    | Slows dramatically with large datasets               |
| **Noise Sensitivity**   | Robust via regularization (lambda, gamma)            | Extremely sensitive to outliers and irrelevant features |
| **Missing Data**        | Native support via tree splitting logic              | Requires explicit imputation and scaling             |
| **Calibration**         | Supports sigmoid calibration                          | Outputs raw discrete votes (poor probabilistic interpretation) |
| **Class Imbalance**     | `scale_pos_weight`, early stopping, and weighted loss| No built-in mechanisms; requires resampling          |
| **Interpretability**    | Feature importances, SHAP values                      | Black-box distances; hard to explain                  |

## 📊 Model Evaluation Summary — XGBoost Classifier

The following metrics represent the **mean cross-validated performance** of the `XGBoost` classifier trained on the transformed exoplanet dataset. These results were obtained using a 10-fold `StratifiedKFold` strategy to preserve class distribution across splits. The pipeline included imputation, scaling, and calibrated classification.

---

### 🔢 Core Performance Metrics

| **Metric**               | **Score** |
|--------------------------|-----------|
| Accuracy                 | 0.9990    |
| Precision                | 0.9990    |
| Recall                   | 0.9990    |
| F1 Score                 | 0.9990    |
| Cohen’s Kappa            | 0.9375    |
| Matthews Correlation Coefficient (MCC) | 0.9387    |
| Log Loss                 | 0.0035    |
| Brier Score Loss         | 0.0008    |
| ROC AUC                  | 0.9998    |
| Average Precision (AP)   | 0.9832    |

---

### 📈 Classification Report

This table shows **per-class performance metrics** for the binary classification task (`is_habitable`), averaged across folds:

| **Class** | **Precision** | **Recall** | **F1-Score** | **Support** |
|-----------|---------------|------------|--------------|-------------|
| 0.0 (Non-Habitable) | 0.9997        | 0.9993     | 0.9995       | 3812.7      |
| 1.0 (Habitable)     | 0.9182        | 0.9625     | 0.9380       | 32.2        |
| **Macro Avg**       | 0.9590        | 0.9809     | 0.9688       | 3844.9      |
| **Weighted Avg**    | 0.9990        | 0.9990     | 0.9990       | 3844.9      |

> ✅ **Interpretation**:
> - **Extremely high overall accuracy** and **ROC AUC** indicate the classifier distinguishes habitable vs. non-habitable exoplanets with near-perfect precision.
> - The **precision-recall gap** on class `1.0` reflects the model’s conservative detection of rare habitable candidates, critical in scientific discovery settings.
---


### 📬 Contact

Authored by **Kirubel Awoke**  
📧 Email: [leburikplc@gmail.com](mailto:leburikplc@gmail.com)








