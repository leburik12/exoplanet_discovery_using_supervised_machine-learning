import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions(df, dist_results, cols, output_file='distributions.png'):
    cols_to_plot = [col for col in cols if col in dist_results and not df[col].dropna().empty]

    num_cols = len(cols_to_plot)
    num_rows = (num_cols + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(14, num_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        data = df[col].dropna()
        if dist_results.get(col, {}).get('transform_suggested', False) and all(data > 0):
            data = np.log10(data + 1e-6)
            ax.set_title(f'Log10({col}) Distribution')
        else:
            ax.set_title(f'{col} Distribution')
        sns.histplot(data, kde=True, ax=ax, bins=50)
        ax.axvline(data.mean(), color='r', linestyle='--', label='Mean')
        ax.axvline(data.median(), color='g', linestyle=':', label='Median')
        ax.legend()

    for j in range(num_cols, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

def plot_correlation_matrix(matrix, title, file_path, cmap='coolwarm'):
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, cmap=cmap, center=0, annot_kws={"size": 8}, fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

def plot_transformation_effects(original_df, processed_df, feature_pairs):
    """
    Visualizes before/after distributions for key exoplanet features
    """
    n = len(feature_pairs)
    cols = 3
    rows = math.ceil(n / cols)

    plt.figure(figsize=(6 * cols, 4 * rows))
    for i, (orig_col, proc_col) in enumerate(feature_pairs, 1):
        plt.subplot(rows, cols, i)

        # Original distribution
        sns.kdeplot(
            original_df[orig_col], color="red", label="Original", fill=True, alpha=0.3
        )
        plt.title(f"{orig_col} Transformation", fontsize=12)
        plt.xlabel("value")
        plt.ylabel("Density")

        # Processed distribution
        sns.kdeplot(
            processed_df[proc_col],
            color="blue",
            label="Processed",
            fill=True,
            alpha=0.3,
        )
        plt.legend()
        plt.grid(True, alpha=0.2)

    plt.suptitle("Exoplanet Feature Distribution Optimization", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def get_corresponding_original_features(processed_df, original_df):
    """
    Extracts only the original features that have processed counterparts,
    maintaining the same row order as processed_df.
    """
    # Create mapping from processed to original column names
    processed_to_original = {
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
    # Get only processed columns that exist in our mapping
    valid_processed_cols = [
        col for col in processed_df.columns if col in processed_to_original
    ]

    # Get corresponding original columns that exist in original_df
    feature_pairs = []
    for proc_col in valid_processed_cols:
        orig_col = processed_to_original[proc_col]
        if orig_col in original_df.columns:
            feature_pairs.append((orig_col, proc_col))

    # Create aligned subset of original_df
    original_subset = original_df[[orig_col for orig_col, _ in feature_pairs]].copy()

    # Ensure same row order as processed_df (critical if preprocessing shuffled data)
    original_subset = original_subset.loc[processed_df.index]

    return original_subset, feature_pairs




