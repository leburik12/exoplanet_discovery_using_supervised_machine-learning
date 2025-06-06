import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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