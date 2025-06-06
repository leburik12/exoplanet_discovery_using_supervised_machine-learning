import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(matrix, title, file_path, cmap='coolwarm'):
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, cmap=cmap, center=0, annot_kws={"size": 8}, fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()