import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

original_df = pd.read_csv("cosine_similarity_matrix.csv", index_col=0)
random_df = pd.read_csv("cosine_similarity_randomized.csv", index_col=0)
perturbed_df = pd.read_csv("cosine_similarity_perturbed.csv", index_col=0)


def plot_heatmap(df, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap='viridis', xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()


plot_heatmap(original_df, "Original Cosine Similarity")
plot_heatmap(random_df, "Randomized Cosine Similarity")
plot_heatmap(perturbed_df, "Perturbed Cosine Similarity")
