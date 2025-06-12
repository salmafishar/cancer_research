import pandas as pd
import matplotlib.pyplot as plt


original_df = pd.read_csv("cosine_similarity_real.csv", index_col=0)
random_df = pd.read_csv("cosine_similarity_randomized.csv", index_col=0)
perturbed_df = pd.read_csv("cosine_similarity_perturbed.csv", index_col=0)
euclidean_real = pd.read_csv("euclidean_distance_real.csv", index_col=0)
euclidean_random = pd.read_csv("euclidean_distance_randomized.csv", index_col=0)
euclidean_perturbed = pd.read_csv("euclidean_distance_perturbed.csv", index_col=0)


def plot_histogram(df, title, bins=50):
    values = df.values.flatten()
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


plot_histogram(original_df, "Original Cosine Similarity")
plot_histogram(random_df, "Randomized Cosine Similarity")
plot_histogram(perturbed_df, "Perturbed Cosine Similarity")
plot_histogram(euclidean_real, "Original Euclidean Distance")
plot_histogram(euclidean_random, "Randomized Euclidean Distance")
plot_histogram(euclidean_perturbed, "Perturbed Euclidean Distance")
