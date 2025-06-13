import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(similarity_matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='viridis', xticklabels=False, yticklabels=False)
    plt.title(f"{title}")
    plt.tight_layout()
    plt.savefig(f"{title}_heatmap.png")
    plt.close()

def plot_histogram(similarity_matrix, title, bins=50):
    values = similarity_matrix.values.flatten()
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Euclidean Distance")
    plt.title(f"{title}")
    plt.tight_layout()
    plt.savefig(f"{title}_histogram.png")
    plt.close()

def plot_density(similarity_matrix, title):
    euc_array = similarity_matrix.to_numpy()
    # get the indices of the lower triangle in the array
    low_diag_indices = np.tril_indices_from(euc_array, k = -1)
    # use indices to get values of the lower triangle in the array
    low_diag_vals = euc_array[low_diag_indices]
    # plot the distribution
    sns.kdeplot(low_diag_vals, fill = "skyblue")
    plt.xlabel("Euclidean Distance")
    plt.title(f"{title}")
    plt.savefig(f"{title}_density.png")
    plt.close()




