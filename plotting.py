import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(similarity_matrix, title, filepath, type_boundaries = None, type_labels = None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='viridis', xticklabels=False, yticklabels=False)
    
    # add grid lines for the boundaries of each type (if provided)
    if type_boundaries:
        cumulative_boundaries = [sum(type_boundaries[:i]) for i in range(1, len(type_boundaries))]
        for boundary in cumulative_boundaries:
            plt.axhline(boundary, color='white', linewidth=1)
            plt.axvline(boundary, color='white', linewidth=1)
        if type_labels:
            start = 0
            for i, length in enumerate(type_boundaries):
                label = f"{type_labels[i]}s"

                y_pos = start + length / 2
                plt.text(-2, y_pos, label, ha='right', va='center', color='black', fontsize = 7)
                
                start += length 

    plt.title(f"{title}")
    plt.tight_layout()
    plt.savefig(f"{filepath}_heatmap.png")
    plt.close()

def plot_histogram(similarity_matrix, title, filepath):
    values = similarity_matrix.values.flatten()
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=50, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Euclidean Distance")
    plt.title(f"{title}")
    plt.tight_layout()
    plt.savefig(f"{filepath}_histogram.png")
    plt.close()

def plot_density(similarity_matrix, title, filepath):
    euc_array = similarity_matrix.to_numpy()
    # get the indices of the lower triangle in the array
    low_diag_indices = np.tril_indices_from(euc_array, k = -1)
    # use indices to get values of the lower triangle in the array
    low_diag_vals = euc_array[low_diag_indices]
    # plot the distribution
    if len(similarity_matrix.columns) < 5:
        plt.hist(low_diag_vals, color = 'skyblue', edgecolor='black')
    else: 
         sns.kdeplot(low_diag_vals, fill = "skyblue", warn_singular=False, clip=(0, None))
    plt.xlabel("Euclidean Distance")
    plt.title(f"{title}")
    plt.savefig(f"{filepath}_density.png")
    plt.close()




