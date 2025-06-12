import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
import matplotlib.pyplot as plt

def get_euclidean_matrix(intensity_matrix):
    # transpose data so drugs become rows instead of columns
    intensity_matrix = intensity_matrix.transpose()
    # save drug names index column for after
    drug_names = intensity_matrix.index

    # plug transposed data into euclidean_distances function
    euc_distance_matrix = euclidean_distances(intensity_matrix)
    # convert to a dataframe and make row and column labels the drug names
    euc_distance_matrix = pd.DataFrame(euc_distance_matrix, columns = drug_names, index = drug_names)

    # normalize euclidean distances (between 0 and 1)
    min_distance = euc_distance_matrix.min().min()
    max_distance = euc_distance_matrix.max().max()
    normalized_euc_distances = (euc_distance_matrix - min_distance) / (max_distance - min_distance)

    # convert to similarity values instead of distance (so 1 = most similar)
    euc_similarity_matrix = 1 - normalized_euc_distances
    # euc_similarity_matrix.to_csv("euc_similarity_matrix.csv")
    return euc_similarity_matrix

def plot_distribution(similarity_matrix, title):
    euc_array = og_euc_similarity_matrix.to_numpy()
    # get the indices of the lower triangle in the array
    low_diag_indices = np.tril_indices_from(euc_array, k = -1)
    # use indices to get values of the lower triangle in the array
    low_diag_vals = euc_array[low_diag_indices]

    # plot the distribution
    sns.kdeplot(low_diag_vals)
    plt.xlabel("Euc Similarity")
    plt.title(f"Euclidean Similarity Between Drugs ({title})")
    plt.show()


# load real data matrix and fill in empty vals with 0
mean_intensity_matrix = pd.read_csv("Mean_Intensity_Matrix_All_Drugs.csv", index_col = 0).fillna(0)
# make column names just the drug name (remove _Mean_Intensity suffix)
mean_intensity_matrix.columns = mean_intensity_matrix.columns.str.removesuffix("_Mean_Intensity")

# get similarity matrix for real data
og_euc_similarity_matrix = get_euclidean_matrix(mean_intensity_matrix)
# plot distribution for real data  
plot_distribution(og_euc_similarity_matrix, "Real Data")


# randomize the real data matrix

# get randomized similarity matrix

# plot distribution for randomized data 