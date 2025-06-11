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
    drugNames = intensity_matrix.index

    # plug transposed data into euclidean_distances function
    eucDistanceMatrix = euclidean_distances(intensity_matrix)
    # convert to a dataframe and make row and column labels the drug names
    eucDistanceMatrix = pd.DataFrame(eucDistanceMatrix, columns = drugNames, index = drugNames)

    # normalize euclidean distances (between 0 and 1)
    minDistance = eucDistanceMatrix.min().min()
    maxDistance = eucDistanceMatrix.max().max()
    normalizedEucDistances = (eucDistanceMatrix - minDistance) / (maxDistance - minDistance)

    # convert to similarity values instead of distance (so 1 = most similar)
    eucSimilarityMatrix = 1 - normalizedEucDistances
    # eucDistanceMatrix.to_csv("euc_similarity_matrix.csv")
    return eucSimilarityMatrix

# load matrix and fill in empty vals with 0 
mean_intensity_matrix = pd.read_csv("Mean_Intensity_Matrix_All_Drugs.csv", index_col = 0).fillna(0)
# make column names just the drug name (remove _Mean_Intensity suffix)
mean_intensity_matrix.columns = mean_intensity_matrix.columns.str.removesuffix("_Mean_Intensity")

euc_similarity_matrix = get_euclidean_matrix(mean_intensity_matrix)