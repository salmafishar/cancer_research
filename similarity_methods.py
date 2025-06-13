import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler

# load and clean data
def clean_data(filepath):
    df = pd.read_csv(filepath)
    # storing and removing the gene column
    gene_names = df['gene']
    df_cleaned = df.drop(columns=['gene'])
    # filling missing values with 0
    df_cleaned = df_cleaned.fillna(0)

    return df_cleaned, gene_names

# randomize the data (randomly reorder the values of each column)
def randomize_dataframe(df):
    df_randomized = df.copy()
    # shuffling values:
    for col in df_randomized.columns:
        df_randomized[col] = random.sample(list(df_randomized[col]), len(df_randomized[col]))
    return df_randomized

# perturbing the data (small noise)
def perturb_dataframe(df, scale=1e5):
    noise = np.random.normal(loc=0, scale=scale, size=df.shape)
    df_perturbed = df + noise
    return df_perturbed

# cosine similarity
def calculate_cosine_similarity(df_cleaned, file_name):
    drug_vectors = df_cleaned.T
    similarity_matrix = cosine_similarity(drug_vectors)
    similarity_df = pd.DataFrame(similarity_matrix, index=drug_vectors.index, columns=drug_vectors.index)
    # similarity_df.to_csv(file_name)
    return similarity_df

# euclidean distance
def calculate_euclidean_distance(df_cleaned, file_name):
    drug_vectors = df_cleaned.T
    # standardizing the data
    scaler = StandardScaler()
    scaled_vectors = scaler.fit_transform(drug_vectors)
    # get the euclidean distance matrix
    distance_matrix = euclidean_distances(scaled_vectors)
    # convert to a dataframe, make row/column labels the drug names
    distance_df = pd.DataFrame(distance_matrix, index=drug_vectors.index, columns=drug_vectors.index)
    # distance_df.to_csv(file_name)
    return distance_df


