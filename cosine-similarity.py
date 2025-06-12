import pandas as pd
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler

# Cleaning the data and loading it
def clean_data(filepath):
    df = pd.read_csv(filepath)
    # removing the gene column and storing it separately
    gene_names = df['gene']
    df_cleaned = df.drop(columns=['gene'])
    # filling missing values with 0
    df_cleaned = df_cleaned.fillna(0)
    return df_cleaned, gene_names


df_cleaned, gene_names = clean_data(filepath='Mean_Intensity_Matrix_All_Drugs.csv')


# COSINE SIMILARITY
def calculating_cosine_similarity(df_cleaned, output):
    drug_vectors = df_cleaned.T
    similarity_matrix = cosine_similarity(drug_vectors)
    similarity_df = pd.DataFrame(similarity_matrix, index=drug_vectors.index, columns=drug_vectors.index)
    similarity_df.to_csv(output)
    return similarity_df


# EUCLIDEAN DISTANCE
def calculate_euclidean_distance(df_cleaned, output):
    drug_vectors = df_cleaned.T
    # standardizing the data
    scaler = StandardScaler()
    scaled_vectors = scaler.fit_transform(drug_vectors)
    distance_matrix = euclidean_distances(scaled_vectors)
    distance_df = pd.DataFrame(distance_matrix, index=drug_vectors.index, columns=drug_vectors.index)
    distance_df.to_csv(output)
    return distance_df


# Create the OG csv files
calculating_cosine_similarity(df_cleaned, "cosine_similarity_real.csv")
calculate_euclidean_distance(df_cleaned, "euclidean_distance_real.csv")


# Randomizing the data (columns)
def randomize_dataframe(df):
    df_randomized = df.copy()
    # shuffling values:
    for col in df_randomized.columns:
        df_randomized[col] = random.sample(list(df_randomized[col]), len(df_randomized[col]))
    return df_randomized


random_df = randomize_dataframe(df_cleaned)
calculating_cosine_similarity(random_df, "cosine_similarity_randomized.csv")
calculate_euclidean_distance(random_df, "euclidean_distance_randomized.csv")


# perturbing the data (small noise)

def perturb_dataframe(df, scale=1e5):
    noise = np.random.normal(loc=0, scale=scale, size=df.shape)
    df_perturbed = df + noise
    return df_perturbed


perturbed_df = perturb_dataframe(df_cleaned, scale=1e5)
calculating_cosine_similarity(perturbed_df, "cosine_similarity_perturbed.csv")
calculate_euclidean_distance(perturbed_df, "euclidean_distance_perturbed.csv")
