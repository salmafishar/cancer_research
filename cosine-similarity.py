import pandas as pd

df = pd.read_csv("Mean_Intensity_Matrix_All_Drugs.csv")

# removing the gene column and storing it separately

gene_names = df['gene']
df_cleaned = df.drop(columns=['gene'])

# filling missing values with 0
df_cleaned = df_cleaned.fillna(0)

from sklearn.metrics.pairwise import cosine_similarity

drug_vectors = df_cleaned.T

similarity_matrix = cosine_similarity(drug_vectors)

similarity_df = pd.DataFrame(similarity_matrix, index=drug_vectors.index, columns=drug_vectors.index)

similarity_df.to_csv('cosine_similarity_matrix.csv')

# Randomizing the data (columns)

import random

df_randomized = df_cleaned.copy()

# shuffling values:
for col in df_randomized.columns:
    df_randomized[col] = random.sample(list(df_randomized[col]), len(df_randomized[col]))

random_vectors = df_randomized.T

random_similarity_matrix = cosine_similarity(random_vectors)

random_df = pd.DataFrame(random_similarity_matrix, index=random_vectors.index, columns=random_vectors.index)

random_df.to_csv("cosine_similarity_randomized.csv")

# perturbing the data (small noise)

import numpy as np

df_perturbed = df_cleaned.copy()

noise = np.random.normal(loc=0, scale=1e5, size=df_perturbed.shape)
df_perturbed += noise

perturbed_vectors = df_perturbed.T
perturbed_similarity_matrix = cosine_similarity(perturbed_vectors)
perturbed_df = pd.DataFrame(perturbed_similarity_matrix, index=perturbed_vectors.index, columns=perturbed_vectors.index)
perturbed_df.to_csv('cosine_similarity_perturbed.csv')
