import pandas as pd
from plotting import plot_density, plot_histogram, plot_heatmap
from similarity_methods import clean_data, calculate_cosine_similarity, calculate_euclidean_distance, randomize_dataframe, perturb_dataframe

# load and get cleaned intensity matrix
df_cleaned, gene_names = clean_data(filepath='Mean_Intensity_Matrix_All_Drugs.csv')

# create og similarity/distance matrix (and csv files)
og_cos_matrix = calculate_cosine_similarity(df_cleaned, "cosine_similarity_real.csv")
og_euc_matrix = calculate_euclidean_distance(df_cleaned, "euclidean_distance_real.csv")

# plot real data 
plot_density(og_euc_matrix, "Real Euclidean Distance")
plot_heatmap(og_euc_matrix, "Real Euclidean Distance")

# randomize og matrix
random_df = randomize_dataframe(df_cleaned)
# get randomized distance matrix
random_euc_matrix = calculate_euclidean_distance(random_df, "euclidean_distance_random.csv")
# plot randomized data
plot_density(random_euc_matrix, "Randomized Euclidean Distance")
plot_heatmap(random_euc_matrix, "Randomized Euclidean Distance")

# perturb og matrix
perturbed_df = perturb_dataframe(df_cleaned, scale=1e5)
# get perturbed distance matrix
perturbed_euc_matrix = calculate_euclidean_distance(perturbed_df, "euclidean_distance_perturbed.csv")
# plot perturbed data
plot_density(perturbed_euc_matrix, "Perturbed Euclidean Distance")
plot_heatmap(perturbed_euc_matrix, "Perturbed Euclidean Distance")



# function that takes a list of drug names and a similarity/distance df, and returns the df only containing the given drugs
def get_sub_matrix(drug_names, similarity_matrix):
    sub_matrix = similarity_matrix.loc[drug_names, drug_names]
    return sub_matrix

# generate dictionary of types (keys) + drugs (values)
drug_info = pd.read_csv("drug_info_supp_table.csv")
drug_info["Drug"] = drug_info["Drug"].str.strip()
drug_types = drug_info.groupby("Type")["Drug"].apply(list).to_dict()

# create dictionary of types (keys) + euclidean distance df (values)
euclidean_by_type = {}
for type in drug_types:
    euclidean_by_type[type] = get_sub_matrix(drug_types[f"{type}"], og_euc_matrix) 

for type, df in euclidean_by_type.items():
    plot_density(df, f"Real Euclidean Distance for {type}")
    plot_heatmap(df, f"Real Euclidean Distance for {type}")




