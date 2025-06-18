import pandas as pd
from plotting import plot_density, plot_histogram, plot_heatmap
from similarity_methods import *
from ftp import get_files
import re

# load table with drug info (types and targets)
drug_info = pd.read_csv("drug_info_supp_table.csv").rename(columns={"Targets (acc to drugbank and vendors)" : "Targets"}).loc[:,["Drug", "Type", "Targets"]]
drug_info["Drug"] = drug_info["Drug"].str.strip()
all_drugs = drug_info["Drug"].to_list()

# load and get cleaned intensity matrix
df_cleaned, gene_names = clean_data(filepath='data/Mean_Intensity_Matrix_All_Drugs.csv')

# create og similarity/distance matrixes 
og_cos_matrix = calculate_cosine_similarity(df_cleaned, "cosine_similarity_real.csv")
og_euc_matrix = calculate_euclidean_distance(df_cleaned, "euclidean_distance_real.csv")

# plot real data 
plot_density(og_euc_matrix, "Real Euclidean Distance", "plots/Density/og_euclidean")
plot_heatmap(og_euc_matrix, "Real Euclidean Distance", "plots/Heatmaps/og_euclidean")

# randomize og matrix, get randomized distance matrix
random_df = randomize_dataframe(df_cleaned)
random_euc_matrix = calculate_euclidean_distance(random_df, "euclidean_distance_random.csv")
# plot randomized data
plot_density(random_euc_matrix, "Randomized Euclidean Distance", "plots/Density/random_euclidean")
plot_heatmap(random_euc_matrix, "Randomized Euclidean Distance", "plots/Heatmaps/random_euclidean")

# perturb og matrix, get perturbed distance matrix
perturbed_df = perturb_dataframe(df_cleaned, scale=1e5)
perturbed_euc_matrix = calculate_euclidean_distance(perturbed_df, "euclidean_distance_perturbed.csv")
# plot perturbed data
plot_density(perturbed_euc_matrix, "Perturbed Euclidean Distance", "plots/Density/perturbed_euclidean")
plot_heatmap(perturbed_euc_matrix, "Perturbed Euclidean Distance", "plots/Heatmaps/perturbed_euclidean")


# working with types 

# create dictionary of types + drugs 
drug_types = drug_info.groupby("Type")["Drug"].apply(list).to_dict()

# create dictionary of types + euclidean distance dfs
euclidean_by_type = {}
for type in drug_types:
    euclidean_by_type[type] = get_sub_matrix(drug_types[f"{type}"], og_euc_matrix) 

# plots by type
for type, df in euclidean_by_type.items():
    plot_density(df, f"Euclidean Distance for {type}", f"plots/by_type/{type.replace(' ', '_')}")
    plot_heatmap(df, f"Euclidean Distance for {type}", f"plots/by_type/{type.replace(' ', '_')}")

# get all drugs ordered by type + lengths of each type
ordered_drugs = []
type_lengths = []
type_labels = list(drug_types.keys())
for drugs in drug_types.values():
    ordered_drugs.extend(drugs)
    type_lengths.append(len(drugs))
# order euc matrix by drug type
ordered_euclidean = og_euc_matrix.loc[ordered_drugs, ordered_drugs]

# plot heatmap with grid lines separating each type
plot_heatmap(ordered_euclidean, "Euclidean Distances Ordered By Type", "plots/by_type/ordered", type_lengths, type_labels)


get_files(all_drugs, "Normalized_Intensity_0", "data/Normalized_Intensity_0_all_drugs")
get_files(all_drugs, "Normalized_Intensity_1", "data/Normalized_Intensity_1_all_drugs")
get_files(all_drugs, "Normalized_Intensity_10", "data/Normalized_Intensity_10_all_drugs")
get_files(all_drugs, "Normalized_Intensity_100", "data/Normalized_Intensity_100_all_drugs")

