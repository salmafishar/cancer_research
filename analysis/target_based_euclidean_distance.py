import pandas as pd
from parsing_targets import drugs_per_target, drug_class
from itertools import combinations
from data_visualization import plot_heatmap, plot_histogram, plot_density, plot_boxplot_by_class


# loading euclidean_distance  & parsed targets matrix
euclidean_distance_drugs_df = pd.read_csv('../data/euclidean_distance/euclidean_distance_real.csv', index_col=0)
targets_df = pd.read_csv("../data/parsed_targets.csv")

# making sure the drug is in the csv
filtered_drugs = {}
for target, drug_name in drugs_per_target.items():
    filtered = []
    for drug in drug_name:
        if drug in euclidean_distance_drugs_df.index:
            filtered.append(drug)
    # Save result under target name
    filtered_drugs[target] = filtered


# computing inter & intra data
def computing_intra_target_distance(filtered_drugs, distance_matrix):
    # collecting rows to save csv
    rows = []
    for target, drug_list in filtered_drugs.items():
        if len(drug_list) < 2:
            continue
        for d1, d2 in combinations(drug_list, 2):
            distance = euclidean_distance_drugs_df.loc[d1, d2]
            rows.append([target, d1, d2, distance])
    return pd.DataFrame(rows, columns=["Target", "Drug 1", "Drug 2", "Distance"])


def compute_inter_target_distances(filtered_drugs, distance_matrix):
    rows = []
    targets = list(filtered_drugs.keys())
    for i in range(len(targets)):
        for j in range(i + 1, len(targets)):
            t1, t2 = targets[i], targets[j]
            drugs1, drugs2 = filtered_drugs[t1], filtered_drugs[t2]
            for d1 in drugs1:
                for d2 in drugs2:
                    if d1 == d2:
                        continue
                    distance = distance_matrix.loc[d1, d2]
                    rows.append([t1, t2, d1, d2, distance])
    return pd.DataFrame(rows, columns=["Target 1", "Target 2", "Drug 1", "Drug 2", "Distance"])


# creating and saving the dataframe
intra_target_df = computing_intra_target_distance(filtered_drugs, euclidean_distance_drugs_df)
inter_target_df = compute_inter_target_distances(filtered_drugs, euclidean_distance_drugs_df)
intra_target_df.to_csv("../data/intra_target.csv", index=False)
inter_target_df.to_csv("../data/inter_target.csv", index=False)

# Plotting the global histograms
plot_histogram(intra_target_df["Distance"],
               title="Intra-Target Euclidean Distance",
               save_path="../plots/target_based_plots/intra_target_hist.png")

plot_histogram(inter_target_df["Distance"],
               title="Inter-Target Euclidean Distance",
               save_path="../plots/target_based_plots/inter_target_hist.png")


# per-target heatmaps & histograms
def save_per_target_plots(filtered_drugs, distance_matrix):
    for target, drugs in filtered_drugs.items():
        if len(drugs) < 2:
            continue

        # Histogram
        pairs = combinations(drugs, 2)
        distances = [distance_matrix.loc[d1, d2] for d1, d2 in pairs]
        hist_path = f"../plots/target_based_plots/histograms/intra_targets/{target}_intra_hist.png"
        plot_histogram(distances, title=f"Intra-Target Distance: {target}", save_path=hist_path)

        # Heatmap
        submatrix = distance_matrix.loc[drugs, drugs]
        heatmap_path = f"../plots/target_based_plots/heatmaps/intra_targets/{target}_heatmap.png"
        plot_heatmap(submatrix, title=f"Euclidean Distance for {target}", save_path=heatmap_path)


# getting target-level summary
def get_target_summary(intra_target_df):
    summary = intra_target_df.groupby("Target")["Distance"].agg(
        Count="count",
        Mean="mean",
        Std="std",
        Min="min",
        Max="max"
    ).reset_index()

    return summary


summary_df = get_target_summary(intra_target_df)
summary_df.to_csv("../data/target_summary.csv", index=False)

plot_heatmap(
    summary_df.set_index("Target")[["Mean"]],
    title="Mean Intra-Target Distance per Target",
    save_path="../plots/target_based_plots/heatmaps/intra_target_mean_heatmap.png",
    figsize=(10, 40)
)

# density for targets with 5 or more drugs
for target, drugs in filtered_drugs.items():
    if len(drugs) < 5:
        continue
    distance = [euclidean_distance_drugs_df.loc[d1, d2]
                for d1, d2 in combinations(drugs, 2)
                ]
plot_density(distance, title=f"Distance Density for {target}",
             save_path=f"../plots/target_based_plots/density/{target}_intra_density.png")

# density for inter data
plot_density(
    data=inter_target_df["Distance"],
    title="Inter-Target Distance Density",
    save_path="../plots/target_based_plots/density/inter_target_density.png"
)

# Load preprocessed intra_target with class
intra_with_class = pd.read_csv("../data/intra_target_with_class.csv")
# Plot boxplot grouped by class
plot_boxplot_by_class(
    intra_with_class,
    save_path="../plots/target_based_plots/boxplots/intra_target_by_class.png"
)

# Compute mean distance per class and plot heatmap
class_means = intra_with_class.groupby("Class")["Distance"].mean().reset_index().set_index("Class")
plot_heatmap(class_means,
             title="Mean Intra-Target Distance by Drug Class",
             save_path="../plots/target_based_plots/heatmaps/intra_target_by_class_heatmap.png")
