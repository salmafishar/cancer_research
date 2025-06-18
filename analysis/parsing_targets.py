import pandas as pd

# importing the drug-target csv
df = pd.read_csv("../data/drug_info_supp_table.csv")

# clean drug name
df['Drug'] = df["Drug"].str.strip()

# creating drug-class mapping
drug_class = dict(zip(df["Drug"], df["Type"]))

# creating the dictionary
drugs_per_target = {}

for _, row in df.iterrows():
    drug = row["Drug"]
    targets = row["Targets (acc to drugbank and vendors)"]

    # splitting the target string into list
    targets_list = [t.strip() for t in targets.split(",")]

    # filling the dictionary
    for target in targets_list:
        if target not in drugs_per_target:
            drugs_per_target[target] = []
        drugs_per_target[target].append(drug)

rows = [{"Target": t, "Drug": ",".join(d)} for t, d in drugs_per_target.items()]
pd.DataFrame(rows).to_csv("../data/parsed_targets.csv", index= False)

drug_class_df = pd.DataFrame(list(drug_class.items()), columns=["Drug", "Class"])
drug_class_df.to_csv("../data/drug_classes.csv", index=False)

# Create intra_target_with_class.csv if intra_target.csv exists
intra_target_df = pd.read_csv("../data/intra_target.csv")
intra_with_class = intra_target_df.copy()
intra_with_class["Class"] = intra_with_class["Drug 1"].map(drug_class)
intra_with_class = intra_with_class.dropna(subset=["Class"])
intra_with_class.to_csv("../data/intra_target_with_class.csv", index=False)
