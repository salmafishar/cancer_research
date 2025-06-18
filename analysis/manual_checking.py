import pandas as pd
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler

# Step 1: Load and clean expression matrix
expr_df = pd.read_csv("../data/Mean_Intensity_Matrix_All_Drugs.csv", index_col=0)
expr_df = expr_df.fillna(0)
expr_df.columns = expr_df.columns.str.replace("_Mean_Intensity", "")

# Step 2: Transpose so drugs are rows (like in similarity methods)
expr_df_T = expr_df.T

# Step 3: Standardize drug vectors
scaler = StandardScaler()
standardized = scaler.fit_transform(expr_df_T)

# Step 4: Convert back to DataFrame to index by drug names
standardized_df = pd.DataFrame(standardized, index=expr_df_T.index, columns=expr_df_T.columns)

# Step 5: Select your drugs
drug1 = "Dactolisib"
drug2 = "Voxtalisib"

# Step 6: Calculate Euclidean distance
vec1 = standardized_df.loc[drug1]
vec2 = standardized_df.loc[drug2]
manual_distance = norm(vec1 - vec2)

print(f"Manual (standardized) Euclidean distance between {drug1} and {drug2}: {manual_distance:.5f}")

# Optional: Compare with precomputed matrix
euc_df = pd.read_csv("../data/euclidean_distance/euclidean_distance_real.csv", index_col=0)
precomputed = euc_df.loc[drug1, drug2]
print(f"Precomputed distance from matrix: {precomputed:.5f}")

