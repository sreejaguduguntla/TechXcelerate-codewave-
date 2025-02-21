import pandas as pd
import pickle

# Load raw data
df = pd.read_csv("raw_data_2.csv")

# Drop unnecessary columns
df.drop(columns=["Count of Disease Occurrence"], inplace=True, errors="ignore")

# Forward fill missing diseases (assuming symptoms below a disease belong to it)
df["Disease"] = df["Disease"].ffill()

# Remove duplicate symptom entries for each disease
df.drop_duplicates(inplace=True)

# Group symptoms per disease
df_grouped = df.groupby("Disease")["Symptom"].apply(set).reset_index()

# Convert symptoms to strings & remove NaNs
df_grouped["Symptom"] = df_grouped["Symptom"].apply(lambda x: [str(s) for s in x if pd.notna(s)])

# One-hot encode symptoms
all_symptoms = sorted(set(symptom for symptoms in df_grouped["Symptom"] for symptom in symptoms))
df_encoded = pd.DataFrame(0, index=df_grouped.index, columns=all_symptoms)

for i, symptoms in enumerate(df_grouped["Symptom"]):
    df_encoded.loc[i, list(symptoms)] = 1  # Use list() to prevent reindexing errors

# Add Disease column back
df_final = pd.concat([df_grouped["Disease"], df_encoded], axis=1)

# Save processed dataset
df_final.to_csv("processed_data.csv", index=False)

# Save feature names for later use
feature_names = list(df_encoded.columns)
with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("✅ Data processing complete! Processed data saved as 'processed_data.csv'.")
