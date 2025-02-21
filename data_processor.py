import pandas as pd

def preprocess_data(input_file, output_file):
    # Load dataset
    df = pd.read_csv(input_file)

    # Drop 'Count of Disease Occurrence' if it exists
    if 'Count of Disease Occurrence' in df.columns:
        df.drop(columns=['Count of Disease Occurrence'], inplace=True)

    # Forward fill missing disease values
    df['Disease'] = df['Disease'].ffill()

    # Group symptoms under each disease
    df_grouped = df.groupby('Disease')['Symptom'].apply(list).reset_index()

    # Remove duplicate symptoms per disease
    df_grouped['Symptom'] = df_grouped['Symptom'].apply(lambda x: list(set(str(symptom) for symptom in x if pd.notna(symptom))))

    # One-hot encode symptoms
    all_symptoms = sorted(set(symptom for symptoms in df_grouped['Symptom'] for symptom in symptoms))
    df_encoded = pd.DataFrame(0, index=df_grouped.index, columns=all_symptoms)
    for i, symptoms in enumerate(df_grouped['Symptom']):
        df_encoded.loc[i, symptoms] = 1

    # Add Disease column back
    df_final = pd.concat([df_grouped['Disease'], df_encoded], axis=1)

    # Save processed data
    df_final.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

# Example usage
input_file = r"D:\project\raw_data_2.csv"  # Use raw string
output_file = r"D:\project\processed_data.csv"
preprocess_data(input_file, output_file)
