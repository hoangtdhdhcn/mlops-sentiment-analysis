# Split the raw dataset into two subdatasets

import pandas as pd

# === Step 1: Load CSV file ===
input_file = "raw/sentiment_analysis_dataset.csv"  # change this to file path
df = pd.read_csv(input_file, sep='\t')  # using tab since sample uses tabs

# === Step 2: (Optional) Shuffle the dataset to ensure randomness ===
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# === Step 3: Split into two equal halves ===
half = len(df) // 2
df_part1 = df.iloc[:half]
df_part2 = df.iloc[half:]

# === Step 4: Save to new CSV files ===
df_part1.to_csv("preprocessed/sentiment_analysis_part1.csv", index=False, sep='\t')
df_part2.to_csv("preprocessed/sentiment_analysis_part2.csv", index=False, sep='\t')

print(f"Split completed: {len(df_part1)} rows in part1, {len(df_part2)} rows in part2.")
