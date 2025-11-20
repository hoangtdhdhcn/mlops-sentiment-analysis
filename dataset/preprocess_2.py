# Handling the commas in csv format

import pandas as pd

def preprocess_comments(input_file):
    # Load the dataset
    df = pd.read_csv(input_file)

    # Strip leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Replace commas inside the 'Comment' column with a semicolon
    df['Comment'] = df['Comment'].str.replace(',', ';', regex=False)  # Replace commas with a semicolon


    # Save the cleaned dataset
    output_file = "preprocessed/cleaned_sentiment_analysis_part1.csv"
    df.to_csv(output_file, index=False)
    
    return df

input_file = "preprocessed/sentiment_analysis_part1.csv"
cleaned_df = preprocess_comments(input_file)

