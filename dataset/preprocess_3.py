# Handling the missing values

import pandas as pd

def preprocess_data(input_file, output_file, fill_missing=True, placeholder='unknown'):
    """
    Preprocesses the dataset by handling missing values in the 'Comment' and 'Sentiment' columns.
    
    Args:
    - input_file: Path to the input CSV or Excel file.
    - output_file: Path to save the processed file (CSV or Excel).
    - fill_missing: Whether to fill missing values with a placeholder (True) or drop them (False).
    - placeholder: The placeholder text to fill missing values with.
    
    Returns:
    - Processed pandas DataFrame.
    """
    # Load the dataset (adjust to read from CSV or Excel depending on file format)
    try:
        # Try reading an Excel file first
        df = pd.read_excel(input_file)
        print(f"Loaded Excel file: {input_file}")
    except Exception as e:
        # If that fails, try reading a CSV file
        df = pd.read_csv(input_file)
        print(f"Loaded CSV file: {input_file}")

    # Check for missing values in the 'Comment' column
    if df['Comment'].isnull().any():
        print(f"Warning: Found missing values in the 'Comment' column.")

        # Option 1: Fill missing values with a placeholder
        if fill_missing:
            print(f"Filling missing values in 'Comment' with '{placeholder}'.")
            df['Comment'] = df['Comment'].fillna(placeholder)
        # Option 2: Drop rows with missing 'Comment' values
        else:
            print("Dropping rows with missing values in the 'Comment' column.")
            df = df.dropna(subset=['Comment'])

    # Check for missing values in the 'Sentiment' column
    if df['Sentiment'].isnull().any():
        print(f"Warning: Found missing values in the 'Sentiment' column.")

        # Option 1: Fill missing values with a placeholder
        if fill_missing:
            print(f"Filling missing values in 'Sentiment' with '{placeholder}'.")
            df['Sentiment'] = df['Sentiment'].fillna(placeholder)
        # Option 2: Drop rows with missing 'Sentiment' values
        else:
            print("Dropping rows with missing values in the 'Sentiment' column.")
            df = df.dropna(subset=['Sentiment'])

    # Optionally save the cleaned dataset to a new file (CSV or Excel)
    if output_file.endswith('.csv'):
        df.to_csv(output_file, index=False)
        print(f"Saved cleaned data to: {output_file}")
    elif output_file.endswith('.xlsx'):
        df.to_excel(output_file, index=False)
        print(f"Saved cleaned data to: {output_file}")
    
    # Return the cleaned dataframe
    return df

if __name__ == "__main__":
    input_file = "preprocessed/sentiment_analysis_part2.xlsx"  # Path to raw dataset
    output_file = "preprocessed/cleaned_sentiment_analysis_part2.xlsx"  # Path to save cleaned dataset
    
    # Preprocess the data and handle missing values (fill with 'unknown' by default)
    cleaned_df = preprocess_data(input_file, output_file, fill_missing=True, placeholder='unknown')
