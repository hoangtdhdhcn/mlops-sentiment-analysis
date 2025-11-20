import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(input_file, test_size=0.2, random_state=42):
    """
    Loads the dataset and splits it into training and test sets.
    Args:
    - input_file: Path to the Excel file.
    - test_size: Proportion of the data to be used for testing.
    - random_state: Random seed for reproducibility.
    
    Returns:
    - X_train, X_test, y_train, y_test: Training and testing data splits.
    """
    # Load the dataset from the Excel file
    df = pd.read_excel(input_file)

    # Split into features and target
    X = df['Comment']
    y = df['Sentiment']

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test
