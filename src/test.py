# test.py

from models.Logistic_Regression import SentimentModel
from data_preprocessing.data_preprocessing import load_and_split_data
from evaluation.evaluation import evaluate_model
import joblib
import os
from utils.paths import PREPROCESSED_DIR, CHECKPOINT_DIR

def main():
    # Step 1: Load the test data (Reuse the same function for splitting data)
    input_file = os.path.join(PREPROCESSED_DIR,"cleaned_sentiment_analysis_part1.xlsx")  # Change this to dataset path
    X_train, X_test, y_train, y_test = load_and_split_data(input_file)

    # Step 2: Load the trained model
    model_filename = 'checkpoints/logreg_model_2025-11-20_09-57-20.pkl'  # Replace with the correct model file path
    model = SentimentModel()
    model.load(model_filename)

    # Step 3: Predict on the test set
    y_pred = model.pipeline.predict(X_test)

    # Step 4: Evaluate the model
    metrics = evaluate_model(y_test, y_pred)

    # Step 5: Print the evaluation metrics
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

if __name__ == "__main__":
    main()
