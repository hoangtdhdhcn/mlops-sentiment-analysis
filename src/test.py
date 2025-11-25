# test.py

from models.Logistic_Regression import SentimentModel
from data_preprocessing.data_preprocessing import load_and_split_data
from evaluation.evaluation import evaluate_model
import joblib
import os
from utils.paths import PREPROCESSED_DIR, CHECKPOINT_DIR, RESULT_DIR
import json
import pandas as pd

def main():
    # Step 1: Load the test data (Reuse the same function for splitting data)
    input_file = os.path.join(PREPROCESSED_DIR,"cleaned_sentiment_analysis_part1.xlsx")  # Change this to dataset path
    X_train, X_test, y_train, y_test = load_and_split_data(input_file)

    # Step 2: Load the trained model
    model_filename = os.path.join(CHECKPOINT_DIR, "logreg_model.pkl")   # Replace with the correct model file path
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

    # Step 6: Save metrics to JSON
    output_metrics_path = os.path.join(RESULT_DIR, "eval_results.json")

    with open(output_metrics_path, "w") as f:
        json.dump({
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1_score": float(metrics["f1_score"]),
            "confusion_matrix": metrics["confusion_matrix"].tolist()
        }, f, indent=4)

    print(f"\nResults saved to {output_metrics_path}")

    # Step 7: Save predictions to CSV
    output_csv_path = os.path.join(RESULT_DIR, "predictions.csv")
    df_results = pd.DataFrame({
        "text": X_test,       # if X_test is a list or Series of text
        "true_label": y_test,
        "predicted_label": y_pred
    })
    df_results.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

if __name__ == "__main__":
    main()
