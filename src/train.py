# train.py
from data_preprocessing.data_preprocessing import load_and_split_data
from models.Logistic_Regression import SentimentModel
from sklearn.metrics import accuracy_score
from datetime import datetime
import os
from utils.paths import PREPROCESSED_DIR, CHECKPOINT_DIR


def main():
    # Step 1: Load and split data
    input_file = os.path.join(PREPROCESSED_DIR,"cleaned_sentiment_analysis_part1.xlsx")  # Change this to dataset path
    X_train, X_test, y_train, y_test = load_and_split_data(input_file)

    # Step 2: Initialize and train the model
    model = SentimentModel()

    # Fit the model once
    print("Training model...\n")
    model.train(X_train, y_train)

    # Step 3: Monitor accuracy after training
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    print(f"Training Accuracy after fitting: {train_accuracy:.4f}")

    # Step 4: Save the trained model with the timestamped filename
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  
    model_filename = os.path.join(CHECKPOINT_DIR, f"logreg_model_{timestamp}.pkl")

    print("Saving model to:", model_filename)
    # Save the model 
    model.save(model_filename)

if __name__ == "__main__":
    main()
