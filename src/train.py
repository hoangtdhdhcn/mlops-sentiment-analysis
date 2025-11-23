<<<<<<< HEAD
# # train.py
# from data_preprocessing.data_preprocessing import load_and_split_data
# from models.Logistic_Regression import SentimentModel
# from sklearn.metrics import accuracy_score
# from datetime import datetime
# import os
# from utils.paths import PREPROCESSED_DIR, CHECKPOINT_DIR


# def main():
#     # Step 1: Load and split data
#     input_file = os.path.join(PREPROCESSED_DIR,"cleaned_sentiment_analysis_part1.xlsx")  # Change this to dataset path
#     X_train, X_test, y_train, y_test = load_and_split_data(input_file)

#     # Step 2: Initialize and train the model
#     model = SentimentModel()

#     # Fit the model once
#     print("Training model...\n")
#     model.train(X_train, y_train)

#     # Step 3: Monitor accuracy after training
#     y_train_pred = model.predict(X_train)
#     train_accuracy = accuracy_score(y_train, y_train_pred)

#     print(f"Training Accuracy after fitting: {train_accuracy:.4f}")

#     # Step 4: Save the trained model with the timestamped filename
#     # timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  
#     # model_filename = os.path.join(CHECKPOINT_DIR, f"logreg_model_{timestamp}.pkl")
#     model_filename = os.path.join(CHECKPOINT_DIR, "logreg_model.pkl")

#     print("Saving model to:", model_filename)
#     # Save the model 
#     model.save(model_filename)

# if __name__ == "__main__":
#     main()




=======
>>>>>>> de940a5017a4206208bb5f7fbc05ee630da50691
# train.py
from data_preprocessing.data_preprocessing import load_and_split_data
from models.Logistic_Regression import SentimentModel
from sklearn.metrics import accuracy_score
from datetime import datetime
import os
from utils.paths import PREPROCESSED_DIR, CHECKPOINT_DIR
<<<<<<< HEAD
import mlflow
import mlflow.sklearn  # For logging sklearn models

def main():
    # Step 0: Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri("file:./mlruns")  # Change to MLflow server URI if needed
    mlflow.set_experiment("sentiment_analysis_logreg")

    # Start MLflow run
    with mlflow.start_run(run_name="sentiment_analysis_logreg"):
        # Step 1: Load and split data
        input_file = os.path.join(PREPROCESSED_DIR, "cleaned_sentiment_analysis_part1.xlsx")
        X_train, X_test, y_train, y_test = load_and_split_data(input_file)

        # Step 2: Initialize model
        model = SentimentModel()

        # Log model type
        mlflow.log_param("model_type", "LogisticRegression")

        # Step 3: Train model
        print("Training model...\n")
        model.train(X_train, y_train)

        # Step 4: Evaluate model
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Step 5: Save trained model locally
        model_filename = os.path.join(CHECKPOINT_DIR, "logreg_model.pkl")
        print("Saving model to:", model_filename)
        model.save(model_filename)

        # Step 6: Log model to MLflow
        mlflow.sklearn.log_model(
            model, name="logreg_model", registered_model_name="logreg_model"
        )

        print(f"MLflow run completed. Run ID: {mlflow.active_run().info.run_id}")
=======


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
>>>>>>> de940a5017a4206208bb5f7fbc05ee630da50691

if __name__ == "__main__":
    main()
