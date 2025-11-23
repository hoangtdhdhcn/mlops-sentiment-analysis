# model.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

class SentimentModel:
    def __init__(self):
        """
        Initializes the sentiment analysis model with a pipeline
        consisting of a TF-IDF vectorizer and a Logistic Regression classifier.
        """
        self.pipeline = make_pipeline(
            TfidfVectorizer(stop_words='english'),  # Convert text to TF-IDF features
            LogisticRegression(max_iter=1000)  # Logistic Regression classifier
        )

    def train(self, X_train, y_train):
        """
        Trains the Logistic Regression model.
        Args:
        - X_train: Features (text data) for training.
        - y_train: Target labels (sentiment) for training.
        """
        self.pipeline.fit(X_train, y_train)
        print("Model training complete!")

    def predict(self, X):
        """
        Predicts sentiment labels for new text data.
        Args:
        - X: List, array, or pandas Series of text data.
        
        Returns:
        - Array of predicted sentiment labels.
        """
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """
        Predicts sentiment probabilities for new text data.
        Args:
        - X: List, array, or pandas Series of text data.
        
        Returns:
        - Array of predicted probabilities for each class.
        """
        return self.pipeline.predict_proba(X)

    def save(self, model_filename):
        """
        Saves the trained model to a file.
        Args:
        - model_filename: Path to save the trained model.
        """
        import joblib
        joblib.dump(self.pipeline, model_filename)
        print(f"Model saved to {model_filename}")
    
    def load(self, model_filename):
        """
        Loads a saved model from a file.
        Args:
        - model_filename: Path to the model file.
        """
        import joblib
        self.pipeline = joblib.load(model_filename)
        print(f"Model loaded from {model_filename}")
