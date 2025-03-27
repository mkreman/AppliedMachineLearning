from sklearn.base import BaseEstimator
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings("ignore")


# Load the training data and fit a bag-of-words transformer
train_df = pd.read_csv('./../Assignment-1/sms+spam+collection/train.csv')
train_df.dropna(inplace=True)
bow_transformer = CountVectorizer(analyzer=lambda x: x.split()).fit(train_df['processed sms'])


def score(text: str, model: BaseEstimator, threshold: float) -> tuple[bool, float]:
    """Scores a trained model on a given text input."""
    if not isinstance(text, str):
        raise ValueError("Text input must be a string")
    if not isinstance(model, BaseEstimator):
        raise ValueError("Model must be a scikit-learn estimator")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("Threshold must be between 0 and 1")
    
    encoded_text = bow_transformer.transform([text])
    propensity = model.predict_proba(encoded_text)[0, 1]
    prediction = bool(propensity >= threshold)
    return prediction, float(propensity)



if __name__ == "__main__":
    # Load the best trained model from file
    import joblib
    model = joblib.load("./../Assignment-2/mlruns/108559129931990894/affe3afbb88d4478959b5404ac51289e/artifacts/svc/model.pkl")
    
    # Score some example messages
    message = "Congratulations! You've won a $1,000 gift card! Click here to claim your prize now!"
    
    prediction, propensity = score(message, model, 0.5)
    print(f"Message: {message}")
    print(f"Prediction: {prediction}")
    print(f"Propensity: {propensity:.2f}\n")