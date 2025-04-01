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
    encoded_text = bow_transformer.transform([text])
    propensity = model.predict_proba(encoded_text)[0, 1]
    prediction = bool(propensity >= threshold)
    return prediction, float(propensity)
