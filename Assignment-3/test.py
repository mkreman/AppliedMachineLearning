import subprocess
import os
import time
import signal
import json
import joblib
from score import score
import requests
import warnings
warnings.filterwarnings("ignore")


model = joblib.load("./../Assignment-2/mlruns/108559129931990894/affe3afbb88d4478959b5404ac51289e/artifacts/svc/model.pkl")

def test_smoke():
    """Check if function runs without crashing."""
    score("Test message", model, 0.5)

def test_output_format():
    """Check if output is of expected types."""
    prediction, propensity = score("Test message", model, 0.5)
    assert isinstance(prediction, bool)
    assert isinstance(propensity, float)

def test_prediction_values():
    """Check if prediction is always 0 or 1."""
    prediction, _ = score("Test message", model, 0.5)
    assert prediction in [True, False]

def test_propensity_range():
    """Check if propensity score is between 0 and 1."""
    _, propensity = score("Test message", model, 0.5)
    assert 0.0 <= propensity <= 1.0

def test_threshold_zero():
    """Check if threshold=0 always predicts 1."""
    prediction, _ = score("Test message", model, 0.0)
    assert prediction is True

def test_threshold_one():
    """Check if threshold=1 always predicts 0."""
    prediction, _ = score("Test message", model, 1.0)
    assert prediction is False

def test_obvious_spam():
    """Check if an obvious spam message is predicted as spam (1)."""
    prediction, _ = score("Congratulations! You've won a $1,000 gift card! Click here to claim your prize now!", model, 0.5)
    assert prediction is True

def test_obvious_ham():
    """Check if an obvious non-spam message is predicted as non-spam (0)."""
    prediction, _ = score("Hello, how are you doing today?", model, 0.5)
    assert prediction is False

def test_flask():
    # Start Flask server
    process = subprocess.Popen(["python", "./app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)

    response = requests.get("http://127.0.0.0:5000/")
    assert response.status_code == 200

    # Send POST request to server with JSON header
    output = requests.post(
        "http://127.0.0.0:5000/predict",
        data={"text": "Test is a message", "threshold": "0.5"},
        headers={"Accept": "application/json"}
    )
    
    # Parse the output and check if it has the expected keys and types
    try:
        response_json = output.json()
        assert "prediction" in response_json
        assert "propensity" in response_json
        assert isinstance(response_json["prediction"], bool)
        assert isinstance(response_json["propensity"], float)
    except json.JSONDecodeError:
        print("Failed to parse JSON response")
        assert False

    # Stop Flask server
    os.kill(process.pid, signal.SIGTERM)
    process.wait()


if __name__ == "__main__":
    # test_smoke()
    # test_output_format()
    # test_prediction_values()
    # test_propensity_range()
    # test_threshold_zero()
    # test_threshold_one()
    # test_obvious_spam()
    # test_obvious_ham()
    # test_flask()
    pass