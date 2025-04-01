import os
import signal
import time
import subprocess
import json
import requests


def test_docker():
    # Stop and remove any existing Docker container and image with the name 'spam_classification_app'
    subprocess.run(["docker", "stop", "spam_classification_app"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(["docker", "rm", "spam_classification_app"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(["docker", "rmi", "spam_classification_app"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Build the Docker image using the Dockerfile in the Assignment-4 directory
    process = subprocess.Popen(["/bin/bash"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    process.communicate(
        """cd ..\ndocker build -t spam_classification_app -f Assignment-4/Dockerfile ."""
    )
    
    # Run the Docker container in detached mode, exposing port 5000
    subprocess.run(["docker", "run", "-d", "-p", "5000:5000", "--name", "spam_classification_app", "spam_classification_app"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Wait for the server to start
    time.sleep(10)

    # Test if the Flask server is running by sending a GET request to the root endpoint
    response = requests.get("http://127.0.0.1:5000/")
    assert response.status_code == 200

    # Send a POST request to the '/predict' endpoint with test data
    data = {"text": "Test is a message", "threshold": "0.5"}
    output = requests.post(
        "http://127.0.0.1:5000/score",
        data=json.dumps(data),
        headers={"Content-Type": "application/json"}
    )
    
    # Parse the output and validate the response structure
    response_json = output.json()
    assert "prediction" in response_json  # Ensure 'prediction' key exists
    assert "propensity" in response_json  # Ensure 'propensity' key exists
    assert isinstance(response_json["prediction"], bool)  # Check type of 'prediction'
    assert isinstance(response_json["propensity"], float)  # Check type of 'propensity'

    # Stop and remove the Docker container and image after testing
    subprocess.run(["docker", "stop", "spam_classification_app"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(["docker", "rm", "spam_classification_app"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(["docker", "rmi", "spam_classification_app"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
