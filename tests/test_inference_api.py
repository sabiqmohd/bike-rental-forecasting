import subprocess
import time
import requests
import signal
import sys

import pytest

@pytest.fixture(scope="module")
def inference_api_server():
    """
    Fixture to start the Flask inference API server in a subprocess before tests,
    and terminate it after tests.
    """
    # Start the Flask API
    proc = subprocess.Popen(
        [sys.executable, "app-ml/entrypoint/inference_api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Give the server time to start
    time.sleep(5)
    yield proc
    # Teardown: terminate the server process
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

def test_inference_api_prediction(inference_api_server):
    """
    Test Case 2: Send a request to the /predict endpoint and assert response.
    """
    url = "http://localhost:5001/predict"
    response = requests.get(url)
    # Check HTTP 200 response
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    json_data = response.json()
    # The response should contain a 'prediction' key with a float value
    assert "prediction" in json_data, "Response JSON missing 'prediction' key"
    prediction = json_data["prediction"]
    # Check that the prediction is a float or int (numeric)
    assert isinstance(prediction, (float, int)), f"Prediction is not a float or int: {prediction}"