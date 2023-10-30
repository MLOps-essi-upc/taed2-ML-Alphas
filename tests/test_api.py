from http import HTTPStatus
import os
import sys
from io import BytesIO
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import pytest
from fastapi.testclient import TestClient
import io
from src.app.api import app


@pytest.fixture(scope="module", autouse=True)
def client():
    # Use the TestClient with a `with` statement to trigger the startup and shutdown events.
    with TestClient(app) as client:
        return client


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["data"]["message"] == "Welcome to the Alzheimer's Disease Presence Classifier, also called ADPC. Upload an image of the patient's brain to know the severity of the situation."



from PIL import Image
import io
import torch
from fastapi.testclient import TestClient

# ... (other imports and fixtures)

def test_model_prediction(client):
    # Use constants if fixture created
    #img_path =
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the image file name
    image_filename = "image_pil.png"

    # Construct the full path to the image file
    image_path = os.path.join(script_dir, image_filename)
    if os.path.isfile(image_path):
        _files = {'uploadFile': open("image_pil.png",'rb')}
        # Define the URL and additional headers
        url = "http://127.0.0.1:8000/models"
        response = client.post("/models", files={"file": ("filename", open(image_filename, "rb"), "image/jpeg")})

        assert response.status_code == 200
    else:
        pytest.fail("File does not exist.")

if __name__ == "__main__":
    pytest.main()
