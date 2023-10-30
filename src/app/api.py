
import pickle
import numpy as np
import os
from io import BytesIO
from PIL import Image
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import List
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from pathlib import Path
from src.app.schemas import ResNet, ResidualBlock



# Define application
app = FastAPI(
    title="Alzheimer's Disease presence classification API",
    description="This API lets you classify the presence of Alzheimer's disease in a brain image",
    version="1.0",
)

def construct_response(f):
    @wraps(f)
    async def wrap(request: Request, *args, **kwargs):
        results = await f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now(),
            "url": request.url._url,
        }
        # Add data
        if "data" in results:
            response["data"] = results["data"]


        return response

    return wrap



@app.on_event("startup")
def _load_models():
    model = ResNet(ResidualBlock,[3,4,6,3])
    model.load_state_dict(torch.load(os.path.abspath("../taed2-ML-Alphas/models/RESNET_0.zip"), map_location=torch.device('cpu'))) # This line uses .load() to read a .pth file and load the network weights on to the architecture.
    return model



@app.get("/", tags=["General"])  # path operation decorator
@construct_response
async def _index(request: Request):
    """Root endpoint."""

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to the Alzheimer's Disease Presence Classifier, also called ADPC. Upload an image of the patient's barin to know the severity of the situation."},
    }

    return response



def get_presence(presence):
    """Convert a label ID to its corresponding name."""
    if presence[0].item()==0:
        return "Mild Demented"
    if presence[0].item()==1:
        return "Moderate Demented"
    if presence[0].item()==2:
        return "Non Demented"
    if presence[0].item()==3:
        return "Very Mild Demented"
        

@app.post("/models", tags=["Prediction"])
@construct_response
async def _predict(request : Request, file : UploadFile):  # Change payload to accept image file
    image_bytes = await file.read()
    stream = BytesIO(image_bytes)
    image = Image.open(stream)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to (224, 224)
        transforms.ToTensor(),          # Convert PIL Image to tensor
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)

    model = _load_models()
    output = model(image)
    output = torch.softmax(output, dim=1)  
    probs, idxs = output.topk(1) 
    presence= get_presence(idxs)

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "Alzheimer presence": presence,
        },
    }

    return response
