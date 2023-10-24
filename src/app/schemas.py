from enum import Enum
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile


class ImageUpload(BaseModel):
    file: UploadFile
    model_conf: dict = {}  # You can specify a default value or leave it as an empty dictionary


class AlzheimerGrade(str, Enum):
    Mild_Demented = 0
    Moderate_Demented = 1
    Non_Demented = 2
    Very_Mild_Demented = 3