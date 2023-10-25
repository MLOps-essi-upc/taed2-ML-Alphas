import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import pickle
import mlflow
import mlflow.pytorch
from PIL import Image
import pandas as pd
import io
import gc
import numpy as np

from configuration_test import sample_dataset
from configuration_test import train_data_loader

"""Check if the image is an array or a tensor"""
def image_is_valid(image):
    if torch.is_tensor(image):
        return True
    else:
        raise ValueError("Image is not a tensor")


"""Check if a given label is valid (either 0, 1, 2 or 3)"""
def label_is_valid(label):
    valid_labels = [0, 1, 2, 3]
    if label in valid_labels:
        return True
    else:
        raise ValueError("Label not in available categories")

"""Check if the sample dataset has images and labels, if the images are tensors,
if the labels are integers with the available categories"""
def check_images_and_labels(sample_data):
    assert len(sample_data.image_tensors) > 0, "Sample data should have images"
    assert len(sample_data.labels) > 0, "Sample data should have labels"

    for image in sample_data.image_tensors:
        assert image_is_valid(image)

    for label in sample_data.labels:
        assert label_is_valid(label)

"""Check if the train and validation loaders have any rows. """
def check_data_loaders(train_loader, valid_loader):
    assert len(train_loader) > 0, "Train loader should have rows"
    assert len(valid_loader) > 0, "Validation loader should have rows"


"""Perform different checks to examine if the data is loaded correctly"""
def test_data_loader():
    sample_data = sample_dataset()

    check_images_and_labels(sample_data)

    train_loader, valid_loader = train_data_loader(dataset=sample_data, batch_size=64)
    check_data_loaders(train_loader, valid_loader)
