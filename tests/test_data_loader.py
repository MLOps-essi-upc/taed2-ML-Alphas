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

def image_is_valid(image):
    if isinstance(image, torch.Tensor):
        return True
    else:
        raise ValueError("Image is not a PyTorch tensor")


"""Check if a given label is valid (either 0, 1, 2 or 3)"""
def label_is_valid(label):
    valid_labels = [0, 1, 2, 3]
    if label in valid_labels:
        return True
    else:
        raise ValueError("Label not in available categories")


"""Perform different checks to examine if the data is loaded correctly.
Check if the sample dataset as images and labels, if the images are tensors,
if the labels are integers with the available categories and if the train and
validation loaders have any rows. """
def test_data_loader():
    sample_data = sample_dataset()

    # check that the data has images and labels
    assert len(sample_data.image_tensors) > 0, "Train_loader should have images"
    assert len(sample_data.labels) > 0, "Train_loader should have labels"

    # check that the images are torch tensors
    for image in sample_data.image_tensors:
        assert image_is_valid(image)

    # check that the labels are integers with available categories
    for label in sample_data.labels:
        assert label_is_valid(label)


    train_loader, valid_loader = train_data_loader(dataset=sample_data, batch_size=64)

    # check the loaded data has rows
    assert len(train_loader) > 0, "Train_loader should have more rows"
    assert len(valid_loader) > 0, "Valid_loader should have more rows"
