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
from configuration_test import specify_resnet
from configuration_test import save_model
from configuration_test import do_experiment


"""Check if the training loss does not increase significantly after some epochs"""
def loss_is_not_increasing_significantly(epoch_losses):
    for i in range(len(epoch_losses) - 1):
        # only check when the model has had time to learn something
        if i > 6:
            loss_increase = epoch_losses[i + 1] - epoch_losses[i]
            # check the training loss does not increase by more than 10% between epochs
            if (loss_increase / epoch_losses[i - 1]) > 0.1:
                raise ValueError("Training loss increased significantly")
    return True


"""Check if the last training loss is smaller than the first one"""
def loss_decreased(epoch_losses):
    if epoch_losses[1] < epoch_losses[len(epoch_losses)-1]:
        return ValueError("Training loss should have decreased")
    else:
        return True


"""Test the model training: if the training loss and validation accuracy are not
None, that the training model improves (loss decreases over the epochs) and that
the accuracy has reasonable values (between 0 and 100%)"""
def test_training_model():
    sample_data = sample_dataset()
    train_loader, valid_loader = train_data_loader(dataset=sample_data, batch_size=64)
    model, params, criterion, optimizer = specify_resnet()

    total_step = len(train_loader)
    with mlflow.start_run():
        train_loss_values, validation_accuracy = do_experiment(train_loader,valid_loader,model,params,criterion,optimizer,str(1))

        # check that the training losses and validation accuracy are not None
        assert train_loss_values is not None, "Train loss should not be None"
        assert validation_accuracy is not None, "Validation accuracy should not be None"

        # check if loss is decreasing (training model is improving)
        assert loss_is_not_increasing_significantly(train_loss_values)
        assert loss_decreased(train_loss_values)

        # check accuracy has reasonable values (between 0 and 100)
        assert 0.0 <= validation_accuracy <= 100.0, "Validation accuracy should be between 0 and 100%"
