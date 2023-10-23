from PIL import Image
import io
import gc
import numpy as np
import torch
import torch.nn as nn
#from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from tensorflow.keras import layers
from pathlib import Path

def prepare_data(dataset, data_augmentation):
    dataset2 = [[0,''] for i in range(len(dataset)*2)]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to (224, 224)
        transforms.ToTensor(),          # Convert PIL Image to tensor
    ])

    if data_augmentation:
        for i in range(0,len(dataset)):
            image_bytes = dataset.iloc[i]['image']['bytes'] # Get bytes
            image_pil = Image.open(io.BytesIO(image_bytes)) # Convert bytes to PIL Image
            aug_image = tf.image.stateless_random_brightness(image_pil, max_delta=np.random.rand(), seed=(3,0))
            pil_format = Image.fromarray(aug_image.numpy())
            dataset2[2*i] = [transform(image_pil),dataset.iloc[i]['label']]
            dataset2[2*i+1] = [transform(pil_format),dataset.iloc[i]['label']]
        df = pd.DataFrame(dataset2)
        return df
    else:
        for i in range(0,len(dataset)):
            image_bytes = dataset.iloc[i]['image']['bytes'] # Get bytes
            image_pil = Image.open(io.BytesIO(image_bytes)) # Convert bytes to PIL Image
            dataset2[i] = [transform(image_pil),dataset.iloc[i]['label']]
        df = pd.DataFrame(dataset2)
        return dataset2


def main(repo_path):
    data_augmentation=False
    data_path = repo_path + "data"
    train_path = data_path + "raw/train"
    test_path = data_path + "raw/val"
    train_files = prepare_data(train_path, data_augmentation)
    test_files = prepare_data(test_path, data_augmentation)
    prepared = data_path + "prepared"
    train_files.to_csv(prepared + "/train.csv")
    test_files.to_csv(prepared + "/test.csv")

main("/Users/aliag/taed-ML-Alphas/")