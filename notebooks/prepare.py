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





def prepare_data(dataset,transform):
    dataset2 = [[0,''] for i in range(len(dataset)*2)]

    for i in range(0,len(dataset)):
        image_bytes = dataset.iloc[i]['image']['bytes'] # Get bytes
        image_pil = Image.open(io.BytesIO(image_bytes)) # Convert bytes to PIL Image
        aug_image = tf.image.stateless_random_brightness(image_pil, max_delta=np.random.rand(), seed=(3,0))
        pil_format = Image.fromarray(aug_image.numpy())
        dataset2[2*i] = [image_pil,dataset.iloc[i]['label']]
        dataset2[2*i+1] = [pil_format,dataset.iloc[i]['label']]
        print("hola")
    return dataset2


def main(repo_path):
    data_path = repo_path / "data"
    train_path = data_path / "raw/train"
    test_path = data_path / "raw/val"
    train_files, train_labels = prepare_data(train_path)
    test_files, test_labels = prepare_data(test_path)
    prepared = data_path / "prepared"
    train_files.to_csv(prepared/"train.csv")
    test_files.to_csv(prepared/"test.csv")

main("/Users/aliag/taed-ML-Alphas/")