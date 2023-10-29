from PIL import Image
import io
import os
import gc
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from tensorflow.keras import layers
from pathlib import Path
import pickle
import sys

def prepare_data(dataset_path, data_augmentation=False):
    dataset = pd.read_parquet(dataset_path)

    num_instances = len(dataset)
    if data_augmentation:
        num_instances *=2
    dataset2 = [[0,''] for i in range(num_instances)]
    images = ['*' for i in range(num_instances)]
    labels = [-1 for i in range(num_instances)]

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
            images[2*1] = transform(image_pil)
            labels[2*i] = dataset.iloc[i]['label']
            dataset2[2*i+1] = [transform(pil_format),dataset.iloc[i]['label']]
            images[2*i+1] = transform(pil_format)
            labels[2*i+1] = dataset.iloc[i]['label']
    else:
        for i in range(0,len(dataset)):
            image_bytes = dataset.iloc[i]['image']['bytes'] # Get bytes
            image_pil = Image.open(io.BytesIO(image_bytes)) # Convert bytes to PIL Image
            dataset2[i] = [transform(image_pil),dataset.iloc[i]['label']]
            images[i] = transform(image_pil)
            labels[i] = dataset.iloc[i]['label']
    #df = pd.DataFrame(dataset2)
    #df.columns = ["Image", "Label"]
    return images,labels


def prepare(test_path, train_path):

    data_augmentation=False

    # prepare the data
    train_img,train_lab = prepare_data(train_path+'/train.parquet', data_augmentation)
    test_img,test_lab = prepare_data(test_path+'/test.parquet')

    # Debugging: Print the lengths of the data
    print("Train data length:", len(train_img))
    print("Test data length:", len(test_img))

    # output path (prepared)
    prepared_path_train = "data/prepared_data/train"
    prepared_path_test =  "data/prepared_data/test"


    #FOLDER_EXISTS = os.path.exists(prepared_path_train)
    #if not FOLDER_EXISTS:
    #    os.mkdir(prepared_path_train)

    #FOLDER_EXISTS = os.path.exists(prepared_path_test)
    #if not FOLDER_EXISTS:
    #    os.mkdir(prepared_path_test)

    with open (prepared_path_test+'/test.pkl','wb') as file:
       pickle.dump((test_img,test_lab), file)
    
    with open (prepared_path_train+'/train.pkl','wb') as file:
       pickle.dump((train_img,train_lab), file)


prepare(sys.argv[1],sys.argv[2])