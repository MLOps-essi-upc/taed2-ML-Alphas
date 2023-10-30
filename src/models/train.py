import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,datasets
from torch.utils.data.sampler import SubsetRandomSampler

import pickle
import mlflow
import mlflow.pytorch
from PIL import Image
import pandas as pd
import io
import gc
import numpy as np
import dagshub

from codecarbon import EmissionsTracker
from codecarbon import track_emissions


print("Establish connection")
dagshub.init("taed2-ML-Alphas", "aligator241", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/aligator241/taed2-ML-Alphas.mlflow")


class AlzheimerDataset(Dataset):
    def __init__(self,image_tensors,labels,transform=None):
        self.image_tensors = image_tensors
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        image = self.image_tensors[idx]
        label = self.labels[idx]

        return image,label


def train_data_loader(dataset,
                batch_size,
                random_seed=42,
                valid_size=0.2,
                shuffle=True):

    # load the dataset
    train_dataset = dataset
    valid_dataset = dataset

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 4):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

@track_emissions
def train(train_loader, model,criterion,optimizer,params,device):

    print("Start training")
    
    total = 0
    correct = 0
    for epoch in range(params['num_epochs']):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Compute accuracy
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

        # Compute and store train loss
        accuracy = 100 * correct/total
        mlflow.log_metric('train_accuracy',accuracy)
        mlflow.log_metric('train_loss',loss.item())
        print ('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                       .format(epoch+1, params['num_epochs'], loss.item(), accuracy))

    return model,optimizer

"""## Function to validate the model"""

def validation(valid_loader,model,device):

    print("Start validation")

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        mlflow.log_metric('val_acc',100*correct/total)
        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))

"""## Function to save the model"""

def save_model(model,optimizer,name):
    print("Save model "+ name)
    checkpoint = {'model': model,
              'state_dict': model.state_dict(),
              'optimizer' : optimizer.state_dict()}
    
    torch.save(checkpoint['state_dict'], name)
    # OPTION 1
    torch.save(checkpoint['model'],name+'.pth')
    torch.save(checkpoint,'checkpoint.pth')
        

"""## Define steps of the experiment"""


def main():

    tracker = EmissionsTracker()
    tracker.start()
    
    print('Logging')
    mlflow.autolog()
    mlflow.pytorch.autolog()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read train and test data
    print("Step1: Reading .pkl files")
    train_path_local = '../data/prepared_data/train.pkl'
    #train_path_kaggle = '/kaggle/input/images/train.pkl'
    with open(train_path_kaggle,'rb') as tr_file:
        image_tensors_tr,labels_tr = pickle.load(tr_file)

    # Create dataset objects
    print("Step2: Creating Dataset objects")
    dataset_train = AlzheimerDataset(image_tensors_tr,labels_tr)
    
    # Define parametres
    params= {
            'num_classes':4,
            'num_epochs':15,
            'batch_size': 64,
            'learning_rate':0.01
        }
    
    # Create loaders
    print("Step3: Creating loaders ")
    trainLoader, validLoader = train_data_loader(dataset_train,batch_size=params['batch_size'],shuffle=True)

    print("Step3: Creating ResNet")
    model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)


    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'], weight_decay = 0.001, momentum = 0.9)
    
    # Train the model  
    print("Start run")  
    idx = 2
    with mlflow.start_run():
        mlflow.log_params(params)
        print("Step4: Start training")
        model,optimizer = train(trainLoader, model,criterion,optimizer,params,device)
        validation(validLoader,model,device)
        save_model(model,optimizer,'Model_alz_'+str(idx))
    tracker.stop()

main()