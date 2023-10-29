import json
import pickle
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Path of the root
ROOT_DIR= Path(Path(__file__).resolve().parent.parent).parent
# Path to the processed data folder
PROCESSED_DATA_DIR = ROOT_DIR / "data/prepared_data"
# Path to the metrics folder
METRICS_DIR = ROOT_DIR / "metrics"
# Path to the models folder
MODELS_FOLDER_PATH = ROOT_DIR / "models"

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


def load_test_data(input_folder_path: Path):
    """Load the test data from the prepared data folder.

    Args:
        input_folder_path (Path): Path to the test data folder.

    Returns:
        Tuple[torch.tensor, int]: Tuple containing the test images and labels.
    """
    with open(input_folder_path / "test.pkl",'rb') as test_file:
        X_test,y_test = pickle.load(test_file)
        #y_test = torch.tensor(y_test)
    return X_test, y_test


def evaluate_model(model_file_name, loader):
    """Evaluate the model using the test data.

    Args:
        model_file_name (str): Filename of the model to be evaluated.
        x (torch.tensor): Test images.
        y (int list): Validation target.

    Returns:
        acc Accuracy of the model on teh test set
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)
    checkpoint = torch.load(MODELS_FOLDER_PATH / model_file_name)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()


    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

    test_acc = 100 * correct / total
    print('Accuracy of the network on the {} test images: {} %'.format(5000, test_acc))

    return test_acc


if __name__ == "__main__":
    # Path to the metrics folder
    metrics_folder_path = METRICS_DIR

    X_test, y_test = load_test_data(PROCESSED_DATA_DIR / "test")

    dataset_test = AlzheimerDataset(X_test,y_test)
    testLoader = DataLoader(dataset_test, batch_size=64, shuffle=True)


    # Load the model
    test_acc = evaluate_model(
        "alzheimer_model.pth", testLoader
    )

    # Save the evaluation metrics to a dictionary to be reused later
    metrics_dict = {"test_accuracy": test_acc}

    # Save the evaluation metrics to a JSON file
    with open(metrics_folder_path / "scores.json", "w") as scores_file:
        json.dump(
            metrics_dict,
            scores_file,
            indent=4,
        )

    print("Evaluation completed.")