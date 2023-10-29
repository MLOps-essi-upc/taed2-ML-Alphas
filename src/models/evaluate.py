import json
import pickle
from pathlib import Path
import torch

import mlflow

from src import METRICS_DIR, PROCESSED_DATA_DIR

# Path to the models folder
MODELS_FOLDER_PATH = Path("models")


def load_test_data(input_folder_path: Path):
    """Load the test data from the prepared data folder.

    Args:
        input_folder_path (Path): Path to the test data folder.

    Returns:
        Tuple[torch.tensor, int]: Tuple containing the test images and labels.
    """
    with open(input_folder_path / "test.pkl",'rb') as test_file:
        X_test,y_test = pickle.load(test_file)
    return X_test, y_test


def evaluate_model(model_file_name, x, y):
    """Evaluate the model using the test data.

    Args:
        model_file_name (str): Filename of the model to be evaluated.
        x (torch.tensor): Test images.
        y (int list): Validation target.

    Returns:
        acc Accuracy of the model on teh test set
    """

    with open(MODELS_FOLDER_PATH / model_file_name, "rb") as pickled_model:
        model = pickle.load(pickled_model)

        # Compute predictions using the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in x,y:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs

        mlflow.log_metric('test_acc',100*correct/total)
        print('Accuracy of the network on the {} test images: {} %'.format(5000, 100 * correct / total))
    
    return test_acc


if __name__ == "__main__":
    # Path to the metrics folder
    Path("metrics").mkdir(exist_ok=True)
    metrics_folder_path = METRICS_DIR

    X_test, y_test = load_test_data(PROCESSED_DATA_DIR)

    mlflow.set_experiment("test_model")

    with mlflow.start_run():
        # Load the model
        test_acc = evaluate_model(
            "iowa_model.pkl", X_test, y_test
        )

        # Save the evaluation metrics to a dictionary to be reused later
        metrics_dict = {"test_accuracy": test_acc}

        # Log the evaluation metrics to MLflow
        mlflow.log_metrics(metrics_dict)

        # Save the evaluation metrics to a JSON file
        with open(metrics_folder_path / "scores.json", "w") as scores_file:
            json.dump(
                metrics_dict,
                scores_file,
                indent=4,
            )

        print("Evaluation completed.")