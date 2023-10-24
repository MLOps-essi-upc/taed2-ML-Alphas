from PIL import Image
import io
import numpy as np
import pandas as pd
from torchvision import transforms

def prepare_data(dataset_path, data_augmentation):
    dataset = pd.read_parquet(dataset_path)

    num_instances = len(dataset)
    if data_augmentation:
        num_instances *= 2
    dataset2 = {'Image': [], 'Label': []}

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to (224, 224)
        transforms.ToTensor(),          # Convert PIL Image to tensor
    ])

    if data_augmentation:
        for i in range(0, len(dataset)):
            image_bytes = dataset.iloc[i]['image']['bytes']
            image_pil = Image.open(io.BytesIO(image_bytes))
            aug_image = tf.image.stateless_random_brightness(image_pil, max_delta=np.random.rand(), seed=(3, 0))
            pil_format = Image.fromarray(aug_image.numpy())
            image_tensor = transform(image_pil).numpy().flatten()  # Convert and flatten the PyTorch tensor
            dataset2['Image'].append(image_tensor)
            dataset2['Label'].append(dataset.iloc[i]['label'])
            image_tensor = transform(pil_format).numpy().flatten()  # Convert and flatten the PyTorch tensor
            dataset2['Image'].append(image_tensor)
            dataset2['Label'].append(dataset.iloc[i]['label'])
    else:
        for i in range(0, len(dataset)):
            image_bytes = dataset.iloc[i]['image']['bytes']
            image_pil = Image.open(io.BytesIO(image_bytes))
            image_tensor = transform(image_pil).numpy().flatten()  # Convert and flatten the PyTorch tensor
            dataset2['Image'].append(image_tensor)
            dataset2['Label'].append(dataset.iloc[i]['label'])

    df = pd.DataFrame(dataset2)
    return df

def main(repo_path):
    data_augmentation = False
    data_path = repo_path + "data"
    train_path = data_path + "/raw/train.parquet"
    test_path = data_path + "/raw/test.parquet"

    train_df = prepare_data(train_path, data_augmentation)
    test_df = prepare_data(test_path, data_augmentation)

    # save prepared parquet datasets
    prepared_path = data_path + "/prepared"
    train_df.to_parquet(prepared_path + '/train.parquet', index=False)
    test_df.to_parquet(prepared_path + '/test.parquet', index=False)

main("../")
