from datasets import load_dataset

# Load the 'train' and 'test' split of the Falah/Alzheimer_MRI dataset
train_dataset = load_dataset('Falah/Alzheimer_MRI', split='train')
test_dataset = load_dataset('Falah/Alzheimer_MRI', split='test')

raw_train_path = 'data/raw_data/train'
raw_test_path = 'data/raw_data/test'

train_dataset.save_to_disk(raw_train_path)
test_dataset.save_to_disk(raw_test_path)
