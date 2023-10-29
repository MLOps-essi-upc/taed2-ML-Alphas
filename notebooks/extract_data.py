from datasets import load_dataset

# Load the 'train' and 'test' split of the Falah/Alzheimer_MRI dataset
train_dataset = load_dataset('Falah/Alzheimer_MRI', split='train')
test_dataset = load_dataset('Falah/Alzheimer_MRI', split='test')

raw_train_path = 'data/raw_data/train'
raw_test_path = 'data/raw_data/test'

#train_dataset.save_to_disk(raw_train_path)
#test_dataset.save_to_disk(raw_test_path)
train_dataset.to_parquet(raw_train_path+'/train.parquet')
test_dataset.to_parquet(raw_test_path+ '/test.parquet')
