import torch
from torch.utils.data import Dataset
from util import MyCustomDataLoader

# Define a custom dataset
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# Create a sample dataset
data = [i for i in range(100)]
dataset = MyDataset(data)

# Create an instance of MyCustomDataLoader
batch_size = 10
replace_size = 5
num_workers = 2

dataloader = MyCustomDataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    prefetch_factor=2,
    replace_size=replace_size
)

# Iterate over the data loader
for epoch in range(5):
    print(f"Epoch {epoch+1}")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1}: {batch}")
    print()