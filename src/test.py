import torch
from torch.utils.data import DataLoader, Dataset
from multiprocessing import Process
import random
from torchvision import datasets, transforms
from torch.utils.data import Subset

class CustomSubsetDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def worker(dataset, batch_size, worker_num):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Worker", worker_num, " has ", len(loader), " batches")
    for i, batch in enumerate(loader):
        continue

if __name__ == '__main__':
    # Example data
    data = [torch.randn(3, 32, 32) for _ in range(500)]
    mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root='data', train=True, transform=mnist_transform, download=False)
    # Number of worker processes
    num_workers = 7

    # Batch size
    batch_size = 32

    # Create and start processes
    processes = []
    data_len = len(mnist_dataset)
    samples_per_worker =500
    

    for worker_num in range(num_workers):
        start_index = (worker_num*samples_per_worker)% data_len
        end_index = start_index + samples_per_worker
        subset_dataset = Subset(mnist_dataset, list(range(start_index, end_index)))
        p = Process(target=worker, args=(subset_dataset, batch_size, worker_num))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()
