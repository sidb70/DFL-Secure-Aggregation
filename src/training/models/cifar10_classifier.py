import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .model import BaseModel


'''
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
'''
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CIFAR10Classifier(BaseModel):
    def __init__(self, epochs: int, batch_size: int, num_samples: int, 
                 node_hash: int, evaluating=False, device=None):
        super().__init__(num_samples, node_hash, epochs, batch_size, evaluating=evaluating)
        self.model = Net().to(self.device)

    def train(self, subset_dataset):
        X_train = DataLoader(subset_dataset, batch_size=self.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.model.train()
        for epoch in range(self.epochs):
            for i, data in enumerate(X_train, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def evaluate(self, dataset):
        X_valid = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        correct = 0
        total = 0
        loss = 0
        criterion = nn.CrossEntropyLoss()
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in  X_valid:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                # calculate outputs by running images through the network
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        loss = loss / len(X_valid)
        return accuracy, loss
    

if __name__ == '__main__':
    import os
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split

    # CIFAR10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='data', train=False, transform=transform, download=True)

    # split the training dataset into training and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # create the model
    model = CIFAR10Classifier(epochs=50, batch_size=4, num_samples=1, node_hash=0, evaluating=False)
    model.train(train_dataset)
    accuracy, loss = model.evaluate(val_dataset)
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Loss: {loss:.4f}')  
