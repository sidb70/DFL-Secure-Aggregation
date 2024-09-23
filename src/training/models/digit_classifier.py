
from .model import BaseModel
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def load_mnist():
    mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root='data', train=True, transform=mnist_transform, download=True)
    return mnist_dataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
class DigitClassifier(BaseModel):
    def __init__(self, epochs: int, batch_size: int, num_samples: int, 
                 node_hash: int, evaluating=False, device=None):
        super().__init__(num_samples, node_hash, epochs, batch_size, evaluating=evaluating, device=device)
        self.model = Net().to(self.device)
        print("Sent {} node's model to device: {}".format(self.node_hash, self.device))

    def train(self, subset_dataset):
        X_train = DataLoader(subset_dataset, batch_size=self.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        
        for epoch in range(self.epochs):
            losses = []
            for batch_idx, (inputs, labels) in enumerate(X_train):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()   
            loss = sum(losses) / len(losses)
            #print(f'Node {self.node_hash} Epoch {epoch + 1}/{self.epochs} Loss: {loss}')


    def evaluate(self, dataset):
        X_valid = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        correct = 0
        total = 0
        losses = []
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for data in X_valid:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # loss
                loss = criterion(outputs, labels)
                losses.append(loss.item())
        loss = sum(losses) / len(losses)
        accuracy = correct / total
        
        return accuracy, loss


if __name__ == '__main__':
    class DummyLogger:
        def log(self, msg):
            print(msg)
            return
    classifier = DigitClassifier(epochs=20, batch_size=128, num_samples=10, 
                                 node_hash=42,logger=DummyLogger(), 
                                 evaluating=False)
    # print num samples
    # print(len(classifier.X_train))
    # classifier.train()
    # plt.plot(classifier.losses)
    # plt.show()
    # save_dir = os.path.join('src','training','results')
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # dt = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # plt.savefig(os.path.join('src','training','results',f'MNIST_accuracy_{dt}.png'))
