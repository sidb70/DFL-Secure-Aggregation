
from .Model import BaseModel
import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras import layers, Sequential
import logging
import numpy as np
from logging import Logger
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import datetime
import os

# class DigitClassifierTensorFlow(BaseModel):
#     def __init__(self, epochs: int, batch_size: int, num_samples: int, node_hash: int, logger: logging.Logger):
#         super().__init__(num_samples, node_hash, epochs, batch_size)
#         self.logger = logger
#         self.losses = {'train': [], 'validation': []}
#         self.load_data()
#         self.build_model()

#     def load_data(self):
#         (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#         np.random.seed(self.node_hash)
#         permutation = np.random.permutation(x_train.shape[0])
#         x_train, y_train = x_train[permutation], y_train[permutation]

#         if self.num_samples > 0:
#             x_train, y_train = x_train[:self.num_samples], y_train[:self.num_samples]

#         x_train, x_test = x_train / 255.0, x_test / 255.0
#         x_train = x_train[..., tf.newaxis]
#         x_test = x_test[..., tf.newaxis]

#         self.train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000,
#                                                                                        seed=self.node_hash).batch(
#             self.batch_size)
#         self.val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.batch_size)

#         AUTOTUNE = tf.data.AUTOTUNE
#         self.train_ds = self.train_ds.cache().prefetch(buffer_size=AUTOTUNE)
#         self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#     def build_model(self):
#         self.model = Sequential([
#             layers.Rescaling(1. / 255, input_shape=(28, 28, 1)),
#             layers.Conv2D(16, 3, padding='same', activation='relu'),
#             layers.MaxPooling2D(),
#             layers.Conv2D(32, 3, padding='same', activation='relu'),
#             layers.MaxPooling2D(),
#             layers.Conv2D(64, 3, padding='same', activation='relu'),
#             layers.MaxPooling2D(),
#             layers.Flatten(),
#             layers.Dense(128, activation='relu'),
#             layers.Dense(10)
#         ])

#         self.model.compile(optimizer='adam',
#                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                            metrics=['accuracy'])

#     def train(self):
#         history = self.model.fit(
#             self.train_ds,
#             validation_data=self.val_ds,
#             epochs=self.epochs
#         )

#         self.losses['train'] = history.history['loss']
#         self.losses['validation'] = history.history['val_loss']

#         self.logger.info('Training complete')

#     def plot_losses(self):
#         epochs_range = range(self.epochs)

#         plt.figure(figsize=(8, 8))
#         plt.subplot(1, 2, 1)
#         plt.plot(epochs_range, self.losses['train'], label='Training Loss')
#         plt.plot(epochs_range, self.losses['validation'], label='Validation Loss')
#         plt.legend(loc='upper right')
#         plt.title('Training and Validation Loss')
#         plt.show()

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
    def __init__(self, epochs: int, batch_size: int, num_samples: int, node_hash: int,  logger: Logger, evaluating=False):
        super().__init__(num_samples, node_hash, epochs, batch_size, evaluating=evaluating)
        self.logger = logger
        self.losses = []
        self.load_data()
        self.logger.log("Loaded dataset")
        self.device = 'cuda:' + str((self.node_hash %8)) if torch.cuda.is_available() else 'cpu'
        self.model = Net().to(self.device)
        self.logger.log("Sent model to device")
        self.state_dict = self.model.state_dict()

    def load_data(self):
        dataset = datasets.MNIST('data', train=True, download=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ]))
        if self.evaluating:
            # use whole dataset for valid set
            self.X_valid = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            return
        
        #training set should have num_samples samples
        train_size = int(self.num_samples)
        test_size = len(dataset) - train_size
        # set node hash for reproducibility
        torch.manual_seed(self.node_hash)
        self.X_train, self.X_valid = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        self.X_train = DataLoader(self.X_train, batch_size=self.batch_size, shuffle=True)
        
        # delete validation set (FL evaluation will be done separately after training)
        del self.X_valid
        del dataset

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        for epoch in range(self.epochs):
            for batch_idx, (inputs, labels) in enumerate(self.X_train):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                self.losses.append(loss.item())
                self.logger.log(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')
        self.state_dict = self.model.state_dict()
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        losses = []
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for data in self.X_valid:
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
        self.logger.log(f'Accuracy: {accuracy}')
        self.logger.log(f'Loss: {loss}')
        
        return accuracy, loss
    def plot_losses(self):
        save_dir = os.path.join('src','training','results')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        dt = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        plt.plot(self.losses)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(os.path.join('src','training','results',f'MNIST_loss_{self.node_hash}.png'))


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
