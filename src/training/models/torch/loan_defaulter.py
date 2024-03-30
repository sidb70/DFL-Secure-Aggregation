import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
import torch
import matplotlib.pyplot as plt
from .Model import BaseModel
from logging import Logger
import copy

losses = [[0,0]]
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, line2 = ax.plot(losses)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_ylim(0, 1)
ax.legend(['Train Loss', 'Validation Loss'])
        


class LoanDefaulter(BaseModel):
    def __init__(self, data_path: str, num_samples: int, node_hash: int, epochs: int, batch_size: int, logger: Logger, test_size: float = 0.2):
        super().__init__(data_path, num_samples, node_hash, epochs, batch_size, test_size)
        self.logger = logger
        self.data = self.get_loan_defaulter_data(data_path)
        if test_size==1:
            self.X_train = self.data
            self.X_valid = self.data
        self.X_train, self.X_valid, self.y_train, self.y_valid = self.train_test_split()
        self.logger.log(f"X_train shape {self.X_train.shape[1]}")
        self.model = torch.nn.Sequential(
                torch.nn.Linear(self.X_train.shape[1], 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 50),
                torch.nn.ReLU(),
                torch.nn.Linear(50, 1),
                torch.nn.Sigmoid()
        ).to(self.device)
        self.model.apply(self.init_weights)
        self.state_dict = self.model.state_dict()


    def get_loan_defaulter_data(self,data_file: str):
        all_data = pd.read_csv(data_file)
        if self.num_samples==-1:
            return all_data
        return all_data[self.node_hash*self.num_samples:(self.node_hash+1)*self.num_samples]

    def process_data(self, data):
        # # drop columns with more than 50% missing values
        # data = data.dropna(thresh=0.5*len(data), axis=1)

        # delete categorical columns
        data = data.select_dtypes(exclude=['object'])

        # convert all columns to float
        data = data.astype(float)

        data = data.fillna(0)

        return data
    def train_test_split(self):
        #self.logger.log(f"data shape {self.data.shape[1]}")

        mergeddf_sample = self.process_data(self.data)

        # self.logger.log(f"Merged shape {mergeddf_sample.shape[1]}")

        # pipeline to drop na, impute missing values, filter by VIF, and normalize
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
        ])

        # get features and labels. drop target column
        X = mergeddf_sample.drop(['TARGET'],axis=1)
        X = num_pipeline.fit_transform(X)

        self.logger.log(f"X shape {X.shape[1]}")

        y = mergeddf_sample['TARGET']

        X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, test_size=self.test_size, random_state=self.node_hash)
    
            # convert data to tensors
        X_train_tensor = torch.from_numpy(X_train).float().to(self.device)
        y_train_tensor = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float()).to(self.device)

        X_valid_tensor = torch.from_numpy(X_valid).float().to(self.device)
        y_valid_tensor = torch.squeeze(torch.from_numpy(y_valid.to_numpy()).float()).to(self.device)
        return X_train_tensor, X_valid_tensor, y_train_tensor, y_valid_tensor
    def train(self, plot=False):
        # define loss function and optimizer
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        # train 1 epoch, in batches of 10
        
        self.prev_model =copy.deepcopy(self.state_dict)
        for epoch in range(self.epochs):
            for i in range(0, len(self.X_train), self.batch_size):
                X_batch = self.X_train[i:i+self.batch_size]
                y_batch = self.y_train[i:i+self.batch_size]
                y_pred = self.model(X_batch).squeeze()
                loss = criterion(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #self.logger.log(grads['0.bias'])
            #self.logger.log('Epoch {}, Loss: {}'.format(epoch, loss.item()))

            # validation loss
            with torch.no_grad():
                y_pred = self.model(self.X_valid).squeeze()
                valid_loss = criterion(y_pred, self.y_valid)
                self.logger.log('Epoch {}, Validation Loss: {}'.format(epoch, valid_loss.item()))


            
            if plot:
                losses.append([loss.item(), valid_loss.item()])
                ax.set_xlim(0, len(losses))
                line1.set_xdata(range(len(losses)))
                line1.set_ydata([loss[0] for loss in losses])
                line2.set_xdata(range(len(losses)))
                line2.set_ydata([loss[1] for loss in losses])
                fig.canvas.draw()
                fig.canvas.flush_events()

                plt.plot(losses)
        self.state_dict = self.model.state_dict()
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.state_dict = state_dict
    def evaluate(self):
        self.model.eval()
        y_pred = self.model(self.X_valid).squeeze()
        y_pred = torch.round(y_pred)
        correct = torch.sum(y_pred == self.y_valid)
        acc = correct.item()/len(self.y_valid)
        self.logger.log('Accuracy: {}'.format(acc))
        return acc
        
