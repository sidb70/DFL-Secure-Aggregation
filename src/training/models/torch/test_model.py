import os
import sys
import random

from loan_defaulter import LoanDefaulter
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..', 'aggregation'))
from strategies import FedAvg

class DummyLogger:
    def log(self, message):
        print(message)

# data_path =  os.path.join("/Users/sidb/Development/DFL-Secure-Aggregation/src/training/data/loan_data.csv")
data_path = os.path.join(os.getcwd(), 'src', 'training', 'data', 'loan_data.csv')
print("data_path", data_path)

n_samples1 = 100
n_samples2 = 100
n_samples3 = 100

model1 = LoanDefaulter(data_path, num_samples=n_samples1, node_hash=0, epochs=2, batch_size=20, logger=DummyLogger())
model2 = LoanDefaulter(data_path, num_samples=n_samples2, node_hash=1, epochs=2, batch_size=20, logger=DummyLogger())
model3 = LoanDefaulter(data_path, num_samples=n_samples3, node_hash=2, epochs=2, batch_size=20, logger=DummyLogger())
assert model1.X_train.shape[1] == model2.X_train.shape[1] == model3.X_train.shape[1]

logger = DummyLogger()
fedavg = FedAvg(logger)

global_model =  LoanDefaulter(data_path, num_samples=n_samples1+n_samples2, node_hash=0, epochs=2, batch_size=32, logger=DummyLogger())

for i in range(50):
    model1.train()
    model1.evaluate()
    model2.train(plot=True)
    model2.evaluate()
    model3.train()
    model3.evaluate()
    
    models = [(model1.state_dict,n_samples1), (model2.state_dict,n_samples2), (model3.state_dict,n_samples3)]

    agg_model = fedavg.aggregate(models)
    print("Aggregated model\n")
    global_model.load_state_dict(agg_model)
    model1.load_state_dict(agg_model)
    model2.load_state_dict(agg_model)
    model3.load_state_dict(agg_model)
    global_model.load_state_dict(agg_model)
    print("Global model eval:")
    global_model.evaluate()
    
