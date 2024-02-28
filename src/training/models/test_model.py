import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
base_dir = os.path.dirname(parent_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)
sys.path.append(base_dir)

from loan_defaulter import LoanDefaulterModel
from aggregation.fedavg import FedAvg

class DummyLogger:
    def log(self, message):
        print(message)
data_path =  os.path.join(parent_dir, 'data/loan_data.csv')

n_samples1 = 401
n_samples2 = 203

model1 = LoanDefaulterModel(data_path, num_samples=n_samples1, node_hash=0, epochs=2)
model2 = LoanDefaulterModel(data_path, num_samples=n_samples2, node_hash=1, epochs=2)

logger = DummyLogger()
fedavg = FedAvg(logger)

global_model =  LoanDefaulterModel(data_path, num_samples=n_samples1+n_samples2, node_hash=0, epochs=2)

for i in range(10):
    model1.train()
    model1.evaluate()
    model2.train(plot=True)
    model2.evaluate()
    
    models = [(model1.state_dict,n_samples1), (model2.state_dict,(n_samples2))]

    agg_model = fedavg.aggregate(models)
    print("Aggregated model\n")

    model1.load_state_dict(agg_model)
    model2.load_state_dict(agg_model)
    global_model.load_state_dict(agg_model)
    print("Global model eval:")
    global_model.evaluate()
    
