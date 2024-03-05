import torch
import os
from training.models.torch.loan_defaulter import LoanDefaulter
from aggregation.strategies import FedAvg
import yaml
class DummyLogger:
    def log(self, msg):
        print(msg)

experiment_yaml = os.path.join('src','config', 'experiment.yaml')
with open(experiment_yaml) as f:
    experiment_params = yaml.safe_load(f)
def eval_global_model():
    """
    Evaluate the global model.
    """
    models = [os.path.join('src','training','models','clients',f'client_{i}.pt') for i in range(5)]
    models = [(torch.load(model), experiment_params['num_samples']) for model in models]
    aggregated_model = FedAvg(DummyLogger()).aggregate(models)
    cwd = os.getcwd()
    data_path = os.path.join(cwd, 'src', 'training', 'data', 'loan_data.csv')
    loan_model = LoanDefaulter(data_path,
                               num_samples=10000, node_hash=0, epochs=10, batch_size=100, logger=DummyLogger())
    loan_model.model.load_state_dict(aggregated_model)
    loan_model.evaluate()
    