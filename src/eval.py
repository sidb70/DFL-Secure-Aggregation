import torch
import os
from training.models.torch.loan_defaulter import (
    LoanDefaulter,
)
from aggregation import strategies
import yaml
import json
class DummyLogger:
    def log(self, msg):
        print(msg)


def eval_global_model():
    """
    Evaluate the global model.
    """
    experiment_yaml = os.path.join('src','config', 'experiment.yaml')
    with open(experiment_yaml) as f:
        experiment_params = yaml.safe_load(f)

    topology_json_file = os.path.join('src', 'config', 'topology.json')
    with open(topology_json_file) as f:
        topology_json = json.load(f)
    topology = {int(node_id):val for node_id,val in topology_json.items()}
    num_clients = experiment_params['nodes']
    models =[]
    for i in range(num_clients):
        if not topology[i]['malicious']:
            models.append((torch.load(os.path.join('src','training','models','clients',f'client_{i}.pt')), 
                           experiment_params['num_samples']))
    if len(models) == 0:
        print("No non-malicious clients to evaluate")
        return
    aggregation_strategy = experiment_params['aggregation']
    aggregated_model = strategies.create_aggregator(aggregation_strategy,logger=DummyLogger()).aggregate(models)
    if experiment_params['model_name'] == 'loan_defaulter':
        model = LoanDefaulter("/Users/sidb/Development/DFL-Secure-Aggregation/src/training/data/loan_data.csv", \
                                num_samples=10000, node_hash=0, epochs=10, batch_size=100, logger=DummyLogger())
    model.model.load_state_dict(aggregated_model)
    model.evaluate()
    