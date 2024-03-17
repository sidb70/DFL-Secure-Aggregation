import torch
import os
from training.models.torch.loan_defaulter import (
    LoanDefaulter,
)
from aggregation.strategies import FedAvg
import yaml
import json
class DummyLogger:
    def log(self, msg):
        #print(msg)
        return


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

    accuracies = []
    if experiment_params['model_name'] == 'loan_defaulter':
        eval_model = LoanDefaulter(experiment_params['data_path'], \
                        num_samples=100, node_hash=0, epochs=1, batch_size=10, logger=DummyLogger())
    base_dir=os.path.join('src','training','models','clients')
    for i in range(num_clients):
        if topology[i]['malicious']:
            continue
        benign_model = torch.load(os.path.join(base_dir,f'client_{i}.pt')) 
        eval_model.load_state_dict(benign_model)
        accuracies.append(eval_model.evaluate())
    
    print(f"Average global acc. of {len(accuracies)} benign clients:\n\t", 
          sum(accuracies)/len(accuracies))
if __name__=='__main__':
    eval_global_model()