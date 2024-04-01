import torch
import os
from training.models.torch.loan_defaulter import LoanDefaulter
from training.models.torch.MNIST_model import DigitClassifier
from aggregation.strategies import FedAvg
import yaml
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

    accuracies_by_round = {round_num: [] for round_num in range(experiment_params['rounds'])}
    if experiment_params['model_name'] == 'loan_defaulter':
        eval_model = LoanDefaulter(experiment_params['data_path'], \
                        num_samples=-1, node_hash=0, epochs=1, batch_size=10, evaluating=True, 
                        logger=DummyLogger())
    elif experiment_params['model_name'] == 'digit_classifier':
        eval_model = DigitClassifier(epochs=20, batch_size=128, num_samples=10, 
                                 node_hash=42,logger=DummyLogger(), 
                                 evaluating=True)
    base_dir=os.path.join('src','training','models','clients')
    for r in range(experiment_params['rounds']):
        accuracies = []
        for i in range(num_clients):
            if topology[i]['malicious']:
                continue
            client_model = torch.load(os.path.join(base_dir,f'round{r}',f'client_{i}.pt')) 
            eval_model.load_state_dict(client_model)
            accuracies.append(eval_model.evaluate())
        accuracies_by_round[r] = sum(accuracies)/len(accuracies)
    # plot accuracy by round
    plt.plot(list(accuracies_by_round.keys()), list(accuracies_by_round.values()))
    ax= plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Round')
    plt.show()
    save_dir = os.path.join('src','training','results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join('src','training','results','accuracy_by_round.png'))
    
    
if __name__=='__main__':
    eval_global_model()