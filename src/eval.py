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

    If id.json exists: create
    Open id.json
    {
    id: id
    description: description
    results: [{'params': {exp_params}, 
            'accuracies_by_round': [r0_avg_acc, 1: r1_avg_acc, ...],
            'loss_by_round': [r0_avg_loss, r1_avg_loss, ...]
                },
                .
                .
                .
            ]
    }

    """
    experiment_yaml = os.path.join('src','config', 'experiment.yaml')
    with open(experiment_yaml) as f:
        experiment_params = yaml.safe_load(f)
    experiment_id = experiment_params['id']
    experiment_desc = experiment_params['description']

    save_dir = os.path.join('src','training','results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try:
        with open(os.path.join(save_dir,f'{experiment_id}.json'),'r') as f:
            results = json.load(f)
        print(f'Loaded results for experiment {experiment_id}\n{experiment_desc}')
    except FileNotFoundError:
        print(f'Creating new results file for experiment {experiment_id}\n{experiment_desc}')
        # create new results file
        results = {}
        results['id'] = experiment_id
        results['description'] = experiment_desc
        results['experiments'] = []

    topology_json_file = os.path.join('src', 'config', 'topology.json')
    with open(topology_json_file) as f:
        topology_json = json.load(f)
    topology = {int(node_id):val for node_id,val in topology_json.items()}
    num_clients = experiment_params['nodes']

    
    if experiment_params['model_name'] == 'loan_defaulter':
        eval_model = LoanDefaulter(experiment_params['data_path'], \
                        num_samples=-1, node_hash=0, epochs=1, batch_size=10, evaluating=True, 
                        logger=DummyLogger())
    elif experiment_params['model_name'] == 'digit_classifier':
        eval_model = DigitClassifier(epochs=20, batch_size=128, num_samples=10, 
                                 node_hash=42,logger=DummyLogger(), 
                                 evaluating=True)
    
    models_pt_dir=os.path.join('src','training','models','clients')
    print("Evaluating global model")
    accuracies_by_round = []
    losses_by_round = []
    for r in range(experiment_params['rounds']):
        print("\nEvaluating round", r)
        accuracies = []
        losses = []
        for i in range(num_clients):
            print("Evaluating client", i)
            if topology[i]['malicious']:
                continue
            client_model = torch.load(os.path.join(models_pt_dir,f'round{r}',f'client_{i}.pt')) 
            eval_model.load_state_dict(client_model)
            accuracy, loss = eval_model.evaluate()
            accuracies.append(accuracy)
            losses.append(loss)

        accuracies_by_round.append(sum(accuracies)/len(accuracies))
        losses_by_round.append(sum(losses)/len(losses))

    print("Global model evaluation complete.")

    # save to results
    results['experiments'].append({'params': experiment_params, 
                                    'accuracies_by_round': accuracies_by_round,
                                    'loss_by_round': losses_by_round
                                    })
    with open(os.path.join(save_dir,f'{experiment_id}.json'),'w') as f:
        json.dump(results, f)
    print("Saved results to", os.path.join(save_dir,f'{experiment_id}.json'))
    # plot accuracy by round
    # plt.plot(list(accuracies_by_round.keys()), list(accuracies_by_round.values()))
    # ax= plt.gca()
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # plt.xlabel('Round')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy by Round')
    # plt.show()

    # plt.savefig(os.path.join('src','training','results','accuracy_by_round.png'))
    
    
if __name__=='__main__':
    eval_global_model()