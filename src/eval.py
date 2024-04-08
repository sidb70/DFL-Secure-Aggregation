import torch
import os
from training.models.torch.loan_defaulter import LoanDefaulter
from training.models.torch.digit_classifier import DigitClassifier
from aggregation.strategies import FedAvg
import yaml
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

experiment_yaml = os.path.join('src','config', 'experiment.yaml')
with open(experiment_yaml) as f:
    experiment_params = yaml.safe_load(f)
experiment_id = experiment_params['id']
experiment_desc = experiment_params['description']


def save_results(exp_id):
    save_dir = os.path.join('src','training','results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try:
        with open(os.path.join(save_dir,f'{exp_id}.json'),'r') as f:
            results = json.load(f)
        print(f'Loaded results for experiment {exp_id}\n{experiment_desc}')
    except FileNotFoundError:
        print(f'Creating new results file for experiment {exp_id}\n{experiment_desc}')
        # create new results file
        results = {}
        results['id'] = exp_id
        results['description'] = experiment_desc
        results['experiments'] = []

    results['experiments'].append({'params': experiment_params,
                                    'accuracies_by_round': [],
                                    'loss_by_round': []
                                    })
    
    # load all nodes metrics
    node_metrics_dir = os.path.join('src','training','results','node_metrics')
    avg_accuracies_by_round = [0]*experiment_params['rounds']
    avg_losses_by_round = [0]*experiment_params['rounds']
    num_benign_nodes = 0
    for node_hash in os.listdir(node_metrics_dir):
        with open(os.path.join(node_metrics_dir,node_hash),'r') as f:
            node_metrics = json.load(f)
        node_accuracies = node_metrics['accuracies']
        node_losses = node_metrics['losses']
        for r in range(len(node_accuracies)):
            avg_accuracies_by_round[r] += node_accuracies[r]
            avg_losses_by_round[r] += node_losses[r]
        num_benign_nodes += 1

        
        
    results['experiments'][-1]['accuracies_by_round'] = [a/num_benign_nodes for a in avg_accuracies_by_round]
    results['experiments'][-1]['loss_by_round'] = [l/num_benign_nodes for l in avg_losses_by_round]
    with open(os.path.join(save_dir,f'{exp_id}.json'),'w') as f:
        json.dump(results, f)
    print("Saved results to", os.path.join(save_dir,f'{exp_id}.json'))



def save_node_metrics(node_hash, accuracy, loss):
    '''
    Save node metrics to a json file.
    '''
    save_dir = os.path.join('src','training','results','node_metrics')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try:
        with open(os.path.join(save_dir,f'{node_hash}.json'),'r') as f:
            results = json.load(f)
        print(f'Loaded results for node {node_hash}')
    except FileNotFoundError:
        print(f'Creating new results file for node {node_hash}')
        # create new results file
        results = {}
        results['node_hash'] = node_hash
        results['accuracies'] = []
        results['losses'] = []

    results['accuracies'].append(accuracy)
    results['losses'].append(loss)
    with open(os.path.join(save_dir,f'{node_hash}.json'),'w') as f:
        json.dump(results, f)
    print("Saved results to", os.path.join(save_dir,f'{node_hash}.json'))

def make_plot(exp_id):
    experiment_json_path = os.path.join('src','training','results',f'{exp_id}.json') 
    with open(experiment_json_path,'r') as f:
        results = json.load(f)

    byzantine_proportion_legend = []
    print(len(results['experiments']))
    for i in range(len(results['experiments'])):
        accuracies_by_round = results['experiments'][i]['accuracies_by_round']
        experiment_params = results['experiments'][i]['params']
        byzantine_proportion = experiment_params['malicious_proportion']
        byzantine_proportion_legend.append(str(byzantine_proportion*100) + '% Byzantine')

        plt.plot(range(1,len(accuracies_by_round)+1), accuracies_by_round, label=f'Byzantine Proportion: {byzantine_proportion}')
    
    plt.legend(byzantine_proportion_legend)
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Round')
    plt.savefig(os.path.join('src','training','results',f'{exp_id}_accuracy_by_round.png'))
    plt.clf()
    # loss
    for i in range(len(results['experiments'])):
        losses_by_round = results['experiments'][i]['loss_by_round']
        experiment_params = results['experiments'][i]['params']
        byzantine_proportion = experiment_params['malicious_proportion']
        plt.plot(range(1,len(losses_by_round)+1), losses_by_round, label=f'Byzantine Proportion: {byzantine_proportion}')

    plt.legend(byzantine_proportion_legend)
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Loss by Round')
    plt.savefig(os.path.join('src','training','results',f'{exp_id}_loss_by_round.png'))
    plt.clf()


# def eval_global_model(dataset):
#     """
#     Evaluate the global model.

#     If id.json exists: create
#     Open id.json
#     {
#     id: id
#     description: description
#     results: [{'params': {exp_params}, 
#             'accuracies_by_round': [r0_avg_acc, 1: r1_avg_acc, ...],
#             'loss_by_round': [r0_avg_loss, r1_avg_loss, ...]
#                 },
#                 .
#                 .
#                 .
#             ]
#     }

#     """
#     experiment_yaml = os.path.join('src','config', 'experiment.yaml')
#     with open(experiment_yaml) as f:
#         experiment_params = yaml.safe_load(f)
#     experiment_id = experiment_params['id']
#     experiment_desc = experiment_params['description']

#     save_dir = os.path.join('src','training','results')
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     try:
#         with open(os.path.join(save_dir,f'{experiment_id}.json'),'r') as f:
#             results = json.load(f)
#         print(f'Loaded results for experiment {experiment_id}\n{experiment_desc}')
#     except FileNotFoundError:
#         print(f'Creating new results file for experiment {experiment_id}\n{experiment_desc}')
#         # create new results file
#         results = {}
#         results['id'] = experiment_id
#         results['description'] = experiment_desc
#         results['experiments'] = []

#     topology_json_file = os.path.join('src', 'config', 'topology.json')
#     with open(topology_json_file) as f:
#         topology_json = json.load(f)
#     topology = {int(node_id):val for node_id,val in topology_json.items()}
#     num_nodes = experiment_params['nodes']

    
#     if experiment_params['model_name'] == 'loan_defaulter':
#         eval_model = LoanDefaulter(experiment_params['data_path'], \
#                         num_samples=-1, node_hash=0, epochs=1, batch_size=10, evaluating=True, 
#                         logger=DummyLogger())
#     elif experiment_params['model_name'] == 'digit_classifier':
#         eval_model = DigitClassifier(epochs=20, batch_size=experiment_params['batch_size'], 
#                                  num_samples=500, 
#                                  node_hash=42,
#                                  evaluating=True)
    
#     models_pt_dir=os.path.join('src','training','models','nodes')
#     print("Evaluating global model")
#     accuracies_by_round = []
#     losses_by_round = []
#     for r in range(1,experiment_params['rounds']):
#         print("\tRound ", r,"evaluation")
#         accuracies = []
#         losses = []
#         for i in range(num_nodes):
#             if topology[i]['malicious']:
#                 print("Skipping malicious node", i)
#                 continue
#             print("Evaluating node", i)
#             node_model = torch.load(os.path.join(models_pt_dir,f'round_{r}',f'node_{i}.pt')) 
#             eval_model.load_state_dict(node_model)
#             accuracy, loss = eval_model.evaluate(dataset)
#             accuracies.append(accuracy)
#             losses.append(loss)

#         accuracies_by_round.append(sum(accuracies)/len(accuracies))
#         losses_by_round.append(sum(losses)/len(losses))

#     print("Global model evaluation complete.")

#     # save to results
#     results['experiments'].append({'params': experiment_params, 
#                                     'accuracies_by_round': accuracies_by_round,
#                                     'loss_by_round': losses_by_round
#                                     })
#     with open(os.path.join(save_dir,f'{experiment_id}.json'),'w') as f:
#         json.dump(results, f)
#     print("Saved results to", os.path.join(save_dir,f'{experiment_id}.json'))
    # plot accuracy by round
    # plt.plot(list(accuracies_by_round.keys()), list(accuracies_by_round.values()))
    # ax= plt.gca()
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # plt.xlabel('Round')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy by Round')
    # plt.show()

    # plt.savefig(os.path.join('src','training','results','accuracy_by_round.png'))
    

    #plt.show()
if __name__=='__main__':
    save_results(999)
    make_plot(1)