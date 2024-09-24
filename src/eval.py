import os
import yaml
import json
import matplotlib.pyplot as plt

def save_results(experiment_params):
    exp_id = experiment_params['id']
    iteration = experiment_params['iteration']
    
    experiment_desc = experiment_params['description']
    save_dir = os.path.join('src','training','results', f'experiment_{exp_id}')
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
    node_metrics_dir = os.path.join('src','training','results',f'experiment_{exp_id}',f'{iteration}','node_metrics')
    avg_accuracies_by_round = [0]*experiment_params['rounds']
    avg_losses_by_round = [0]*experiment_params['rounds']
    num_benign_nodes = 0
    for node_hash_json in os.listdir(node_metrics_dir):
        with open(os.path.join(node_metrics_dir,node_hash_json),'r') as f:
            node_metrics = json.load(f)
        node_accuracies = node_metrics['accuracies']
        node_losses = node_metrics['losses']
        for r in range(30):
            avg_accuracies_by_round[r] += node_accuracies[r]
            avg_losses_by_round[r] += node_losses[r]
        num_benign_nodes += 1
    results['experiments'][-1]['accuracies_by_round'] = [a/num_benign_nodes for a in avg_accuracies_by_round]
    results['experiments'][-1]['loss_by_round'] = [l/num_benign_nodes for l in avg_losses_by_round]
    with open(os.path.join(save_dir,f'{exp_id}.json'),'w') as f:
        json.dump(results, f)
    print("Saved results to", os.path.join(save_dir,f'{exp_id}.json'))



def save_node_metrics(node_hash, accuracy, loss, exp_id, iteration):
    '''
    Save node metrics to a json file.
    '''
    save_dir = os.path.join('src','training','results',f'experiment_{exp_id}',f'{iteration}','node_metrics')
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
    experiment_json_path = os.path.join('src','training','results',f'experiment_{exp_id}', f'{exp_id}.json')
    with open(experiment_json_path,'r') as f:
        results = json.load(f)

    print(len(results['experiments']))
    line_type = { 'scale-free': '-', 'small-world': '--', 'two-f-1': '-.'}
    line_color = {0.0: 'green', 0.3: 'blue', 0.6: 'red', .15: 'pink', .45: 'orange', 0.1: 'pink', 0.05: 'purple', 0.15: 'brown', 0.25: 'black'}
    for i in range(len(results['experiments'])):
        accuracies_by_round = results['experiments'][i]['accuracies_by_round']
        experiment_params = results['experiments'][i]['params']
        byzantine_proportion = results['experiments'][i]['params']['malicious_proportion']
        topology = results['experiments'][i]['params']['topology']
        plt.plot(range(1,len(accuracies_by_round)+1), accuracies_by_round, label=f'{topology} b={byzantine_proportion}', linestyle=line_type[topology], color=line_color[byzantine_proportion])

    plt.legend()

    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Trimmed Mean Accuracy by Round \nStrategic Byzantine Placement')
    plt.savefig(os.path.join('src','training','results',f'{exp_id}_accuracy_by_round.png'))

    plt.clf()

if __name__=='__main__':
    # with open(os.path.join('src','config','experiment.yaml')) as f:
    #     experiment_params = yaml.safe_load(f)
    # save_results(experiment_params)
    make_plot(18)

