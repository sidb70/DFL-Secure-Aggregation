'''
This file is the main entry point for the simulator. It sets up the server and runs the nodes.
'''
from network import graph
import multiprocessing
from multiprocessing import Manager
import os
import random
import yaml
import eval
import torch
from torch.utils.data import Subset
import signal
from training.models.model_loader import load_model
from training.dataloader import load_data
from aggregation import strategies
import time
from attack import attacks
# seed
random.seed(42)



experiment_yaml = os.path.join('src','config', 'experiment.yaml')
with open(experiment_yaml) as f:
    experiment_params = yaml.safe_load(f)


class DFLTrainer:
    def __init__(self, num_nodes, topology, num_workers, num_rounds, epochs_per_round, batch_size,
                 num_samples, aggregation_method, attack_method, exp_id=experiment_params['id'], exp_iteration=experiment_params['iteration'],
                 dataset=experiment_params['dataset']):
        self.num_nodes = num_nodes
        self.topology = topology
        self.num_workers = num_workers
        self.num_rounds = num_rounds
        self.dataset = dataset
        self.epochs_per_round = epochs_per_round
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.aggregation_method = aggregation_method
        self.attack_method = attack_method

        self.nodes = list(range(self.num_nodes))
        self.malicious_nodes = set(node_hash for node_hash in self.nodes if \
                                 self.topology[node_hash]['malicious'])
        
        self.exp_id = exp_id
        self.exp_iteration = exp_iteration
        
        
        self.dataset_name = dataset
        self.models_base_dir = os.path.join('src','training','models', 
                                            f'experiment_{self.exp_id}', f'{self.exp_iteration}','nodes')
        self.current_round=0

        manager = Manager()
        metrics_dict = {round_num: {'accuracy': 0, 'loss': 0} for round_num in range(self.num_rounds)}
        self.metrics_dict = manager.dict(metrics_dict)
    def load_data(self):
        """ Load the data for the simulation. """
        print('Loading data')
        self.dataset = load_data(self.dataset)
        
    def run(self):
        """
        Run the simulation.
        """
        print('Starting simulation')
        print(f'Number of nodes: {self.num_nodes}')
        print(f'Number of workers: {self.num_workers}')
        print(f'Number of rounds: {self.num_rounds}')
        print(f'Epochs per round: {self.epochs_per_round}')
        print(f'Batch size: {self.batch_size}')
        print(f'Number of samples: {self.num_samples}')
        print(f'Aggregation method: {self.aggregation_method}')
        print(f'Attack method: {self.attack_method}')

        for round_num in range(self.num_rounds):
            print(f'\n\tStarting round {round_num}')
            # train models
            if not os.path.exists(os.path.join(self.models_base_dir, f'round_{self.current_round}')):
                os.makedirs(os.path.join(self.models_base_dir, f'round_{self.current_round}'))
            self.train_network()
            self.aggregate_network()

            # delete files from current round
            if self.current_round>0:
                prev_dir = os.path.join(self.models_base_dir, f'round_{self.current_round-1}')
                for file in os.listdir(prev_dir):
                    os.remove(os.path.join(prev_dir, file))

            self.current_round+=1
    def run_tasks(self, processes):
        self.processes = processes
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        # kill all processes just in case
        for p in processes:
            p.terminate()
        self.processes = []

    def train_network(self):
        """
        Train the models on the nodes.
        """
        print('Training models')
        for idx in range(0,self.num_nodes, self.num_workers):
            worker_hashes = [num for num in range(idx, idx+self.num_workers) if num<self.num_nodes]
            processes = []
            for worker in worker_hashes:
                p = multiprocessing.Process(target=self.train_worker, args=(worker,))
                processes.append(p)
            self.run_tasks(processes)
    def train_worker(self, node_hash):
        """
        Train a model for a worker.

        Args:
            worker (int): The worker number.

        Returns:
            None
        """
        print(f'Training model for node {node_hash} round {self.current_round}')
        # create model
        model = load_model(self.dataset_name)(epochs=self.epochs_per_round, batch_size=self.batch_size, num_samples=self.num_samples, 
                         node_hash=node_hash,
                         evaluating=False)
        if self.current_round>0:
            model.load_model(os.path.join(self.models_base_dir, f'round_{self.current_round}', f'node_{node_hash}.pt'))
    
        start_index = (node_hash*self.num_samples)% len(self.dataset)
        end_index = start_index + self.num_samples
        # print(f'Node {node_hash} training on samples {start_index} to {end_index}')
        if end_index < len(self.dataset):
            subset_dataset = Subset(self.dataset, list(range(start_index, end_index)))
        else:
            subset_dataset = Subset(self.dataset, list(range(start_index, len(self.dataset))))
            subset_dataset += Subset(self.dataset, list(range(0, end_index%len(self.dataset))))
        model.train(subset_dataset)
    
        # save model in current round dir
        filename = os.path.join(self.models_base_dir, f'round_{self.current_round}', f'node_{node_hash}.pt')
        model.save_model(filename)
        print("Saved model for node ", node_hash, "to", filename)
    def aggregate_network(self):
        # save model in round+1 dir
        print('\nAggregating models')

        # malicious nodes should aggregate first
        for idx in range(0, len(self.nodes), self.num_workers):
            worker_hashes = [num for num in self.nodes[idx:idx+self.num_workers] \
                                if num<len(self.nodes) and num in self.malicious_nodes]
            if len(worker_hashes)==0:
                continue
            
            print("Malicous worker hashes: ", worker_hashes)
            processes = []
            for worker in worker_hashes:
                p = multiprocessing.Process(target=self.aggregate_worker, args=(worker, self.metrics_dict))
                processes.append(p)
            self.run_tasks(processes)

        # benign nodes aggregate
        for idx in range(0, len(self.nodes), self.num_workers):
            worker_hashes = [num for num in self.nodes[idx:idx+self.num_workers] \
                             if num<len(self.nodes) and num not in self.malicious_nodes]
            if len(worker_hashes)==0:
                continue

            print("Benign worker hashes: ", worker_hashes)
            processes = []
            for worker in worker_hashes:
                p = multiprocessing.Process(target=self.aggregate_worker, args=(worker, self.metrics_dict))
                processes.append(p)
            self.run_tasks(processes)

        time.sleep(.5)
        

    
    def aggregate_worker(self, node_hash, metrics_dict):
        """
        Aggregate the models on a worker.

        Args:
            worker (int): The worker number.
            metrics_dict (dict): A dictionary to store the metrics.
        Returns:
            None
        """
        neighbors = self.topology.get_neighbors(node_hash)
        is_malicous = self.topology[node_hash]['malicious']
        if not is_malicous:
            model_paths = [os.path.join(self.models_base_dir, f'round_{self.current_round}', f'node_{neighbor}.pt') 
                        for neighbor in neighbors]
            model_paths.append(os.path.join(self.models_base_dir, f'round_{self.current_round}', f'node_{node_hash}.pt'))
        else:
            model_paths = [os.path.join(self.models_base_dir, f'round_{self.current_round}', f'node_{neighbor}.pt')\
                           for neighbor in neighbors if not self.topology[neighbor]['malicious']]
        if (not is_malicous and len(model_paths)>1) or (is_malicous and len(model_paths)>0):
            agg_args = {'f': len(model_paths), 
                        'm': len([neighbor for neighbor in neighbors if self.topology[neighbor]['malicious']]),
                        'trimmed_mean_beta': experiment_params['trimmed_mean_beta']}
            aggregator = strategies.create_aggregator(node_hash, agg_args)
            print("Node ", node_hash, "aggregating")
            aggregated_model = aggregator.aggregate(model_paths)
            print("Node ", node_hash, "aggregation complete")
        else:
            # use current model
            aggregated_model = torch.load(os.path.join(self.models_base_dir, f'round_{self.current_round}', f'node_{node_hash}.pt'), weights_only=True)
        # load model
        model = load_model(self.dataset_name)(epochs=self.epochs_per_round, batch_size=self.batch_size, num_samples=self.num_samples,
                            node_hash=node_hash, evaluating=True,
                            device = torch.device('cuda:{}'.format(node_hash%torch.cuda.device_count())))
        model.model.load_state_dict(aggregated_model)

        # save model for next round
        save_dir = os.path.join(self.models_base_dir, f'round_{self.current_round+1}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model.save_model(os.path.join(save_dir, f'node_{node_hash}.pt'))
        print("Saved model for node ", node_hash, "to", os.path.join(save_dir, f'node_{node_hash}.pt'))
        # if malicious, dont evaluate. Instead, attack the model
        if self.topology[node_hash]['malicious']:
            #attacker
            attack_type =experiment_params['attack_type'].lower()
            attack_args = experiment_params['attack_args']
            attack_args['defense'] = experiment_params['aggregation'].lower()
            neighbors = self.topology.get_neighbors(node_hash)
            attack_args['nodes'] = len(neighbors)
            attack_args['malicious_nodes'] = len([neighbor for neighbor in neighbors if self.topology[neighbor]['malicious']])
            attacker = attacks.create_attacker(attack_type, attack_args, node_hash)
            if attack_type=='alie':
                print("starting attack alie")
                poisoned_model = attacker.attack(model_paths)
            else:
                print("starting attack")
                poisoned_model = attacker.attack(model.model.state_dict())
            #save the model in current round dir
            model.model.load_state_dict(poisoned_model)
            model.save_model(os.path.join(self.models_base_dir, f'round_{self.current_round}', f'node_{node_hash}.pt'))
        else:
            # evaluate model on whole dataset
            start_index = 0
            end_index = len(self.dataset)
            subset_dataset = Subset(self.dataset, list(range(start_index, end_index)))
            accuracy, loss = model.evaluate(subset_dataset)
            metrics_dict[self.current_round]['accuracy'] += accuracy 
            metrics_dict[self.current_round]['loss'] += loss
            print(f'Node {node_hash} round {self.current_round} accuracy: {accuracy} loss: {loss}')
            # save to node metrics json
        
            eval.save_node_metrics(node_hash, accuracy, loss, self.exp_id, self.exp_iteration)

    def __del__(self):
        torch.cuda.empty_cache()
        delete_files(self.exp_id, self.exp_iteration)

def delete_files(exp_id, iteration, node_metrics=False):
    """
    Delete files in the models and core* files
    """
    models_dir = os.path.join('src','training','models', f'experiment_{exp_id}', f'{iteration}','nodes')
    if not os.path.exists(models_dir):
        return
    for round_dir in os.listdir(models_dir):
        for file in os.listdir(os.path.join(models_dir, round_dir)):
            os.remove(os.path.join(models_dir, round_dir, file))

    # remove json
    if node_metrics:
        node_metrics_dir = os.path.join('src','training','results',f'experiment_{exp_id}',f'{iteration}','node_metrics')
        if os.path.exists(node_metrics_dir):
            for file in os.listdir(node_metrics_dir):
                os.remove(os.path.join(node_metrics_dir, file))

    # remove core files
    for file in os.listdir('.'):
        if file.startswith('core'):
            os.remove(file)

trainer = None
def run_simulation(params):
    """
    Runs the simulation with the experiment arguments.

    - Creates a network graph, adds nodes to the network, and makes connections between them.
    - Starts the nodes.
    - Starts the server.

    Args:
        params (dict): A dictionary of parameters for the simulation.

    Returns:
        None
    """
    global topology, trainer
    ### get args
    num_nodes = params['nodes']
    malicious_proportion = params['malicious_proportion']
    exp_id = params['id']
    topology = graph.Topology()
    topology_file = experiment_params['topology_file']
    if not params['use_saved_topology']:
        print(f'Created topology with {num_nodes} nodes')
        malicous_nodes = random.sample(range(num_nodes), int(malicious_proportion*num_nodes))
        print("Malicious nodes: ", malicous_nodes)
        #### add nodes to network
        if experiment_params['topology']=='random':
            edge_density = params['edge_density']
            topology.create_random_graph(num_nodes, edge_density, malicous_nodes)
        elif experiment_params['topology']=='small-world':
            k = params['small_world_k']
            p = params['small_world_beta']
            topology.create_small_world_graph(num_nodes, k, p, malicous_nodes)
        elif experiment_params['topology']=='scale-free':
            m = params['scale_free_m']
            topology.create_scale_free_graph(num_nodes, m,malicous_nodes)
        elif experiment_params['topology'] == 'two-f-1':
            topology.create_2f1_disjoint_graph(num_nodes, malicous_nodes)
        else:
            raise ValueError('Invalid topology: must be random, small-world, or scale-free')


        # save topology
        topology.save(topology_file)
    else:
        topology.load(topology_file)
        print('Using saved topology')

    dfl_trainer = DFLTrainer(num_nodes=num_nodes, topology=topology, num_workers=params['num_workers'], num_rounds=params['rounds'],
                             epochs_per_round=params['epochs_per_round'], batch_size=params['batch_size'], num_samples=params['num_samples'],
                             aggregation_method=params['aggregation'], attack_method=params['attack_type'], dataset=params['dataset'])


    trainer = dfl_trainer
    dfl_trainer.load_data()
    dfl_trainer.run()

    eval.save_results(experiment_params)
    eval.make_plot(exp_id)

def signal_handler(sig, frame):
    global trainer
    if trainer is not None:
        for p in trainer.processes:
            p.kill()
    torch.cuda.empty_cache()
    delete_files(experiment_params['id'], experiment_params['iteration'])
    exit(0)

if __name__=='__main__':
    experiment_yaml = os.path.join('src','config', 'experiment.yaml')
    with open(experiment_yaml) as f:
        experiment_params = yaml.safe_load(f)
    print("Starting simulation with the following parameters:\n")
    print(experiment_params)
    print()

    delete_files(experiment_params['id'], experiment_params['iteration'], node_metrics=True)
    run_simulation(experiment_params)