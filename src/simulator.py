'''
This file is the main entry point for the simulator. It sets up the server and runs the nodes.
'''
from network import graph
import multiprocessing
from multiprocessing import Manager
import os
import random
import yaml
import logging
import eval
import torch
from torch.utils.data import Subset
import signal
from training.models.torch.digit_classifier import DigitClassifier, BaseModel
from training.models.torch.digit_classifier import load_data as MNIST_load_data
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
                 num_samples, aggregation_method, attack_method):
        self.num_nodes = num_nodes
        self.topology = topology
        self.num_workers = num_workers
        self.num_rounds = num_rounds
        self.epochs_per_round = epochs_per_round
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.aggregation_method = aggregation_method
        self.attack_method = attack_method

        self.nodes = list(range(self.num_nodes))
        self. malicious_nodes = [node_hash for node_hash in self.nodes if \
                                 self.topology[node_hash]['malicious']]
        
        
        self.mnist_dataset = None
        self.models_base_dir = os.path.join('src','training','models','nodes')
        self.current_round=0

        manager = Manager()
        metrics_dict = {round_num: {'accuracy': 0, 'loss': 0} for round_num in range(self.num_rounds)}
        self.metrics_dict = manager.dict(metrics_dict)
    def load_data(self):
        """ Load the data for the simulation. """
        print('Loading data')
        self.mnist_dataset = MNIST_load_data()
        
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
            self.current_round+=1
    def run_tasks(self, processes):
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        # kill all processes just in case
        for p in processes:
            p.terminate()

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
        model = DigitClassifier(epochs=self.epochs_per_round, batch_size=self.batch_size, num_samples=self.num_samples, 
                         node_hash=node_hash,
                         evaluating=False)
        if self.current_round>0:
            # load model in current round dir (aggregated from previous round)
            #print(os.listdir(os.path.join(self.models_base_dir, f'round_{self.current_round}')))
            model.load_model(os.path.join(self.models_base_dir, f'round_{self.current_round}', f'node_{node_hash}.pt'))
    
        start_index = (node_hash*self.num_samples)% len(self.mnist_dataset)
        end_index = start_index + self.num_samples
        # print(f'Node {node_hash} training on samples {start_index} to {end_index}')
        subset_dataset = Subset(self.mnist_dataset, list(range(start_index, end_index)))
        model.train(subset_dataset)
    
        # save model in current round dir
        model.save_model(os.path.join(self.models_base_dir, f'round_{self.current_round}', f'node_{node_hash}.pt'))
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

        time.sleep(2)
        

    
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

        aggregator = strategies.create_aggregator(node_hash)
        aggregated_model = aggregator.aggregate(model_paths)
        print("Node ", node_hash, "aggregation complete")

        # load model
        model = DigitClassifier(epochs=self.epochs_per_round, batch_size=self.batch_size, num_samples=self.num_samples,
                            node_hash=node_hash, evaluating=True)
        model.model.load_state_dict(aggregated_model)

        # save model for next round
        save_dir = os.path.join(self.models_base_dir, f'round_{self.current_round+1}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model.save_model(os.path.join(save_dir, f'node_{node_hash}.pt'))
        # if malicious, dont evaluate. Instead, attack the model
        if self.topology[node_hash]['malicious']:
            #attacker
            attack_type =experiment_params['attack_type']
            attack_args = experiment_params['attack_args']
            attack_args['defense'] = experiment_params['aggregation']
            attacker = attacks.create_attacker(attack_type, attack_args, node_hash)
            poisoned_model = attacker.attack(model.model.state_dict())
            #save the model in current round dir
            model.model.load_state_dict(poisoned_model)
            model.save_model(os.path.join(self.models_base_dir, f'round_{self.current_round}', f'node_{node_hash}.pt'))
        else:
                    # evaluate model on whole dataset
            start_index = 0
            end_index = len(self.mnist_dataset)
            subset_dataset = Subset(self.mnist_dataset, list(range(start_index, end_index)))
            accuracy, loss = model.evaluate(subset_dataset)
            metrics_dict[self.current_round]['accuracy'] += accuracy
            metrics_dict[self.current_round]['loss'] += loss
            print(f'Node {node_hash} round {self.current_round} accuracy: {accuracy} loss: {loss}')
            # save to node metrics json
            eval.save_node_metrics(node_hash, accuracy, loss)

    def __del__(self):
        torch.cuda.empty_cache()
        delete_files()

# ______________ Simulator ______________
# def run_nodes(num_nodes):
#     """
#     Run the specified number of nodes.

#     Args:
#         num_nodes (int): The number of nodes to run.

#     Returns:
#         None
#     """
#     processes = []
#     for i in range(num_nodes):
#         # run as separate process to avoid GIL
#         node_file = os.path.join(os.getcwd(),'src','node.py')
#         process = subprocess.Popen(['python', node_file, '--id', str(i)])
#         processes.append(process)
#         print(f'Started node {i}')
#     return processes
# def wait_for_nodes(processes: list):
#     # kill nodes if server is killed
#     def kill_nodes(signum, frame):
#         for process in processes:
#             process.kill()
#         print('Killed all nodes')
#         torch.cuda.empty_cache()
#         # clear model directory
#         delete_files()
#         exit(0)
    # signal.signal(signal.SIGINT, kill_nodes)
    
    # for process in processes:
    #     process.wait()
    # print('All nodes finished')
def delete_files():
    """
    Delete files in the models and core* files
    """
    model_dir = os.path.join('src','training','models','nodes')
    # for each round dir
    for round_dir in os.listdir(model_dir):
        round_dir = os.path.join(model_dir, round_dir)
        # delete each file in directory
        for file in os.listdir(round_dir):
            os.remove(os.path.join(round_dir, file))
    node_results_dir = os.path.join('src','training','results','node_metrics')
    for file in os.listdir(node_results_dir):
        os.remove(os.path.join(node_results_dir, file))

    # remove core files
    
    for file in os.listdir('.'):
        if file.startswith('core'):
            os.remove(file)
def signal_handler(sig, frame):
    print('Exiting simulation')
    delete_files()
    torch.cuda.empty_cache()
    exit(0)
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
    global topology
    ### get args
    num_nodes = params['nodes']
    malicious_proportion = params['malicious_proportion']
    exp_id = params['id']

    if not params['use_saved_topology']:
        #### create network graph
        topology = graph.create_graph()
        print(f'Created topology with {num_nodes} nodes')
        malicous_nodes = random.sample(range(num_nodes), int(malicious_proportion*num_nodes))
        print("Malicious nodes: ", malicous_nodes)
        #### add nodes to network
        if experiment_params['topology']=='random':
            edge_density = params['edge_density']
            topology.create_random_graph(num_nodes, edge_density, malicous_nodes)
        elif experiment_params['topology']=='scale-free':
            m0=experiment_params['m0']
            m=experiment_params['m']
            topology.create_scale_free_graph(num_nodes, m0, m, malicous_nodes)
        else:
            raise ValueError('Invalid topology: must be random, small-world, or scale-free')


        # save topology
        topology.save(os.path.join(os.getcwd(), 'src','config','topology.json'))
    else:
        print('Using saved topology')

    # processes = run_nodes(num_nodes)
    # wait_for_nodes(processes)

    # interrupt signal
    signal.signal(signal.SIGINT, signal_handler)


    dfl_trainer = DFLTrainer(num_nodes=num_nodes, topology=topology, num_workers=params['num_workers'], num_rounds=params['rounds'],
                             epochs_per_round=params['epochs_per_round'], batch_size=params['batch_size'], num_samples=params['num_samples'],
                             aggregation_method=params['aggregation'], attack_method=params['attack_type'])

    
    dfl_trainer.load_data()
    dfl_trainer.run()

    exp_id = params['id']

    eval.save_results(exp_id)
    eval.make_plot(exp_id)
    delete_files()


if __name__=='__main__':
    experiment_yaml = os.path.join('src','config', 'experiment.yaml')
    with open(experiment_yaml) as f:
        experiment_params = yaml.safe_load(f)
    print("Starting simulation with the following parameters:\n")
    print(experiment_params)
    print()
    run_simulation(experiment_params)
