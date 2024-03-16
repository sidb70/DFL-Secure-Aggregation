import argparse
import time
import os
import requests
import json
from network import listen
import yaml
from logger import Logger
import threading
from network import listen
from training.models.torch import loan_defaulter
from training.models.torch.loan_defaulter import LoanDefaulter
import torch
from aggregation import strategies
from attack import attacks
# Load experiment parameters
with open(os.path.join('src','config', 'experiment.yaml')) as f:
    experiment_params = yaml.safe_load(f)
topology_json_file = os.path.join('src', 'config', 'topology.json')
with open(topology_json_file) as f:
    topology_json = json.load(f)

logger = None
class BaseClient:
    def __init__(self, id_num: int):
        """
        Initializes client info from the simulator manager
            - hostname
            - port
            - neighbors
            - is_synch
            - rounds
            - topology
            - current_round
            - listener
            - received_msgs
            - received_msgs_lock
            - aggregator
        """
        self.id = id_num
        self.hostname = None
        self.port = None
        self.neighbors = None
        self.am_malicious = False
        self.attack_type = None
        self.attack_strength = None
        self.is_synch = None
        self.rounds = None
        self.topology = None
        self.current_round = None
        self.listener = None
        self.received_msgs = None
        self.received_msgs_lock = None
        self.aggregator = None
        self.model = None
        # get topology
        self.topology = {int(node_id):val for node_id,val in topology_json.items()}
        
        logger.log(f'Loaded topology from {topology_json_file}\n')
        self.current_round = 0
        self.rounds = experiment_params['rounds']
        self.is_synch = experiment_params['synchronous']

        # get my info
        my_info = self.topology[int(args.id)]
        self.hostname = my_info['ip']
        self.port = my_info['port']
        self.listener = listen.ModelListener(self.hostname, self.port)

        # get neighbors
        self.neighbors = my_info['edges']

        logger.log(f'Got topology: {self.topology}\n')
        logger.log(f'My info: {my_info}\n')
        logger.log(f'My neighbors: {self.neighbors}\n')
        ## initialize received messages and lock
        if self.is_synch:
            self.received_msgs = {round_num: {neighbor:None for neighbor in self.neighbors} for round_num in range(self.rounds)}
        else:
            self.received_msgs = {round_num: {} for round_num in range(self.rounds)}
        self.received_msgs_lock  = threading.Lock() 
        self.load_aggregator()
        self.load_model()

    def load_aggregator(self):
        """
        Load the aggregator
        """
        aggregation = experiment_params['aggregation']
        self.aggregator = strategies.create_aggregator(aggregation, experiment_params, logger)
        logger.log(f'Loaded aggregator {aggregation}\n')
    def load_model(self):
        """
        Load the model
        """
        data_path = experiment_params['data_path']
        num_samples = experiment_params['num_samples']
        epochs = experiment_params['epochs_per_round']
        batch_size = experiment_params['batch_size']
        

        modelname = experiment_params['model_name']
        if modelname == 'loan_defaulter':
                self.model = LoanDefaulter(data_path, num_samples, self.id, epochs, batch_size, logger)
        else:
            raise ValueError(f'Unknown model name: {modelname}')
        logger.log(f'Loaded model {modelname}\n')
    def train_fl(self):
        # listen on a separate thread
        self.listener.listen_for_models()
        time.sleep(1.5) # wait for other clients to start
        for r in range(self.rounds):
            logger.log(f'Round {r}\n')
            self.model.train(plot=True)
            self.send_model()
            self.aggregate()
            self.current_round += 1
        loan_defaulter.fig.savefig(os.path.join(os.getcwd(),'src','training','results',f"client{self.id}.png"))
        time.sleep(3) # wait for other clients to finish
        logger.log(f'Training complete\n')
        self.listener.shutdown()
    def recieve_model(self, msg: dict):
        if not self.is_synch and msg['round'] < self.current_round:
            return
        msg['id'] = int(msg['id'])
        msg['round'] = int(msg['round'])
        msg['num_samples'] = int(msg['num_samples'])
        model_path= msg['model_path']
        msg['model'] = torch.load(model_path)
        with self.received_msgs_lock:
            if not self.is_synch and msg['round'] >= self.current_round:
                # if async, only keep the latest model for each neighbor
                self.received_msgs[self.current_round][msg['id']] = msg
            elif self.is_synch:
                # if synch, keep all models for each round
                self.received_msgs[msg['round']][msg['id']] = msg

        logger.log(f'received message from {msg["id"]} for round {msg["round"]}\n')
    def wait_for_neighbors(self, waiting_for: set):
        while len(waiting_for)>0:
            # wait for neighbors to send their models
            with self.received_msgs_lock:
                for neighbor in self.neighbors:
                    if neighbor in waiting_for and self.received_msgs[self.current_round][neighbor] is not None:
                        waiting_for.remove(neighbor)
            time.sleep(0.1)
            if len(waiting_for) > 0:
                logger.log(f'Waiting for {waiting_for}\n')
    def post_model(self, data, neighbor):
        neighbor_addr = self.topology[neighbor]['ip'] + ':' + str(self.topology[neighbor]['port'])
        url = f'http://{neighbor_addr}/'
        try:
            response = requests.post(url, data=data)
            if response.status_code != 200:
                raise Exception(f'Status code: {response.status_code}')
            else:
                logger.log(f'Sent model to {neighbor} for round {self.current_round}\n')
        except Exception as e:
            if self.is_synch:
                raise Exception(f'Could not send model to {neighbor}: {e}')
            else:
                logger.log(f'Could not send model to {neighbor}: {e}\n')
    def aggregate(self):
        if self.is_synch:
            waiting_for = set(self.neighbors)
            self.wait_for_neighbors(waiting_for)
        models = []
        with self.received_msgs_lock:
            # aggregate models
            if len(self.received_msgs[self.current_round]) > 0:
                models = [(msg['model'], msg['num_samples']) for msg in self.received_msgs[self.current_round].values()]
                self.received_msgs[self.current_round] = {} # send models to garbage collector
        models.append((self.model.state_dict, self.model.num_samples))
        aggregated_model = self.aggregator.aggregate(models)
        self.model.load_state_dict(aggregated_model)
        logger.log(f'Aggregated models\n')
    def send_model(self):
        raise NotImplementedError
    
class BenignClient(BaseClient):
    def __init__(self, id_num: int):
        super().__init__(id_num)
        listen.set_globals(self.recieve_model, logger) # set the callback for the listen module
        time.sleep(1.5) # wait for other clients to start
    def send_model(self):
        # send model to neighbors
        if not os.path.exists(os.path.join('src', 'training', 'models', 'clients')):
            os.makedirs(os.path.join('src', 'training', 'models', 'clients'))
        model_path = os.path.join(os.getcwd(),'src', 'training', \
                                    'models', 'clients', f'client_{self.id}.pt')
        #print('model_path', model_path)
        torch.save(self.model.state_dict, model_path)
        data = {'id': self.id,'round': self.current_round, 'num_samples': self.model.num_samples, 'model_path': str(model_path)}
        for neighbor in self.neighbors:
            self.post_model(data, neighbor)
            #time.sleep(0.1)

class MaliciousClient(BaseClient):
    def __init__(self, id_num: int):
        '''
        Initialize the malicious client
        :param id_num: The id of the client
        :param logger: The logger object
        '''
        super().__init__(id_num)
        listen.set_globals(self.recieve_model, logger)
        self.load_attacker()
  
    def load_attacker(self):
        '''
        Load the attacker with the specified attack type and args
        :param log: Whether to log the attacker info
        '''
        self.attack_type =experiment_params['attack_type']
        attack_args = experiment_params['attack_args']
        attack_args['defense'] = experiment_params['aggregation']
        self.attacker = attacks.create_attacker(attack_type=self.attack_type, attack_args=attack_args, logger=logger)
        self.benign_neighbors = [int(node_id) for node_id in self.neighbors if not is_malicious(int(node_id))]
        
        logger.log(f'Loaded attacker with type {self.attack_type} and args {attack_args}\n')
        logger.log(f'Benign neighbors: {self.benign_neighbors}\n')

    def send_model(self):
        '''
        Malicious clients send an attacked model to their benign neighbors
        model sent depends on the attack type
        '''
        if self.is_synch:
            waiting_for = set(self.benign_neighbors)
            self.wait_for_neighbors(waiting_for)
        # send model to neighbors
        if not os.path.exists(os.path.join('src', 'training', 'models', 'clients')):
            os.makedirs(os.path.join('src', 'training', 'models', 'clients'))
        model_path = os.path.join(os.getcwd(),'src', 'training', \
                                    'models', 'clients', f'client_{self.id}.pt')

        if self.attack_type == 'noise':
            attack_model = self.attacker.attack(self.model.state_dict)
        # elif self.attack_type=='innerproduct':
        #     with self.received_msgs_lock:
        #         models = [msg['model'] for msg in self.received_msgs[self.current_round].values()]
        #     models.append(self.model.state_dict)
        #     attack_model = self.attacker.attack(models)
        torch.save(attack_model, model_path)
        data = {'id': self.id,'round': self.current_round, 'num_samples': self.model.num_samples, 'model_path': str(model_path)}
        logger.log(f'Sending poisoned model to neighbors for round {self.current_round}\n')
        for neighbor in self.benign_neighbors:
            self.post_model(data, neighbor)
            #time.sleep(0.1)
            
  
def is_malicious(id_num: int):
    return topology_json[str(id_num)]['malicious']
def main(args):
    if is_malicious(args.id):
        client = MaliciousClient(args.id)
    else:
        client = BenignClient(args.id)
    client.train_fl()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start a DFL node.')
    parser.add_argument('--id', type=int, help='id of this client')
    args = parser.parse_args()

    log_dir = os.path.join(os.getcwd(),'src','logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f'client_{args.id}.log')
    logger = Logger(log_filename)
    
    logger.log(f'Client {args.id} started')
    # wait for the simulator to start
    main(args)
