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
from training.models.torch.loan_defaulter import LoanDefaulter
import torch
from aggregation import strategies
from attack import attacks
# Load experiment parameters
with open(os.path.join('src','config', 'experiment.yaml')) as f:
    experiment_params = yaml.safe_load(f)

class Client:
    def __init__(self, id_num: int, simulator_addr: str, logger: Logger):
        self.id = id_num
        self.simulator_addr = simulator_addr
        self.logger = logger
        self.hostname = None
        self.port = None
        self.neighbors = None
        self.am_malicious = False
        self.attack_type = None
        self.attack_strength = None
        
        self.load_attributes(log=True)
        self.load_aggregator(log=True)
        self.load_model(log=True)
        self.load_attacker(log=True)
    def load_attributes(self,log=False):
        """
        Get client info from the simulator manager
            - hostname
            - port
            - neighbors
            - am_malicious
            - attack_type
            - attack_strength
        """
        # get topology
        topology_json_file = os.path.join('src', 'config', 'topology.json')
        with open(topology_json_file) as f:
            topology_json = json.load(f)
        self.topology = {int(node_id):val for node_id,val in topology_json.items()}
        if log:
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
        # get role
        self.am_malicious = my_info['malicious']

        if log:
            logger.log(f'Got topology: {self.topology}\n')
            logger.log(f'My info: {my_info}\n')
            logger.log(f'My neighbors: {self.neighbors}\n')
            logger.log(f'Am malicious: {self.am_malicious}\n')

        if self.am_malicious: # get attack params if I am malicious
            self.attack_type = experiment_params['attack_type']
            self.attack_strength = experiment_params['attack_strength']
            if log:
                logger.log(f'I am an attacker with with attack type "{self.attack_type}" and strength {self.attack_strength}\n')
        ## initialize received messages and lock
        if self.is_synch:
            self.received_msgs = {round_num: {neighbor:None for neighbor in self.neighbors} for round_num in range(self.rounds)}
        else:
            self.received_msgs = {round_num: {} for round_num in range(self.rounds)}
        self.received_msgs_lock  = threading.Lock() 

        listen.set_globals(self.recieve_model, self.logger) # set the callback for the listen module
        time.sleep(1.5) # wait for other clients to start
    def load_model(self, log=False):
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
        if log:
            logger.log(f'Loaded model {modelname}\n')
    def load_aggregator(self, log=False):
        """
        Load the aggregator
        """
        aggregation = experiment_params['aggregation']
        self.aggregator = strategies.create_aggregator(aggregation, experiment_params, self.logger)
    def load_attacker(self, log=False):
        if self.am_malicious:
            self.attacker = attacks.create_attacker(attack_type=self.attack_type, attack_strength=self.attack_strength, logger=self.logger)
            if log:
                logger.log(f'Loaded attacker with type {self.attack_type} and strength {self.attack_strength}\n')
        
    def train_fl(self):
        # listen on a separate thread
        self.listener.listen_for_models()
        for r in range(self.rounds):
            logger.log(f'Round {r}\n')
            # train model
            self.model.train()
            self.send_model()
            self.aggregate()
            self.current_round += 1
        logger.log(f'Training complete\n')
        #time.sleep(5) # wait for other clients to finish
        self.listener.shutdown()

        
    def recieve_model(self, msg: dict):
        msg['id'] = int(msg['id'])
        msg['round'] = int(msg['round'])
        msg['num_samples'] = int(msg['num_samples'])
        if not self.is_synch and msg['round'] < self.current_round:
            return
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
    def send_model(self):
        # send model to neighbors
        if not os.path.exists(os.path.join('src', 'training', 'models', 'clients')):
            os.makedirs(os.path.join('src', 'training', 'models', 'clients'))
        model_path = os.path.join(os.getcwd(),'src', 'training', \
                                    'models', 'clients', f'client_{self.id}.pt')
        #print('model_path', model_path)

        if not self.am_malicious:
            torch.save(self.model.state_dict, model_path)
        else:
            attack_model = self.attacker.attack(self.model.state_dict)
            torch.save(attack_model, model_path)
        data = {'id': self.id,'round': self.current_round, 'num_samples': self.model.num_samples, 'model_path': str(model_path)}
        for neighbor in self.neighbors:
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
                    raise e
                else:
                    logger.log(f'Could not send model to {neighbor}: {e}\n')
                    continue
        #time.sleep(0.1)
    def aggregate(self):
        if self.is_synch:
            waiting_for = set(self.neighbors)
            while len(waiting_for)>0:
                # wait for all neighbors to send their models
                with self.received_msgs_lock:
                    for neighbor in self.neighbors:
                        if neighbor in waiting_for and self.received_msgs[self.current_round][neighbor] is not None:
                            waiting_for.remove(neighbor)
                time.sleep(0.1)
                if len(waiting_for) > 0:
                    logger.log(f'Waiting for {waiting_for}\n')
        models = []
        with self.received_msgs_lock:
            # aggregate models
            if len(self.received_msgs[self.current_round]) > 0:
                models = [(msg['model'], msg['num_samples']) for msg in self.received_msgs[self.current_round].values() \
                            if msg['round'] >= self.current_round]
                self.received_msgs[self.current_round] = {} # send models to garbage collector
        models.append((self.model.state_dict, self.model.num_samples))
        aggregated_model = self.aggregator.aggregate(models)
        self.model.load_state_dict(aggregated_model)
        logger.log(f'Aggregated models\n')

def main(args, logger):
    client = Client(args.id, args.simulator, logger)
    client.train_fl()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start a DFL node.')
    parser.add_argument('--simulator', type=str, help='ip:port of the simulator')
    parser.add_argument('--id', type=int, help='id of this client')
    args = parser.parse_args()

    log_dir = os.path.join(os.path.abspath(__file__).strip('client.py'),'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f'client_{args.id}.log')
    logger = Logger(log_filename)
    
    logger.log(f'Client {args.id} started')
    logger.log(f'Simulator: {args.simulator}\n')
    # wait for the simulator to start
    #time.sleep(1)
    main(args, logger)
