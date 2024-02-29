import argparse
import time
import os
import requests
from network import listen
from training.models import Model
import yaml
from logger import Logger
import threading
# Load experiment parameters
with open(os.path.join(os.path.abspath(__file__).strip('client.py'), 'experiment.yaml')) as f:
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
        self.rounds = experiment_params['rounds']
        
        self.load_client_info(log=True)
        self.load_model(log=True)
        assert self.model is not None, 'Model not loaded'

    def load_client_info(self,log=False):
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
        url = f'http://{args.simulator}/topology'
        response = requests.get(url)

        if response.status_code != 200:
            logger.log(f'Error getting topology: {response.status_code}\n')
            return
        topology = response.json()

        # get my
        my_info = topology[str(args.id)]
        self.hostname = my_info['ip']
        self.port = my_info['port']

        # get neighbors
        self.neighbors = my_info['edges']
                
        self.am_malicious = my_info['malicious']
        
        if log:
            logger.log(f'Got topology: {topology}\n')
            logger.log(f'My info: {my_info}\n')
            logger.log(f'My neighbors: {self.neighbors}\n')
            logger.log(f'Am malicious: {self.am_malicious}\n')

        if self.am_malicious: # get attack params if I am malicious
            self.attack_type = experiment_params['attack_type']
            self.attack_strength = experiment_params['attack_strength']
            if log:
                logger.log(f'I am an attacker with with attack type "{self.attack_type}" and strength {self.attack_strength}\n')

    def load_model(self, log=False):
        """
        Load the model
        """
        model_name = experiment_params['model_name']

        if log:
            logger.log(f'Loading model {model_name}\n')
        
        self.model = Model.get_model_by_name(model_name)

    def train_fl(self):
        if self.am_malicious:
            attack_type = experiment_params['attack_type']
            attack_strength = experiment_params['attack_strength']
            
            logger.log(f'I am an attacker with with attack type "{attack_type}" and strength {attack_strength}\n')
        for round in range(self.rounds):
            logger.log(f'Round {round}\n')
            # train()
            # send() and listen() concurrently
            # aggregate()

def main(args, logger):
    client = Client(args.id, args.simulator, logger)
    client.train_fl()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start simulation of DFL network.')
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
    time.sleep(3)
    main(args, logger)
