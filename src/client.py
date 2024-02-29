import argparse
import time
import os
import requests
from network import listen
from training.models import Model
import yaml
from logger import Logger
import threading
from network import listen
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
        self.topology = response.json()
        # convert keys to int
        self.topology = {int(k):v for k,v in self.topology.items()}

        # get my
        my_info = self.topology[int(args.id)]
        self.hostname = my_info['ip']
        self.port = my_info['port']

        # get neighbors
        self.neighbors = my_info['edges']
                
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
        self.recieved_models = []
        self.recieved_models_lock  = threading.Lock()
    def load_model(self, log=False):
        """
        Load the model
        """
        data_path = experiment_params['data_path']
        num_samples = experiment_params['num_samples']
        epochs = experiment_params['epochs']
        batch_size = experiment_params['batch_size']
        

        modelname = experiment_params['model_name']
        if modelname == 'loan_defaulter':
                from training.models.loan_defaulter import LoanDefaulter
                self.model = LoanDefaulter(data_path, num_samples, self.id, epochs, batch_size, logger)
        else:
            raise ValueError(f'Unknown model name: {modelname}')
        if log:
            logger.log(f'Loaded model {modelname}\n')
        

    def train_fl(self):
        # listen on a separate thread
        listen_thread = threading.Thread(target=listen.listen_for_models,\
                                        args=(self.hostname, self.port, 10, \
                                                self.logger,self.recieve_model))
        listen_thread.start()
        for round in range(self.rounds):
            logger.log(f'Round {round}\n')
            # train model
            self.model.train()
            self.send_model()
            # aggregate()
    def recieve_model(self, model):
        with self.recieved_models_lock:
            self.recieved_models.append(model)
        logger.log(f'Recieved model from {model["id"]}\n')
    def send_model(self):
        # send model to neighbors
        for neighbor in self.neighbors:
            neighbor_addr = self.topology[neighbor]['ip'] + ':' + str(self.topology[neighbor]['port'])
            url = f'http://{neighbor_addr}/'
            data = {'id': self.id, 'model': self.model}
            response = requests.post(url, data=data)
            if response.status_code != 200:
                logger.log(f'Error sending model to {neighbor}: {response.status_code}\n')
            else:
                logger.log(f'Sent model to {neighbor}\n')
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
