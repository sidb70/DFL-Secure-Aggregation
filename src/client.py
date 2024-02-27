import argparse
import time
import os
import requests
from network import listen
import yaml
from logger import Logger


logger = None
with open(os.path.join(os.path.abspath(__file__).strip('client.py'), 'experiment.yaml')) as f:
        experiment_params = yaml.safe_load(f)


def train_fl(topology, neighbors, am_malicious, rounds):
    global logger
    if am_malicious:
        attack_type = experiment_params['attack_type']
        attack_strength = experiment_params['attack_strength']
        
        logger.log(f'I am an attacker with with attack type "{attack_type}" and strength {attack_strength}\n')
    for round in range(rounds):
        logger.log(f'Round {round}\n')
        # train()
        # send() and listen() concurrently
        # aggregate()
def start_client():
    """
    Retrieve the topology from the simulator and TODO
    """
    global logger
    # get topology
    url = f'http://{args.simulator}/topology'
    response = requests.get(url)

    if response.status_code != 200:
        logger.log(f'Error getting topology: {response.status_code}\n')
        return
    topology = response.json()
    logger.log(f'Got topology: {topology}\n')

    # get my info
    my_info = topology[str(args.id)]
    logger.log(f'My info: {my_info}\n')

    # get neighbors
    neighbors = my_info['edges']
    logger.log(f'My neighbors: {neighbors}\n')

    # get attack params if I am malicious
    am_malicious = my_info['malicious']
    logger.log(f'Am malicious: {am_malicious}\n')
    if am_malicious:
        attack_type = experiment_params['attack_type']
        attack_strength = experiment_params['attack_strength']
        logger.log(f'I am an attacker with with attack type "{attack_type}" and strength {attack_strength}\n')

    rounds = experiment_params['rounds']
    logger.log(f'Starting DFL with {rounds} rounds\n')
    train_fl(topology, neighbors, am_malicious, rounds)

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
    start_client()