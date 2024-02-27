import argparse
import time
import os
import requests

log_file=None

def start_client():
    global log_file
    # get topology
    url = f'http://{args.simulator}/topology'
    response = requests.get(url)
    if response.status_code != 200:
        with open(log_file, 'a') as f:
            f.write(f'Error getting topology: {response.status_code}\n')
        return
    topology = response.json()
    with open(log_file, 'a') as f:
        f.write(f'Got topology: {topology}\n')

    # get my info
    my_info = topology[str(args.id)]
    with open(log_file, 'a') as f:
        f.write(f'My info: {my_info}\n')

    # get neighbors
    neighbors = my_info['edges']
    with open(log_file, 'a') as f:
        f.write(f'My neighbors: {neighbors}\n')

    # send a message to each neighbor

    with open(log_file, 'a') as f:
        f.write('Done\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start simulation of DFL network.')
    parser.add_argument('--simulator', type=str, help='ip:port of the simulator')
    parser.add_argument('--id', type=int, help='id of this client')
    args = parser.parse_args()

    log_dir = os.path.join(os.path.abspath(__file__).strip('client.py'),'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f'client_{args.id}.log')
    with open(log_file, 'w') as f:
        f.write(f'Client {args.id} started\n')
        f.write(f'Simulator: {args.simulator}\n\n')

    # wait for the simulator to start
    time.sleep(3)
    start_client()