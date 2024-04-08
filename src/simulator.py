'''
This file is the main entry point for the simulator. It sets up the server and runs the clients.
'''
from network import graph
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# import torch
import subprocess
import os
import random
import yaml
import logging
import eval
import torch
import signal

# seed
random.seed(42)
# ______________ Globals ______________
# toplogy = None
# server_port = 5999
# ______________ Setup ______________
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # This allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # This allows all HTTP methods
#     allow_headers=["*"],  # This allows all headers
# )

# ______________ Routes ______________
# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# @app.get("/topology")
# async def get_topology():
#     global topology
#     if topology is None:
#         return {"error": "topology not set"}
#     return topology.to_json()


# ______________ Simulator ______________
def run_clients(num_clients):
    """
    Run the specified number of clients.

    Args:
        num_clients (int): The number of clients to run.

    Returns:
        None
    """
    processes = []
    for i in range(num_clients):
        # run as separate process to avoid GIL
        client_file = os.path.join(os.getcwd(),'src','client.py')
        process = subprocess.Popen(['python', client_file, '--id', str(i)])
        processes.append(process)
        print(f'Started client {i}')
    return processes
def wait_for_clients(processes: list):
    # kill clients if server is killed
    def kill_clients(signum, frame):
        for process in processes:
            process.kill()
        print('Killed all clients')
        torch.cuda.empty_cache()
        # clear model directory
        delete_files()
        exit(0)
    signal.signal(signal.SIGINT, kill_clients)
    
    for process in processes:
        process.wait()
    print('All clients finished')
def delete_files():
    """
    Delete files in the models and core* files
    """
    model_dir = os.path.join('src','training','models','clients')
    # for each round dir
    for round_dir in os.listdir(model_dir):
        round_dir = os.path.join(model_dir, round_dir)
        # delete each file in directory
        for file in os.listdir(round_dir):
            os.remove(os.path.join(round_dir, file))

    # remove core files
    
    for file in os.listdir('.'):
        if file.startswith('core'):
            os.remove(file)

def run_simulation(params):
    """
    Runs the simulation with the experiment arguments.

    - Creates a network graph, adds users to the network, and makes connections between them.
    - Starts the clients.
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
        #### add users to network
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
    # print(topology)
    # print()
    # print(topology.to_json())
    processes = run_clients(num_nodes)
    wait_for_clients(processes)
    eval.eval_global_model()
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
