'''
This file is the main entry point for the simulator. It sets up the server and runs the clients.
'''
from network import graph
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import subprocess
import os
import random
import yaml
import logging
import eval
# ______________ Globals ______________
toplogy = None
server_port = 5999
# ______________ Setup ______________
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # This allows all HTTP methods
    allow_headers=["*"],  # This allows all headers
)

# ______________ Routes ______________
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/topology")
async def get_topology():
    global topology
    if topology is None:
        return {"error": "topology not set"}
    return topology.to_json()


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
        client_file = os.path.join('src','client.py')
        process = subprocess.Popen(['python3', client_file, '--simulator', f'localhost:{server_port}', '--id', str(i)])
        processes.append(process)
        print(f'Started client {i}')
    return processes
def wait_for_clients(processes: list):
    for process in processes:
        process.wait()
    print('All clients finished')

def eval_global_model():
    """
    Evaluate the global model.
    """
    eval.eval_global_model()

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
    edge_density = params['edge_density']
    malicious_proportion = params['malicious_proportion']

    #### create network graph
    topology = graph.create_graph()
    print(f'Created topology with {num_nodes} nodes')

    #### add users to network
    # choose malicious nodes
    malicous_nodes = random.sample(range(num_nodes), int(malicious_proportion*num_nodes))
    print("Malicious nodes: ", malicous_nodes)
    for i in range(num_nodes):
        is_malicious = i in malicous_nodes
        topology.add_user(i, f'localhost', 6000+i, malicious=is_malicious)
    
    #### add edges to graph
    topology.make_connections(p=edge_density)
    # save topology
    topology.save(os.path.join(os.getcwd(), 'src','config','topology.json'))
 
    # print(topology)
    # print()
    # print(topology.to_json())

    processes = run_clients(num_nodes)
    wait_for_clients(processes)
    eval_global_model()

    # delete models
        # clear model directory
    model_dir = os.path.join('src','training','models','clients')
    for file in os.listdir(model_dir):
        os.remove(os.path.join(model_dir, file))
    #
    # uvicorn.run(app, host="localhost", port=server_port)


if __name__=='__main__':
    experiment_yaml = os.path.join('src','config', 'experiment.yaml')
    with open(experiment_yaml) as f:
        experiment_params = yaml.safe_load(f)
    print("Starting simulation with the following parameters:\n")
    print(experiment_params)
    print()
    run_simulation(experiment_params)
