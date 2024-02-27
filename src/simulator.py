import argparse
from network import graph
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import subprocess
import os
import random
import yaml
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
    for i in range(num_clients):
        # run as separate process to avoid GIL
        client_file = os.path.join(os.path.abspath(__file__).strip('simulator.py'), 'client.py')
        subprocess.Popen(['python3', client_file, '--simulator', f'localhost:{server_port}', '--id', str(i)])
        print(f'Started client {i}')

def start_simulation(params):
    """
    Starts the simulation with the given arguments.

    - Creates a network graph, adds users to the network, and makes connections between them.
    - Starts the clients.
    - Starts the server.

    Args:
        args (argparse.Namespace): The command-line arguments.

    Returns:
        None
    """
    global topology
    ### get args
    num_nodes = params['nodes']
    edge_density = params['edge_density']
    malicious_proportion = params['malicious_proportion']
    attack_type = params['attack_type']
    attack_strength = params['attack_strength']

    #### create network graph
    topology = graph.create_graph()
    print(f'Created topology with {num_nodes} nodes')

    #### add users to network
    for i in range(num_nodes):
        is_malicious =  random.random() < malicious_proportion
        topology.add_user(i, f'localhost', 6000+i, malicious=is_malicious)
    
    #### add edges to graph
    topology.make_connections(p=edge_density)

    # print(topology)
    # print()
    # print(topology.to_json())

    run_clients(num_nodes)

    uvicorn.run(app, host="localhost", port=server_port)


if __name__=='__main__':
    experiment_yaml = os.path.join(os.path.abspath(__file__).strip('simulator.py'), 'experiment.yaml')
    with open(experiment_yaml) as f:
        experiment_params = yaml.safe_load(f)
    print("Starting simulation with the following parameters:\n")
    print(experiment_params)
    print()
    start_simulation(experiment_params)
