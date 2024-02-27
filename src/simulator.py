import argparse
from network import graph
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import subprocess
import os
import random
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

def start_simulation(args):
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
    num_nodes = args.nodes
    edge_density = args.edge_density
    malicious_proportion = args.malicious

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
    parser = argparse.ArgumentParser(description='Start simulation of DFL network.')
    parser.add_argument('--nodes', type=int, default=10, help='Number of nodes in DFL network')
    parser.add_argument('--edge_density', type=float, default=1.0, help='Density of connections in DFL network')
    parser.add_argument('--malicious', type=float, default=0.0, help='Proportion of malicious nodes in DFL network')
    args = parser.parse_args()
    print(f'Starting simulation with {args.nodes} nodes')
    print(f"Edge density: {args.edge_density}")
    print(f"Malicious proportion: {args.malicious}")
    start_simulation(args)
