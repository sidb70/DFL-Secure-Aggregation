import random
import json
from typing import Mapping
import networkx
import matplotlib.pyplot as plt
# seed
random.seed(42)

class Topology(networkx.Graph):
    def __init__(self):
        super().__init__()
        self.network_type = None
    def add_node(self, node_num, malicious=False):
        if node_num in self.nodes:
            return
        super().add_node(node_num, malicious=malicious)

    def add_edge(self, node1, node2):
        super().add_edge(node1, node2)

    def get_neighbors(self, node):
        return set(self.neighbors(node))
    
    def load_from_graph(self, graph: networkx.Graph):
        for node in graph.nodes:
            self.add_node(node)
        for node1, node2 in graph.edges:
            self.add_edge(node1, node2)
    def create_random_graph(self, num_nodes, edge_density, malicious_nodes=[]):
        self.network_type = 'random'
        for i in range(num_nodes):
            is_malicious = i in malicious_nodes
            self.add_node(i, malicious=is_malicious)
        
        for node1 in range(num_nodes):
            total_deg = math.ceil(int(edge_density*num_nodes))
            total_deg -= len(self.get_neighbors(node1))
            connections = set(self.get_neighbors(node1))
            while len(connections) < total_deg:
                node2 = random.randint(0, num_nodes-1)
                if node2 != node1 and node2 not in connections:
                    connections.add(node2)
            for node2 in connections:
                self.add_edge(node1, node2)
            # print(len(self.get_neighbors(node1)))
        
    def create_small_world_graph(self, num_nodes, k, b, malicious_nodes = []):
        self.network_type = 'small_world'
        ws = networkx.watts_strogatz_graph(num_nodes, k, b)
        self.load_from_graph(ws)
        self.set_malicous(malicious_nodes)

    def create_scale_free_graph(self, num_nodes, m, malicious_nodes = []):
        self.network_type = 'scale_free'
        ba = networkx.barabasi_albert_graph(n=num_nodes, m=m)
        self.load_from_graph(ba)
        self.set_malicous(malicious_nodes)

    def save(self, filename):
        # save as json
        with open(filename, 'w') as f:
            json.dump(networkx.node_link_data(self), f)
    
    def load(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        self.directed = data['directed']
        for node in data['nodes']:
            self.add_node(node['id'], malicious=node['malicious'])
            
        for link in data['links']:
            self.add_edge(link['source'], link['target'])
    def draw(self):
        if self.network_type is None:
            raise ValueError("Network not created yet")
        if self.network_type == 'random' or self.network_type == 'scale_free':
            pos = networkx.spring_layout(self, k=0.2, iterations=100)
        elif self.network_type == 'small_world':
            pos = networkx.circular_layout(self)
        fig, ax = plt.subplots(figsize=(10, 10))
        networkx.draw(self, pos, with_labels=False, node_size=20, 
                  node_color=['r' if self.nodes[node]['malicious'] else 'b' for node in self.nodes],
                  ax=ax)
        return fig, ax


    def set_malicous(self, nodes):
        for node in nodes:
            self.nodes[node]['malicious'] = True
    
    def __getitem__(self, key):
        return self.nodes[key]
import math
if __name__== '__main__':
    n=128
    topology = Topology()
    # malicious_proportion = 0.6
    # num_malicious = int(math.ceil(128*malicious_proportion))

    # malicious_nodes = random.sample(range(128), num_malicious)
    # # topology.create_random_graph(num_nodes=128, edge_density=0.15, malicious_nodes=malicious_nodes)


    # topology.create_scale_free_graph(num_nodes=128, m=10)
    # #sorted_nodes_by_deg = sorted(topology.nodes, key=lambda x: topology.degree(x), reverse=True)
    # #malicious_nodes = sorted_nodes_by_deg[:num_malicious]
    # topology.set_malicous(malicious_nodes)

    
    # .graph.load('/mnt/home/bhatta70/Documents/DFL-Secure-Aggregation/src/config/topology.json')
    
    # graph.create_small_world_graph(num_nodes=128, k=k, b=beta, malicious_nodes=[])
    # poisoned = set()
    # for node in range(len(graph.nodes)):
    #     neighbors = graph.get_neighbors(node)
    #     for neighbor in neighbors:
    #         # if neighbor is a rewired
    #         if abs(node - neighbor) > k and neighbor not in poisoned:
    #             poisoned.add(node)
    #             graph.nodes[node]['malicious'] = True
    #             break  
    
    #ngraph = networkx.Graph()

    beta = .15
    k=10
    topology.create_small_world_graph(n, k, beta)
    #topology.set_malicous(malicious_nodes)
    #topology.load('/mnt/home/bhatta70/Documents/DFL-Secure-Aggregation/src/config/topology.json')

    # get all rewires
    rewires = set()
    malicious=[]
    for node in topology.nodes:
        neighbors = topology.get_neighbors(node)
        for neighbor in neighbors:
            edge = (node, neighbor) if node < neighbor else (neighbor, node)
            if abs(node%n - neighbor%n) > k and edge not in rewires:
                rewires.add(edge)
                malicious.append(node)
                break
    topology.set_malicous(malicious)
    print(malicious)
    print(len(malicious))

    # all_edges = set()
    # total_rewires=0
    # total_poisoned_rewires=0
    # for node in topology.nodes:
    #     neighbors = topology.get_neighbors(node)
    #     for neighbor in neighbors:
    #         edge = (node, neighbor) if node < neighbor else (neighbor, node)
    #         if abs(node - neighbor) > k and edge not in all_edges:
    #             total_rewires += 1
    #             if topology.nodes[node]['malicious'] or topology.nodes[neighbor]['malicious']:
    #                 total_poisoned_rewires += 1
    #         all_edges.add(edge)
    # print("total rewires",total_rewires)
    # print("total poisoned rewires",total_poisoned_rewires)
    # print("total edges",len(all_edges))
    # honest_to_malicous_connections = 0
    # total_honest_connections = 0
    # for node in topology.nodes:
    #     if topology.nodes[node]['malicious']:
    #         for neighbor in topology.get_neighbors(node):
    #             if not topology.nodes[neighbor]['malicious']:
    #                 honest_to_malicous_connections += 1
    #     else:
    #         for neighbor in topology.get_neighbors(node):
    #             if not topology.nodes[neighbor]['malicious']:
    #                 total_honest_connections += 1

    # print("num malicious",sum([1 for node in topology.nodes if topology.nodes[node]['malicious']]))
    # #print(graph)
    # print(honest_to_malicous_connections)
    # print(total_honest_connections)

    # topology.save('/mnt/home/bhatta70/Documents/DFL-Secure-Aggregation/src/config/topology.json')
    #print(graph)
    
    #print(topology.get_neighbors(25))
    # draw as ring lattice
    pos = networkx.circular_layout(topology)
    #pos = networkx.spring_layout(topology, k=0.2, iterations=100)
    fig, ax = plt.subplots(figsize=(10, 10))
    #edge_colors = ['red' if topology.nodes[node]['malicious'] or topology.nodes[neighbor]['malicious'] else 'blue' for node, neighbor in topology.edges]
    networkx.draw(topology, pos, with_labels=False, node_size=20, 
                  node_color=['r' if topology.nodes[node]['malicious'] else 'b' for node in topology.nodes],
                  ax=ax)
    
    plt.savefig('/mnt/home/bhatta70/Documents/DFL-Secure-Aggregation/src/config/topology.png')