import random
import json
from typing import Mapping
import networkx
import matplotlib.pyplot as plt
# seed
random.seed(42)

# class Graph:
#     def __init__(self):
#         self.nodes = {}
#         self.edges = {}
#     def add_node(self, node_num, ip, port, malicious=False):
#         if node_num in self.nodes:
#             return
#         self.nodes[node_num] = {'ip': ip, 'port': port, 'malicious': malicious}
#         if node_num not in self.edges:
#             self.edges[node_num] = set()

#     def make_connections(self, p=1.0, directed=False):
#         '''
#         Make connections between nodes with probability p
#         :param p: probability of connection
#         :param directed: whether the graph is directed
#         '''
#         for node1 in self.nodes:
#             for node2 in self.nodes:
#                 if node1 == node2:
#                     continue
#                 if random.random() < p:
#                     self.add_edge(node1, node2, directed=directed)
#                     if not directed:
#                         self.add_edge(node2, node1, directed=directed)
#                 else:
#                     self.remove_edge(node1, node2, directed=directed)
#                     if not directed:
#                         self.remove_edge(node2, node1, directed=directed)
#     def add_edge(self, node1, node2, directed=False):
#         '''
#         Add an edge between two nodes
#         :param node1: node 1
#         :param node2: node 2
#         :param directed: whether the graph is directed
#         '''
#         if node1 in self.edges and node2 in self.edges[node1]:
#             return
#         if node1 not in self.edges:
#             self.edges[node1] = set()
#         if node2 not in self.edges:
#             self.edges[node2] = set()
#         self.edges[node1].add(node2)
#         if not directed:
#             self.edges[node2].add(node1)
#     def remove_edge(self, node1, node2, directed=False):
#         '''
#         Remove an edge between two nodes
#         :param node1: node 1
#         :param node2: node 2
#         :param directed: whether the graph is directed
#         '''
#         if node1 in self.edges and node2 in self.edges[node1]:
#             self.edges[node1].remove(node2)
#         if not directed:
#             if node2 in self.edges and node1 in self.edges[node2]:
#                 self.edges[node2].remove(node1)
#     def get_neighbors(self, node):
#         '''
#         Get the neighbors of a node
#         :param node: node
#         :return: set of neighbors
#         '''
#         if node in self.edges:
#             return self.edges[node]
#         return set()

#     def create_random_graph(self, num_nodes, edge_density, malicious_nodes):
#         '''
#         Create a random graph
#         num_nodes: number of nodes in the graph
#         edge_density: probability of connection between nodes
#         '''
#         for i in range(num_nodes):
#             is_malicious = i in malicious_nodes
#             self.add_node(i, f'localhost', 50000+i, malicious=is_malicious)
#         self.make_connections(p=edge_density)
#     def create_small_world_graph(self, num_nodes, k, b, malicious_nodes):
#         '''
#         Create a small world graph using the Watts-Strogatz model
#         num_nodes: number of nodes in the graph
#         k: number of nearest neighbors to connect
#         b: probability of rewiring
#         '''

#         for i in range(num_nodes):
#             is_malicious = i in malicious_nodes
#             self.add_node(i, f'localhost', 6000+i, malicious=is_malicious)
        
#         for node1 in range(num_nodes):
#             for node2 in range(node1 + 1, node1 + k//2 + 1):
#                 #if random.random() < pC:
#                 self.add_edge(node1, node2 % num_nodes)
#         # rewiring
#         all_edges = set((node1, node2) if node1 < node2 else (node2, node1) for node1 in range(num_nodes) for node2 in self.get_neighbors(node1))
#         # rewire each edge with probability b
#         for node1, node2 in list(all_edges):
#             if random.random() < b:
#                 new_neighbor = random.choice([node for node in range(num_nodes) if node != node1 and node not in self.get_neighbors(node1)])
#                 self.add_edge(node1, new_neighbor)
#                 self.remove_edge(node1, node2)
#         # rewired_edges = set()
#         # for node1 in range(num_nodes):
#         #     neighbors = list(self.get_neighbors(node1))
#         #     for neighbor in neighbors:
#         #         edge = (node1, neighbor) if node1 < neighbor else (neighbor, node1)
#         #         if random.random() < b and edge not in rewired_edges:
#         #             new_neighbor = random.choice([node for node in range(num_nodes) if node != node1 and node not in neighbors])
#         #             self.add_edge(node1, new_neighbor)
#         #             self.remove_edge(node1, neighbor)
#         #             rewired_edges.add(edge)
#     def create_scale_free_graph(self, num_nodes, m0, m, malicious_nodes):
#         '''
#         create a scale free graph using the Barabasi-Albert model
#         num_nodes: number of nodes in the graph
#         m0: number of initial nodes
#         m: number of edges to attach from each new node to existing nodes
#         '''
#         if num_nodes < m0:
#             raise ValueError("Number of nodes must be greater than m0")
#         for i in range(m0):
#             is_malicious = i in malicious_nodes
#             self.add_node(i, f'localhost', 50000+i, malicious=is_malicious)
#         self.make_connections(p=1.0)
#         remaining_malicious = len(malicious_nodes) - m0
#         for i in range(m0, num_nodes):
#             # choose m nodes to connect to
#             choices = [[node]*self.node_degree(node) for node in self.nodes]
#             choices = [item for sublist in choices for item in sublist]

#             # if i in malicious_nodes and remaining_malicious<1:
#             #     is_malicious = False
#             # else:
#             is_malicious = i in malicious_nodes
#             # remaining_malicious -= 1
        
#             self.add_node(i, f'localhost', 50000+i, malicious=is_malicious)

#             while self.node_degree(i) < m: # choose m neighbors
#                 neighbor = random.choice(choices)
#                 if neighbor not in self.get_neighbors(i):
#                     self.add_edge(i, neighbor)

#     def node_degree(self, node):
#         '''
#         Get the degree of a node
#         :param node: node
#         :return: degree of the node
#         '''
#         return len(self.edges[node])

#     def to_dict(self):
#         ''' 
#         Convert the graph to a dictionary
#         :return: dictionary representation of the graph
#         '''
#         graph_dict = {}
#         for node in self.nodes.keys():
#             graph_dict[node] = {
#                 "ip": self.nodes[node]['ip'],
#                 "port": self.nodes[node]['port'],
#                 "edges": list(self.get_neighbors(node)),
#                 "malicious": self.nodes[node]['malicious']
#             }
#         return graph_dict
#     def save(self, filename):
#         '''
#         Save the graph to a file
#         :param filename: name of the file
#         '''
#         with open(filename, 'w') as f:
#             json.dump(self.to_dict(), f)
#     def load(self, filename):
#         '''
#         Load the graph from a file into this graph object
#         :param filename: name of the file
#         '''
#         with open(filename, 'r') as f:
#             graph_dict = json.load(f)
#         for node, data in graph_dict.items():
#             self.add_node(int(node), data['ip'], int(data['port']), bool(data['malicious']))
#             for edge in data['edges']:
#                 self.add_edge(int(node), int(edge))

#     def __iter__(self):
#         return iter(self.nodes)
    
#     def __str__(self):
#         out = ""
#         for node, edges in self.edges.items():
#             if len(edges) == 0:
#                 out += f"{node}: None\n"
#             else:
#                 out += f"{node}: {edges}\n"
#             out+= f"malicious: {self.nodes[node]['malicious']}\n"
#         return out
#     def __repr__(self):
#         return self.__str__()
#     def __getitem__(self, key):
#         return self.nodes[key]
    

class Topology(networkx.Graph):
    def __init__(self):
        super().__init__()
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
        for i in range(num_nodes):
            is_malicious = i in malicious_nodes
            self.add_node(i, malicious=is_malicious)
        for node1 in range(num_nodes):
            for node2 in range(num_nodes):
                if node1 == node2:
                    continue
                if random.random() < edge_density:
                    self.add_edge(node1, node2)
    def create_small_world_graph(self, num_nodes, k, b, malicious_nodes = []):
        ws = networkx.watts_strogatz_graph(num_nodes, k, b)
        self.load_from_graph(ws)

    def create_scale_free_graph(self, num_nodes, m, malicious_nodes = []):
        ba = networkx.barabasi_albert_graph(n=num_nodes, m=m)
        self.load_from_graph(ba)

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

    def set_malicous(self, node):
        self.nodes[node]['malicious'] = True
    
    def __getitem__(self, key):
        return self.nodes[key]
import math
if __name__== '__main__':
    topology = Topology()

    # malicious_proportion = 0.1
    # selection_set = list(range(int(math.ceil(2*malicious_proportion*128))))
    # initial_malicious = random.sample(selection_set, math.ceil(len(selection_set)/2))
    # print(initial_malicious)
    # print(len(initial_malicious))
    # remaining_malicious = []
    # #remaining_malicious = [10 + 2*i for i in range(33)]
    # graph.create_scale_free_graph(num_nodes=128, m0=10, m=5,
    #                               malicious_nodes=initial_malicious)
    
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

    beta = .1
    k=10
    #ws_graph = networkx.watts_strogatz_graph(128, k, beta)
    #ba_graph = networkx.barabasi_albert_graph(n=128, m=5)
    topology.create_small_world_graph(128, k, beta)
    #topology.load('/mnt/home/bhatta70/Documents/DFL-Secure-Aggregation/src/config/topology.json')

    # get all rewires
    rewires = set()
    for node in topology.nodes:
        neighbors = topology.get_neighbors(node)
        for neighbor in neighbors:
            edge = (node, neighbor) if node < neighbor else (neighbor, node)
            if abs(node - neighbor) > k and edge not in rewires:
                rewires.add(edge)
                topology.set_malicous(node)

    all_edges = set()
    total_rewires=0
    total_poisoned_rewires=0
    for node in topology.nodes:
        neighbors = topology.get_neighbors(node)
        for neighbor in neighbors:
            edge = (node, neighbor) if node < neighbor else (neighbor, node)
            if abs(node - neighbor) > k and edge not in all_edges:
                total_rewires += 1
                if topology.nodes[node]['malicious'] or topology.nodes[neighbor]['malicious']:
                    total_poisoned_rewires += 1
            all_edges.add(edge)
    print("total rewires",total_rewires)
    print("total poisoned rewires",total_poisoned_rewires)
    print("total edges",len(all_edges))
    honest_to_malicous_connections = 0
    total_honest_connections = 0
    for node in topology.nodes:
        if topology.nodes[node]['malicious']:
            for neighbor in topology.get_neighbors(node):
                if not topology.nodes[neighbor]['malicious']:
                    honest_to_malicous_connections += 1
        else:
            for neighbor in topology.get_neighbors(node):
                if not topology.nodes[neighbor]['malicious']:
                    total_honest_connections += 1

    print("num malicious",sum([1 for node in topology.nodes if topology.nodes[node]['malicious']]))
    #print(graph)
    print(honest_to_malicous_connections)
    print(total_honest_connections)

    topology.save('/mnt/home/bhatta70/Documents/DFL-Secure-Aggregation/src/config/topology.json')
    #print(graph)

    
 
    G = networkx.Graph()
    for node in topology.nodes:
        G.add_node(node)
    for node in topology.nodes:
        for neighbor in topology.get_neighbors(node):
            G.add_edge(node, neighbor)
    # draw as ring lattice
    pos = networkx.circular_layout(G)
    # spacing
    # 4k resolution
    fig, ax = plt.subplots(figsize=(10, 10))
    networkx.draw(G, pos, with_labels=False, node_size=20, node_color=['r' if topology.nodes[node]['malicious'] else 'b' for node in topology.nodes])
    
    plt.savefig('/mnt/home/bhatta70/Documents/DFL-Secure-Aggregation/src/config/topology.png')