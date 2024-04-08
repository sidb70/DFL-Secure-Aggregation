import random
import json
import bisect
# seed
random.seed(42)

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
    def add_node(self, node_num, ip, port, malicious=False):
        if node_num in self.nodes:
            return
        self.nodes[node_num] = {'ip': ip, 'port': port, 'malicious': malicious}
        if node_num not in self.edges:
            self.edges[node_num] = set()

    def make_connections(self, p=1.0, directed=False):
        '''
        Make connections between nodes with probability p
        :param p: probability of connection
        :param directed: whether the graph is directed
        '''
        for node1 in self.nodes:
            for node2 in self.nodes:
                if node1 == node2:
                    continue
                if random.random() < p:
                    self.add_edge(node1, node2, directed=directed)
                    if not directed:
                        self.add_edge(node2, node1, directed=directed)
                else:
                    self.remove_edge(node1, node2, directed=directed)
                    if not directed:
                        self.remove_edge(node2, node1, directed=directed)
    def add_edge(self, node1, node2, directed=False):
        '''
        Add an edge between two nodes
        :param node1: node 1
        :param node2: node 2
        :param directed: whether the graph is directed
        '''
        if node1 in self.edges and node2 in self.edges[node1]:
            return
        if node1 not in self.edges:
            self.edges[node1] = set()
        if node2 not in self.edges:
            self.edges[node2] = set()
        self.edges[node1].add(node2)
        if not directed:
            self.edges[node2].add(node1)
    def remove_edge(self, node1, node2, directed=False):
        '''
        Remove an edge between two nodes
        :param node1: node 1
        :param node2: node 2
        :param directed: whether the graph is directed
        '''
        if node1 in self.edges and node2 in self.edges[node1]:
            self.edges[node1].remove(node2)
        if not directed:
            if node2 in self.edges and node1 in self.edges[node2]:
                self.edges[node2].remove(node1)
    def get_neighbors(self, node):
        '''
        Get the neighbors of a node
        :param node: node
        :return: set of neighbors
        '''
        if node in self.edges:
            return self.edges[node]
        return set()

    def create_random_graph(self, num_nodes, edge_density, malicious_nodes):
        '''
        Create a random graph
        num_nodes: number of nodes in the graph
        edge_density: probability of connection between nodes
        '''
        for i in range(num_nodes):
            is_malicious = i in malicious_nodes
            self.add_node(i, f'localhost', 50000+i, malicious=is_malicious)
        self.make_connections(p=edge_density)
    def create_scale_free_graph(self, num_nodes, m0, m, malicious_nodes):
        '''
        create a scale free graph using the Barabasi-Albert model
        num_nodes: number of nodes in the graph
        m0: number of initial nodes
        m: number of edges to attach from each new node to existing nodes
        '''
        if num_nodes < m0:
            raise ValueError("Number of nodes must be greater than m0")
        for i in range(m0):
            is_malicious = i in malicious_nodes
            self.add_node(i, f'localhost', 50000+i, malicious=is_malicious)
        self.make_connections(p=1.0)
        for i in range(m0, num_nodes):
            # choose m nodes to connect to
            choices = [[node]*self.node_degree(node) for node in self.nodes]
            choices = [item for sublist in choices for item in sublist]

            is_malicious = i in malicious_nodes
            self.add_node(i, f'localhost', 50000+i, malicious=i in malicious_nodes)

            while self.node_degree(i) < m: # choose m neighbors
                neighbor = random.choice(choices)
                if neighbor not in self.node_edges(i):
                    self.add_edge(i, neighbor)

    def node_degree(self, node):
        '''
        Get the degree of a node
        :param node: node
        :return: degree of the node
        '''
        return len(self.edges[node])

    def to_dict(self):
        ''' 
        Convert the graph to a dictionary
        :return: dictionary representation of the graph
        '''
        graph_dict = {}
        for node in self.nodes.keys():
            graph_dict[node] = {
                "ip": self.nodes[node]['ip'],
                "port": self.nodes[node]['port'],
                "edges": list(self.get_neighbors(node)),
                "malicious": self.nodes[node]['malicious']
            }
        return graph_dict
    def save(self, filename):
        '''
        Save the graph to a file
        :param filename: name of the file
        '''
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f)
    def load(self, filename):
        '''
        Load the graph from a file into this graph object
        :param filename: name of the file
        '''
        with open(filename, 'r') as f:
            graph_dict = json.load(f)
        for node, data in graph_dict.items():
            self.add_node(node, data['ip'], data['port'], data['malicious'])
            for edge in data['edges']:
                self.add_edge(node, edge)

    def __iter__(self):
        return iter(self.nodes)
    
    def __str__(self):
        out = ""
        for node, edges in self.edges.items():
            if len(edges) == 0:
                out += f"{node}: None\n"
            else:
                out += f"{node}: {edges}\n"
        return out
    def __repr__(self):
        return self.__str__()
    def __getitem__(self, key):
        return self.nodes[key]
    




def create_graph():
    graph = Graph()
    return graph

