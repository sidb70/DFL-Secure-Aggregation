import random
import json
import bisect
# seed
random.seed(42)

class UserGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
    def add_user(self, user_num, ip, port, malicious=False):
        if user_num in self.nodes:
            return
        self.nodes[user_num] = {'ip': ip, 'port': port, 'malicious': malicious}
        if user_num not in self.edges:
            self.edges[user_num] = set()

    def make_connections(self, p=1.0, directed=False):
        '''
        Make connections between users with probability p
        :param p: probability of connection
        :param directed: whether the graph is directed
        '''
        for user1 in self.nodes:
            for user2 in self.nodes:
                if user1 == user2:
                    continue
                if random.random() < p:
                    self.add_edge(user1, user2, directed=directed)
                    if not directed:
                        self.add_edge(user2, user1, directed=directed)
                else:
                    self.remove_edge(user1, user2, directed=directed)
                    if not directed:
                        self.remove_edge(user2, user1, directed=directed)
    def add_edge(self, user1, user2, directed=False):
        '''
        Add an edge between two users
        :param user1: user 1
        :param user2: user 2
        :param directed: whether the graph is directed
        '''
        if user1 in self.edges and user2 in self.edges[user1]:
            return
        if user1 not in self.edges:
            self.edges[user1] = set()
        if user2 not in self.edges:
            self.edges[user2] = set()
        self.edges[user1].add(user2)
        if not directed:
            self.edges[user2].add(user1)
    def remove_edge(self, user1, user2, directed=False):
        '''
        Remove an edge between two users
        :param user1: user 1
        :param user2: user 2
        :param directed: whether the graph is directed
        '''
        if user1 in self.edges and user2 in self.edges[user1]:
            self.edges[user1].remove(user2)
        if not directed:
            if user2 in self.edges and user1 in self.edges[user2]:
                self.edges[user2].remove(user1)
    def get_neighbors(self, user):
        '''
        Get the neighbors of a user
        :param user: user
        :return: set of neighbors
        '''
        if user in self.edges:
            return self.edges[user]
        return set()

    def create_random_graph(self, num_nodes, edge_density, malicious_nodes):
        '''
        Create a random graph
        num_nodes: number of nodes in the graph
        edge_density: probability of connection between nodes
        '''
        for i in range(num_nodes):
            is_malicious = i in malicious_nodes
            self.add_user(i, f'localhost', 50000+i, malicious=is_malicious)
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
            self.add_user(i, f'localhost', 50000+i, malicious=is_malicious)
        self.make_connections(p=1.0)
        for i in range(m0, num_nodes):
            # choose m nodes to connect to
            choices = [[node]*self.user_degree(node) for node in self.nodes]
            choices = [item for sublist in choices for item in sublist]

            is_malicious = i in malicious_nodes
            self.add_user(i, f'localhost', 50000+i, malicious=i in malicious_nodes)

            while self.user_degree(i) < m: # choose m neighbors
                neighbor = random.choice(choices)
                if neighbor not in self.user_edges(i):
                    self.add_edge(i, neighbor)

    def user_degree(self, user):
        '''
        Get the degree of a user
        :param user: user
        :return: degree of the user
        '''
        return len(self.edges[user])
    def user_edges(self, user):
        ''' 
        Get the edges of a user
        :param user: user
        :return: set of edges
        '''
        return self.edges[user]
    def to_dict(self):
        ''' 
        Convert the graph to a dictionary
        :return: dictionary representation of the graph
        '''
        graph_dict = {}
        for user in self.nodes.keys():
            graph_dict[user] = {
                "ip": self.nodes[user]['ip'],
                "port": self.nodes[user]['port'],
                "edges": list(self.get_neighbors(user)),
                "malicious": self.nodes[user]['malicious']
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
        for user, data in graph_dict.items():
            self.add_user(user, data['ip'], data['port'], data['malicious'])
            for edge in data['edges']:
                self.add_edge(user, edge)

    def __iter__(self):
        return iter(self.nodes)
    
    def __str__(self):
        out = ""
        for user, edges in self.edges.items():
            if len(edges) == 0:
                out += f"{user}: None\n"
            else:
                out += f"{user}: {edges}\n"
        return out
    def __repr__(self):
        return self.__str__()
    




def create_graph():
    graph = UserGraph()
    return graph

