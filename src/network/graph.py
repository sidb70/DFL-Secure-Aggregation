import random
import json
class UserGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
    def add_user(self, user_num, ip, port, malicious=False):
        if user_num in self.nodes:
            return
        self.nodes[user_num] = {'ip': ip, 'port': port, 'malicious': malicious}
        self.edges[user_num] = set()

    def make_connections(self, p=1.0, directed=False):
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
        if user1 in self.edges and user2 in self.edges[user1]:
            return
        self.edges[user1].add(user2)
        if not directed:
            self.edges[user2].add(user1)
    def remove_edge(self, user1, user2, directed=False):
        if user1 in self.edges and user2 in self.edges[user1]:
            self.edges[user1].remove(user2)
        if not directed:
            if user2 in self.edges and user1 in self.edges[user2]:
                self.edges[user2].remove(user1)
    def get_neighbors(self, user):
        if user in self.edges:
            return self.edges[user]
        return set()
    def to_json(self):
        graph_dict = {}
        for user in self.nodes.keys():
            graph_dict[user] = {
                "ip": self.nodes[user]['ip'],
                "port": self.nodes[user]['port'],
                "edges": list(self.get_neighbors(user)),
                "malicious": self.nodes[user]['malicious']
            }
        return json.dumps(graph_dict)
    def save(self, filename):
        with open(filename, 'w') as f:
            f.write(self.to_json())
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




if __name__ == '__main__':
    graph = create_graph()
    for i in range(10):
        graph.add_user(i, f'localhost', 6000+i)
    graph.make_connections(p=.6)
    
    print()
    print(graph)

    print()
    print(graph.to_json())