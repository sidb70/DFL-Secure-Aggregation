from network import graph
import random

random.seed(0)

def main():
    num_nodes = 128
    proportions = [0, .15, .3, .45, .6]

    small_world_k = 4
    small_world_beta = .2

    scale_free_m = 10
    
    # randomly placed malicious nodes
    for p in proportions:
        num_malicious = int(num_nodes * p)
        malicious_nodes = random.sample(range(num_nodes), num_malicious)
        
        small_world = graph.Topology()
        small_world.create_small_world_graph(num_nodes, small_world_k, small_world_beta, malicious_nodes)
        small_world.save(f'/mnt/home/bhatta70/Documents/DFL-Secure-Aggregation/src/config/topologies/random_placement/small_world_{p}.json')
        
        scale_free = graph.Topology()
        scale_free.create_scale_free_graph(num_nodes, scale_free_m, malicious_nodes)
        scale_free.save(f'/mnt/home/bhatta70/Documents/DFL-Secure-Aggregation/src/config/topologies/random_placement/scale_free_{p}.json')

        random_graph = graph.Topology()
        random_graph.create_random_graph(num_nodes, edge_density=.06, malicious_nodes=malicious_nodes)
        random_graph.save(f'/mnt/home/bhatta70/Documents/DFL-Secure-Aggregation/src/config/topologies/random_placement/random_{p}.json')

    # strategic placement of malicious nodes
    small_world_strategic_betas = [0,.05, .1, .15]
    scale_free_strategic_b = [0, .05, .15, .25]
    random_strategic_b = [0, .05, .15, .25]

    for b in small_world_strategic_betas:
        small_world = graph.Topology()
        top = graph.Topology()
        top.create_small_world_graph(num_nodes, small_world_k, b, malicious_nodes=[])
        rewires = set()
        malicious=[]
        for node in top.nodes:
            neighbors = top.get_neighbors(node)
            for neighbor in neighbors:
                edge = (node, neighbor) if node < neighbor else (neighbor, node)
                if abs(node%num_nodes - neighbor%num_nodes) > small_world_k and edge not in rewires:
                    rewires.add(edge)
                    malicious.append(node)
                    break
        top.set_malicous(malicious)
        top.save(f'/mnt/home/bhatta70/Documents/DFL-Secure-Aggregation/src/config/topologies/strategic_placement/small_world_strategic_{b}.json')
        fig, ax = top.draw()
        fig.savefig(f'/mnt/home/bhatta70/Documents/DFL-Secure-Aggregation/src/config/topologies/figs/small_world_strategic_{b}.png')

    for b in scale_free_strategic_b:
        top = graph.Topology()
        top.create_scale_free_graph(num_nodes, scale_free_m, malicious_nodes=[])

        num_malicious = int(num_nodes * b)
        sorted_nodes_by_deg = sorted(top.nodes, key=lambda x: top.degree(x), reverse=True)
        malicious_nodes = sorted_nodes_by_deg[:num_malicious]
        top.set_malicous(malicious_nodes)

        top.save(f'/mnt/home/bhatta70/Documents/DFL-Secure-Aggregation/src/config/topologies/strategic_placement/scale_free_strategic_{b}.json')
        fig, ax = top.draw()
        fig.savefig(f'/mnt/home/bhatta70/Documents/DFL-Secure-Aggregation/src/config/topologies/figs/scale_free_strategic_{b}.png')
    for b in random_strategic_b:
        top = graph.Topology()
        top.create_random_graph(num_nodes, edge_density=.06, malicious_nodes=[])

        num_malicious = int(num_nodes * b)
        sorted_nodes_by_deg = sorted(top.nodes, key=lambda x: top.degree(x), reverse=True)
        malicious_nodes = sorted_nodes_by_deg[:num_malicious]
        top.set_malicous(malicious_nodes)
        top.save(f'/mnt/home/bhatta70/Documents/DFL-Secure-Aggregation/src/config/topologies/strategic_placement/random_strategic_{b}.json')
        fig, ax = top.draw()
        fig.savefig(f'/mnt/home/bhatta70/Documents/DFL-Secure-Aggregation/src/config/topologies/figs/random_strategic_{b}.png')

if __name__ == '__main__':
    main()