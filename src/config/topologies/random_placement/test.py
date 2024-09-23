import json
import os

path = '/mnt/home/bhatta70/Documents/DFL-Secure-Aggregation/src/config/topologies/random_placement'
for filename in os.listdir(path):
    if not filename.endswith('.json'):
        continue
    with open(os.path.join(path, filename), 'r') as f:
        net_type = filename.split('_')[0]
        if net_type != 'small':
            continue
        byz_prop = float(filename.split('_')[-1].strip('.json'))

        data = json.load(f)
        nodes = data['nodes']
        num_malicioius = float(sum([1 for node in nodes if node['malicious']]))
        
        
        adj_dict = {}
        links = data['links']
        for link in links:
            if link['source'] not in adj_dict:
                adj_dict[link['source']] = set()
            if link['target'] not in adj_dict:
                adj_dict[link['target']] = set()
            adj_dict[link['source']].add(link['target'])
            adj_dict[link['target']].add(link['source'])
        avg_mal_deg = 0
        for node in adj_dict:
            if nodes[node]['malicious']:
                continue
            adj_dict[node] = list(adj_dict[node])
            num_byz = sum([1 for neighbor in adj_dict[node] if nodes[neighbor]['malicious']])
            if num_byz > 0:
                avg_mal_deg += (num_byz / len(adj_dict[node]))
        avg_mal_deg /= (len(adj_dict) - num_malicioius)


        avg_deg = sum([len(adj_dict[node]) for node in adj_dict]) / len(adj_dict)
        print("\nNet type: ", net_type, " Byz prop: ", byz_prop, "Actual malicious prop ", num_malicioius / len(nodes), "Avg degree: ", avg_deg, "Avg malicious degree: ", avg_mal_deg)

