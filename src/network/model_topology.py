# creates a visual graph of the topology
import os
import random
import yaml
# visualization
import graph

import cv2

import numpy as np

experiment_yaml  = os.path.join('src','config', 'experiment.yaml')
experiment_params = yaml.safe_load(open(experiment_yaml))
topology = None
def load_topology():
    global topology
    topology = graph.create_graph()
    topology.load(os.path.join(os.getcwd(), 'src','config','topology.json'))
    print("Loaded topology")
    print(topology)
    return topology

def visualize_topology():
    global topology
    topology = load_topology()
    print("Visualizing topology")
    # create a blank image
    img = 255 * np.ones((1000, 1000, 3), np.uint8)
    # draw nodes
    user_locations = {}
    for user in topology:
        x = random.randint(0, 1000)
        y = random.randint(0, 1000)
        user_locations[user] = (x, y)
        cv2.circle(img, (x, y), 10, (0, 0, 0), -1)
        cv2.putText(img, str(user), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    # draw edges
    for user, edges in topology.edges.items():
        for edge in edges:
            x1, y1 = user_locations[str(user)]
            x2, y2 = user_locations[str(edge)]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 2)

    cv2.imshow('Topology', img)
    cv2.waitKey(0)
    

if __name__=='__main__':
    load_topology()
    visualize_topology()
