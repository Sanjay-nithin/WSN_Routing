import os, json, copy
import pandas as pd
from V2.simulator.network import WSNNetwork
from V2.datasets.labeling import GroundTruthLabeler

def default_routing(u_id, net):
    node = net.nodes[u_id]
    alive = [n for n in node.neighbors if net.nodes[n].is_alive]
    if not alive:
        return None
    return min(alive, key=lambda n: net.nodes[n].dist_to_sink)

class DatasetGenerator:
    def __init__(self, base_config_path="V2/configs/default_config.json", output_dir="V2/datasets"):
        self.base_config = json.load(open(base_config_path))
        self.output_dir = output_dir
        self.processed_dir = os.path.join(output_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
