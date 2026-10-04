"""
WSN Interactive Web Dashboard & Topology Visualizer Backend.
Launches a lightweight local server (zero extra dependencies) that serves
the real-time interactive 2D canvas, ML classification engine, and Dueling DDQN router.

Run with:
    python V2/visualizer.py (or python visualizer.py inside V2/)
Access via browser at:
    http://localhost:8080
"""

import sys
import os
import json
import time
import urllib.parse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import joblib
import numpy as np

# Ensure root directory is in sys.path
V2_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(V2_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from V2.simulator.network import WSNNetwork
from V2.simulator.packet import Packet
from V2.preprocessing import FeaturePreprocessor
from V2.routing.rl_agent import DuelingDDQNAgent
from V2.routing.rl_environment import WSNRoutingRLEnv


class WSNVisualizerApp:
    def __init__(self):
        self.config_path = os.path.join(V2_DIR, "configs", "default_config.json")
        with open(self.config_path, "r") as f:
            self.cfg = json.load(f)

        self.cfg["network"]["num_nodes"] = 35
        self.cfg["network"]["seed"] = 42
        self.cfg["queue_and_traffic"]["traffic_pattern"] = "bursty"
        self.cfg["queue_and_traffic"]["arrival_rate_pkts_per_sec"] = 4.0

        # Load ML (1D-CNN) & RL (Dueling Double DQN) Models
        import torch
        from V2.models.cnn_1d import TemporalCNN1D

        scaler_path = os.path.join(V2_DIR, "preprocessing", "scaler.joblib")
        cnn_path = os.path.join(V2_DIR, "results", "checkpoints", "cnn_1d.pt")
        ddqn_path = os.path.join(V2_DIR, "results", "checkpoints", "dueling_ddqn_router.pt")

        self.scaler = FeaturePreprocessor().load(scaler_path)
        self.cnn_model = TemporalCNN1D(in_channels=12, seq_len=5, num_classes=3)
        self.cnn_model.load_state_dict(torch.load(cnn_path, map_location="cpu", weights_only=False))
        self.cnn_model.eval()

        self.ddqn_agent = DuelingDDQNAgent(state_dim=44, action_dim=8)
        self.ddqn_agent.load(ddqn_path)

        self.reset()

    def reset(self):
        self.network = WSNNetwork(self.cfg)
        self.rl_env = WSNRoutingRLEnv(self.network, ml_model=self.cnn_model, preprocessor=self.scaler)

        # Warm up slightly
        for _ in range(20):
            self.network.step(dt=0.2, routing_function=lambda u, net: min(net.nodes[u].neighbors, key=lambda v: net.nodes[v].dist_to_sink) if net.nodes[u].neighbors else None)

        self.inject_congestion()

    def classify_all(self):
        features, _ = self.network.collect_dataset_sample()
        scaled = self.scaler.transform(features)
        self.probs = self.cnn_model.predict_proba(scaled)
        self.preds = self.cnn_model.predict(scaled)

    def inject_congestion(self):
        from V2.simulator.packet import Packet
        candidates = [n for n in range(self.network.num_nodes) if n != self.network.sink_id and len(self.network.nodes[n].neighbors) > 0]
        
        # 3 Congested nodes (high buffer bloat, queue occupancy > 85%)
        congested_candidates = [n for n in candidates if 35 < self.network.nodes[n].dist_to_sink < 90][:3]
        for c_id in congested_candidates:
            c_node = self.network.nodes[c_id]
            for k in range(int(c_node.queue_capacity * 0.88)):
                c_node.packet_queue.append(Packet(packet_id=8000 + k, source_id=c_id, dest_id=self.network.sink_id, creation_time=0.0))
            c_node.epoch_arrivals = 30
            c_node.epoch_serviced = 5

        # 4 Unhealthy nodes (critical battery depletion E_res <= 0.20J and high packet loss rate)
        unhealthy_candidates = [n for n in candidates if n not in congested_candidates and self.network.nodes[n].dist_to_sink > 30][:4]
        for u_id in unhealthy_candidates:
            u_node = self.network.nodes[u_id]
            u_node.energy.residual_energy = 0.20  # ~4% battery remaining
            u_node.epoch_drops = 25
            u_node.epoch_arrivals = 5

        self.classify_all()

    def find_route(self, source_id: int):
        node_risks = self.rl_env.get_node_ml_risks()
        curr = source_id
        path = [curr]
        avoided_congested = []
        avoided_unhealthy = []

        t0 = time.perf_counter()
        for _ in range(15):
            if curr == self.network.sink_id:
                break
            state, valid_neighbors, action_mask = self.rl_env.get_state(curr, node_risks)
            if not valid_neighbors:
                break

            action_idx = self.ddqn_agent.select_action(state, action_mask, evaluate=True)
            next_id = valid_neighbors[action_idx] if action_idx < len(valid_neighbors) else valid_neighbors[0]

            for v in valid_neighbors:
                if v < self.network.num_nodes:
                    if self.preds[v] == 1 and v not in path and v != next_id:
                        avoided_congested.append(v)
                    elif self.preds[v] == 2 and v not in path and v != next_id:
                        avoided_unhealthy.append(v)

            if next_id in path:
                alt = [v for v in valid_neighbors if v not in path]
                if alt:
                    next_id = alt[0]
                else:
                    break

            path.append(next_id)
            curr = next_id
            if curr == self.network.sink_id:
                break

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "path": path,
            "latency_ms": latency_ms,
            "hops": len(path) - 1,
            "avoided": sorted(list(set(avoided_congested))),
            "avoided_unhealthy": sorted(list(set(avoided_unhealthy)))
        }

    def get_state_json(self):
        nodes_list = []
        for i in range(self.network.num_nodes + 1):
            node = self.network.nodes[i]
            if node.is_sink:
                cls_idx = 0
                p = [1.0, 0.0, 0.0]
                e_res = 100.0
                e_ratio = 1.0
                q_len = 0
                q_cap = 100
                q_occ = 0.0
            else:
                cls_idx = int(self.preds[i])
                p = [float(x) for x in self.probs[i]]
                e_res = float(node.energy.residual_energy)
                e_ratio = float(node.energy.energy_ratio)
                q_len = int(node.queue_length)
                q_cap = int(node.queue_capacity)
                q_occ = float(node.queue_occupancy)

            nodes_list.append({
                "id": node.node_id,
                "x": float(node.x),
                "y": float(node.y),
                "is_sink": bool(node.is_sink),
                "residual_energy": e_res,
                "energy_ratio": e_ratio,
                "queue_length": q_len,
                "queue_capacity": q_cap,
                "queue_occupancy": q_occ,
                "arrival_rate": float(getattr(node, "epoch_arrivals", 3.5)),
                "dist_to_sink": float(node.dist_to_sink),
                "neighbors_count": len(node.neighbors),
                "classification": cls_idx,
                "probs": p
            })

        links_set = set()
        for u in range(self.network.num_nodes):
            for v in self.network.nodes[u].neighbors:
                links_set.add(tuple(sorted((u, v))))

        return {
            "field_width": float(self.network.field_width),
            "field_height": float(self.network.field_height),
            "tx_range": float(self.network.tx_range),
            "sink_id": self.network.sink_id,
            "nodes": nodes_list,
            "links": [list(link) for link in sorted(list(links_set))]
        }


# Singleton App Instance
app_instance = WSNVisualizerApp()


class VisualizerRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            html_file = os.path.join(V2_DIR, "visualizer", "index.html")
            with open(html_file, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif parsed.path == "/api/state":
            self._send_json(app_instance.get_state_json())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/classify":
            app_instance.classify_all()
            self._send_json(app_instance.get_state_json())
        elif parsed.path == "/api/congest":
            app_instance.inject_congestion()
            self._send_json(app_instance.get_state_json())
        elif parsed.path == "/api/reset":
            app_instance.reset()
            self._send_json(app_instance.get_state_json())
        elif parsed.path == "/api/route":
            qs = urllib.parse.parse_qs(parsed.query)
            source_id = int(qs.get("source", [11])[0])
            route_res = app_instance.find_route(source_id)
            self._send_json(route_res)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress noisy HTTP request logging in terminal
        pass


def run_visualizer_server(port: int = 8080):
    for p in range(port, port + 10):
        try:
            server = HTTPServer(("0.0.0.0", p), VisualizerRequestHandler)
            print("=" * 80)
            print("      WSN AUTONOMOUS ROUTING: INTERACTIVE TOPOLOGY VISUALIZER")
            print("=" * 80)
            print(f"[+] Server running successfully at: http://localhost:{p}")
            print(f"[+] Opening in browser... (Press Ctrl+C to stop)")
            print("=" * 80)
            try:
                webbrowser.open(f"http://localhost:{p}")
            except Exception:
                pass
            server.serve_forever()
            break
        except OSError:
            continue


if __name__ == "__main__":
    run_visualizer_server()
