"""
WSN Network Simulator Engine.
Implements topology generation, graph construction, traffic generators,
discrete-event step execution, and holistic metric tracking.
"""

import math
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import networkx as nx

from .node import SensorNode
from .channel import WirelessChannel
from .packet import Packet, PacketType, DropReason


class WSNNetwork:
    FEATURE_NAMES = [
        "Residual_Energy_Ratio",
        "Energy_Depletion_Rate",
        "Buffer_Occupancy_Ratio",
        "Packet_Arrival_Rate",
        "Packet_Service_Rate",
        "Traffic_Intensity",
        "Packet_Loss_Rate",
        "Average_Queuing_Delay",
        "Average_Neighbor_RSSI",
        "Average_Neighbor_PRR",
        "Neighbor_Degree",
        "Distance_to_Sink_Ratio"
    ]

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.net_cfg = config.get("network", {})
        self.chan_cfg = config.get("channel", {})
        self.energy_cfg = config.get("energy", {})
        self.traffic_cfg = config.get("queue_and_traffic", {})
        self.label_cfg = config.get("labeling", {})

        self.num_nodes = int(self.net_cfg.get("num_nodes", 50))
        self.field_width = float(self.net_cfg.get("field_width", 200.0))
        self.field_height = float(self.net_cfg.get("field_height", 200.0))
        self.tx_range = float(self.net_cfg.get("tx_range", 50.0))
        self.max_field_dim = math.hypot(self.field_width, self.field_height)
        self.seed = int(self.net_cfg.get("seed", 42))

        self.rng = np.random.default_rng(self.seed)

        self.channel = WirelessChannel(
            path_loss_exponent=self.chan_cfg.get("path_loss_exponent", 2.5),
            ref_distance=self.chan_cfg.get("ref_distance", 1.0),
            ref_path_loss_db=self.chan_cfg.get("ref_path_loss_db", 40.0),
            shadowing_std=self.chan_cfg.get("shadowing_std", 2.0),
            noise_floor_dbm=self.chan_cfg.get("noise_floor_dbm", -95.0),
            tx_power_dbm=self.chan_cfg.get("tx_power_dbm", 0.0),
            sinr_threshold_db=self.chan_cfg.get("sinr_threshold_db", 4.0),
            seed=self.seed
        )

        self.nodes: Dict[int, SensorNode] = {}
        self.sink_id = self.num_nodes
        self.graph = nx.DiGraph()

        self.packet_counter = 0
        self.delivered_packets: List[Packet] = []
        self.all_generated_packets: List[Packet] = []
        self.current_time = 0.0

        self.fnd_time: Optional[float] = None
        self.hnd_time: Optional[float] = None
        self.lnd_time: Optional[float] = None
        self.total_route_changes = 0

        self._build_topology()
        self._build_links()

    def _build_topology(self) -> None:
        topo_type = self.net_cfg.get("topology_type", "random").lower()
        sink_pos_cfg = self.net_cfg.get("sink_pos", "center")

        if isinstance(sink_pos_cfg, list) and len(sink_pos_cfg) == 2:
            sink_x, sink_y = float(sink_pos_cfg[0]), float(sink_pos_cfg[1])
        elif sink_pos_cfg == "corner":
            sink_x, sink_y = self.field_width, self.field_height
        elif sink_pos_cfg == "edge":
            sink_x, sink_y = self.field_width, self.field_height / 2.0
        else:
            sink_x, sink_y = self.field_width / 2.0, self.field_height / 2.0

        positions: List[Tuple[float, float]] = []
        if topo_type == "grid":
            side = int(math.ceil(math.sqrt(self.num_nodes)))
            dx = self.field_width / (side + 1)
            dy = self.field_height / (side + 1)
            for i in range(side):
                for j in range(side):
                    if len(positions) < self.num_nodes:
                        jx = (self.rng.uniform(-0.1, 0.1)) * dx
                        jy = (self.rng.uniform(-0.1, 0.1)) * dy
                        positions.append(((i + 1) * dx + jx, (j + 1) * dy + jy))
        elif topo_type == "clustered":
            k_clusters = 3
            cluster_centers = [
                (self.rng.uniform(30, self.field_width - 30), self.rng.uniform(30, self.field_height - 30))
                for _ in range(k_clusters)
            ]
            for i in range(self.num_nodes):
                center = cluster_centers[i % k_clusters]
                px = float(np.clip(self.rng.normal(center[0], 20.0), 5.0, self.field_width - 5.0))
                py = float(np.clip(self.rng.normal(center[1], 20.0), 5.0, self.field_height - 5.0))
                positions.append((px, py))
        else:
            for _ in range(self.num_nodes):
                px = float(self.rng.uniform(5.0, self.field_width - 5.0))
                py = float(self.rng.uniform(5.0, self.field_height - 5.0))
                positions.append((px, py))

        is_hetero = self.energy_cfg.get("heterogeneous", False)
        e_init = float(self.energy_cfg.get("initial_energy_j", 5.0))
        e_min = float(self.energy_cfg.get("hetero_min_j", 2.0))
        e_max = float(self.energy_cfg.get("hetero_max_j", 5.0))

        energy_params = {
            "e_elec": self.energy_cfg.get("e_elec", 50e-9),
            "e_fs": self.energy_cfg.get("e_fs", 10e-12),
            "e_mp": self.energy_cfg.get("e_mp", 0.0013e-12),
            "d0": self.energy_cfg.get("d0", 87.0),
            "e_da": self.energy_cfg.get("e_da", 5e-9),
            "e_crit_ratio": self.label_cfg.get("energy_unhealthy_ratio", 0.15)
        }

        for i, (px, py) in enumerate(positions):
            node_e = float(self.rng.uniform(e_min, e_max)) if is_hetero else e_init
            self.nodes[i] = SensorNode(
                node_id=i,
                pos_x=px,
                pos_y=py,
                is_sink=False,
                initial_energy=node_e,
                queue_capacity=self.traffic_cfg.get("queue_capacity", 30),
                service_rate=self.traffic_cfg.get("service_rate_pkts_per_sec", 12.0),
                energy_model_params=energy_params
            )

        self.nodes[self.sink_id] = SensorNode(
            node_id=self.sink_id,
            pos_x=sink_x,
            pos_y=sink_y,
            is_sink=True,
            initial_energy=1e6,
            queue_capacity=1000,
            service_rate=1000.0,
            energy_model_params=energy_params
        )

    def _build_links(self) -> None:
        self.graph.clear()
        for node_id in self.nodes:
            self.graph.add_node(node_id)

        sink_node = self.nodes[self.sink_id]

        for u_id, u_node in self.nodes.items():
            u_node.dist_to_sink = u_node.distance_to(sink_node)
            u_node.neighbors.clear()

            for v_id, v_node in self.nodes.items():
                if u_id == v_id:
                    continue
                d = u_node.distance_to(v_node)
                if d <= self.tx_range:
                    prr = self.channel.compute_prr(d, apply_shadowing=False)
                    rssi = self.channel.compute_received_power_dbm(d, apply_shadowing=False)
                    etx = self.channel.compute_etx(d)

                    u_node.neighbors[v_id] = {
                        "distance": d,
                        "prr": prr,
                        "rssi": rssi,
                        "etx": etx
                    }
                    self.graph.add_edge(u_id, v_id, weight=d, prr=prr, etx=etx)

        for u_id, u_node in self.nodes.items():
            if u_id == self.sink_id:
                u_node.hop_count_to_sink = 0
            else:
                try:
                    path = nx.shortest_path(self.graph, source=u_id, target=self.sink_id)
                    u_node.hop_count_to_sink = len(path) - 1
                except nx.NetworkXNoPath:
                    u_node.hop_count_to_sink = 999

    def generate_traffic_step(self, dt: float) -> None:
        pattern = self.traffic_cfg.get("traffic_pattern", "poisson").lower()
        arr_rate = float(self.traffic_cfg.get("arrival_rate_pkts_per_sec", 2.0))
        pkt_size = int(self.traffic_cfg.get("packet_size_bits", 4000))
        source_ratio = float(self.traffic_cfg.get("source_ratio", 0.5))

        num_sources = max(1, int(self.num_nodes * source_ratio))
        sources = list(range(num_sources))

        burst_prob = float(self.traffic_cfg.get("burst_prob", 0.15))
        burst_rate = float(self.traffic_cfg.get("burst_peak_rate", 15.0))

        for src_id in sources:
            src_node = self.nodes[src_id]
            if not src_node.is_alive:
                continue

            current_rate = arr_rate
            if pattern == "bursty" and self.rng.uniform(0.0, 1.0) < burst_prob:
                current_rate = burst_rate

            expected_pkts = current_rate * dt
            num_pkts = self.rng.poisson(expected_pkts) if pattern != "cbr" else (1 if self.rng.uniform(0, 1) < expected_pkts else 0)

            for _ in range(num_pkts):
                pkt = Packet(
                    packet_id=self.packet_counter,
                    source_id=src_id,
                    dest_id=self.sink_id,
                    creation_time=self.current_time,
                    size_bits=pkt_size,
                    packet_type=PacketType.DATA
                )
                self.packet_counter += 1
                self.all_generated_packets.append(pkt)
                src_node.total_packets_generated += 1
                src_node.enqueue_packet(pkt, self.current_time)

    def step(self, dt: float, routing_function: Callable[[int, "WSNNetwork"], Optional[int]]) -> None:
        self.current_time += dt
        self.generate_traffic_step(dt)

        pkt_size = int(self.traffic_cfg.get("packet_size_bits", 4000))

        for u_id in range(self.num_nodes):
            u_node = self.nodes[u_id]
            if not u_node.is_alive or u_node.queue_length == 0:
                continue

            pkts_to_send = min(u_node.queue_length, max(1, int(u_node.service_rate_capacity * dt)))

            for _ in range(pkts_to_send):
                packet = u_node.dequeue_packet(self.current_time)
                if packet is None:
                    break

                if not packet.add_hop(u_id):
                    continue

                next_hop = routing_function(u_id, self)
                if next_hop is None or next_hop not in self.nodes:
                    packet.mark_dropped(u_id, DropReason.NO_ROUTE)
                    u_node.total_link_drops += 1
                    continue

                next_node = self.nodes[next_hop]
                if not next_node.is_alive:
                    packet.mark_dropped(u_id, DropReason.NEXT_HOP_DEAD)
                    u_node.total_link_drops += 1
                    continue

                dist = u_node.distance_to(next_node)
                u_node.energy.consume_tx(pkt_size, dist)
                if not next_node.is_sink:
                    next_node.energy.consume_rx(pkt_size)

                success, prr, rssi = self.channel.test_packet_transmission(dist, pkt_size)
                if not success:
                    packet.mark_dropped(u_id, DropReason.LINK_ERROR)
                    u_node.total_link_drops += 1
                    continue

                u_node.total_packets_forwarded += 1

                if next_hop == self.sink_id:
                    packet.mark_delivered(self.current_time)
                    self.delivered_packets.append(packet)
                    self.nodes[self.sink_id].total_packets_received_dest += 1
                else:
                    next_node.enqueue_packet(packet, self.current_time)

        dead_nodes = sum(1 for i in range(self.num_nodes) if not self.nodes[i].is_alive)
        if dead_nodes >= 1 and self.fnd_time is None:
            self.fnd_time = self.current_time
        if dead_nodes >= (self.num_nodes // 2) and self.hnd_time is None:
            self.hnd_time = self.current_time
        if dead_nodes >= self.num_nodes and self.lnd_time is None:
            self.lnd_time = self.current_time

    def collect_dataset_sample(self, epoch_duration: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        X_list = []
        y_list = []
        for i in range(self.num_nodes):
            node = self.nodes[i]
            feat = node.extract_features(self.current_time, epoch_duration, self.max_field_dim)
            label = node.get_ground_truth_label(
                queue_congested_ratio=self.label_cfg.get("queue_congested_ratio", 0.70),
                traffic_intensity_threshold=self.label_cfg.get("traffic_intensity_threshold", 0.90),
                energy_unhealthy_ratio=self.label_cfg.get("energy_unhealthy_ratio", 0.15),
                plr_unhealthy_threshold=self.label_cfg.get("plr_unhealthy_threshold", 0.40)
            )
            X_list.append(feat)
            y_list.append(label)
            node.reset_epoch_counters()

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

    def compute_metrics(self) -> Dict[str, Any]:
        total_gen = len(self.all_generated_packets)
        total_deliv = len(self.delivered_packets)
        pdr = (total_deliv / max(1, total_gen)) * 100.0

        delays = [p.end_to_end_delay for p in self.delivered_packets if p.end_to_end_delay is not None]
        avg_delay = float(np.mean(delays)) if delays else 0.0
        p95_delay = float(np.percentile(delays, 95)) if delays else 0.0

        hops = [p.hop_count for p in self.delivered_packets]
        avg_hop = float(np.mean(hops)) if hops else 0.0

        total_energy = sum(self.nodes[i].energy.total_consumed for i in range(self.num_nodes))
        pkt_size = int(self.traffic_cfg.get("packet_size_bits", 4000))
        delivered_bits = total_deliv * pkt_size
        energy_eff = delivered_bits / max(1e-6, total_energy)

        queue_drops = sum(1 for p in self.all_generated_packets if p.drop_reason == DropReason.QUEUE_OVERFLOW)
        link_drops = sum(1 for p in self.all_generated_packets if p.drop_reason in (DropReason.LINK_ERROR, DropReason.NEXT_HOP_DEAD))
        ttl_drops = sum(1 for p in self.all_generated_packets if p.drop_reason == DropReason.TTL_EXPIRED)

        dead_count = sum(1 for i in range(self.num_nodes) if not self.nodes[i].is_alive)

        return {
            "pdr_percent": pdr,
            "total_generated": total_gen,
            "total_delivered": total_deliv,
            "total_dropped": total_gen - total_deliv,
            "avg_delay_s": avg_delay,
            "p95_delay_s": p95_delay,
            "avg_hop_count": avg_hop,
            "total_energy_j": total_energy,
            "energy_efficiency_bits_per_j": energy_eff,
            "fnd_time_s": self.fnd_time if self.fnd_time is not None else float("inf"),
            "hnd_time_s": self.hnd_time if self.hnd_time is not None else float("inf"),
            "lnd_time_s": self.lnd_time if self.lnd_time is not None else float("inf"),
            "dead_node_count": dead_count,
            "queue_drops": queue_drops,
            "link_drops": link_drops,
            "ttl_drops": ttl_drops,
            "route_changes": self.total_route_changes
        }
