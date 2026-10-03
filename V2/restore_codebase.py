import os

files = {}

files["V2/simulator/packet.py"] = '''"""
WSN Packet Definition and Telemetry Dataclass.
Models packet headers, lifecycle, hop tracking, and drop reasons.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class PacketType(Enum):
    DATA = "DATA"
    CONTROL = "CONTROL"
    BEACON = "BEACON"


class DropReason(Enum):
    NONE = "NONE"
    QUEUE_OVERFLOW = "QUEUE_OVERFLOW"
    TTL_EXPIRED = "TTL_EXPIRED"
    LINK_ERROR = "LINK_ERROR"
    NEXT_HOP_DEAD = "NEXT_HOP_DEAD"
    NO_ROUTE = "NO_ROUTE"


@dataclass
class Packet:
    packet_id: int
    source_id: int
    dest_id: int
    creation_time: float
    size_bits: int = 4000  # 500 bytes default
    packet_type: PacketType = PacketType.DATA
    hop_count: int = 0
    max_hops: int = 30
    path: List[int] = field(default_factory=list)
    delivery_time: Optional[float] = None
    is_delivered: bool = False
    is_dropped: bool = False
    drop_reason: DropReason = DropReason.NONE
    drop_node_id: Optional[int] = None
    queuing_delay_accum: float = 0.0

    def add_hop(self, current_node_id: int) -> bool:
        self.path.append(current_node_id)
        self.hop_count += 1
        if self.hop_count > self.max_hops:
            self.is_dropped = True
            self.drop_reason = DropReason.TTL_EXPIRED
            self.drop_node_id = current_node_id
            return False
        return True

    def mark_delivered(self, arrival_time: float) -> None:
        self.is_delivered = True
        self.delivery_time = arrival_time

    def mark_dropped(self, node_id: int, reason: DropReason) -> None:
        self.is_dropped = True
        self.drop_reason = reason
        self.drop_node_id = node_id

    @property
    def end_to_end_delay(self) -> Optional[float]:
        if self.is_delivered and self.delivery_time is not None:
            return self.delivery_time - self.creation_time
        return None
'''

files["V2/simulator/energy.py"] = '''"""
First-Order Radio Energy Model for Wireless Sensor Networks (Heinzelman et al., 2000/2002).
Tracks battery depletion, transmission/reception dissipation, and consumption rates.
"""

import math
from typing import Dict, Any


class FirstOrderRadioModel:
    def __init__(
        self,
        initial_energy: float = 5.0,
        e_elec: float = 50e-9,      # 50 nJ/bit
        e_fs: float = 10e-12,       # 10 pJ/bit/m^2
        e_mp: float = 0.0013e-12,   # 0.0013 pJ/bit/m^4
        d0: float = 87.0,           # crossover threshold (m)
        e_da: float = 5e-9,         # 5 nJ/bit aggregation
        e_crit_ratio: float = 0.15  # Unhealthy threshold (15% remaining)
    ):
        self.initial_energy = float(initial_energy)
        self.residual_energy = float(initial_energy)
        self.e_elec = float(e_elec)
        self.e_fs = float(e_fs)
        self.e_mp = float(e_mp)
        self.d0 = float(d0) if d0 > 0 else math.sqrt(self.e_fs / self.e_mp)
        self.e_da = float(e_da)
        self.e_crit_ratio = float(e_crit_ratio)

        self.total_tx_energy = 0.0
        self.total_rx_energy = 0.0
        self.total_proc_energy = 0.0
        self.last_energy_snapshot = float(initial_energy)
        self.last_time_snapshot = 0.0
        self.consumption_rate = 0.0

    @property
    def is_alive(self) -> bool:
        return self.residual_energy > 0.0

    @property
    def is_critically_depleted(self) -> bool:
        if self.initial_energy <= 0:
            return True
        return (self.residual_energy / self.initial_energy) <= self.e_crit_ratio

    @property
    def energy_ratio(self) -> float:
        if self.initial_energy <= 0:
            return 0.0
        return max(0.0, min(1.0, self.residual_energy / self.initial_energy))

    @property
    def total_consumed(self) -> float:
        return self.initial_energy - self.residual_energy

    def calculate_tx_energy(self, num_bits: int, distance: float) -> float:
        if distance < self.d0:
            e_amp = self.e_fs * (distance ** 2)
        else:
            e_amp = self.e_mp * (distance ** 4)
        return num_bits * (self.e_elec + e_amp)

    def calculate_rx_energy(self, num_bits: int) -> float:
        return num_bits * self.e_elec

    def consume_tx(self, num_bits: int, distance: float) -> bool:
        cost = self.calculate_tx_energy(num_bits, distance)
        self.residual_energy -= cost
        self.total_tx_energy += cost
        if self.residual_energy <= 0.0:
            self.residual_energy = 0.0
            return False
        return True

    def consume_rx(self, num_bits: int) -> bool:
        cost = self.calculate_rx_energy(num_bits)
        self.residual_energy -= cost
        self.total_rx_energy += cost
        if self.residual_energy <= 0.0:
            self.residual_energy = 0.0
            return False
        return True

    def update_consumption_rate(self, current_time: float) -> float:
        dt = current_time - self.last_time_snapshot
        if dt > 0:
            dE = self.last_energy_snapshot - self.residual_energy
            instant_rate = max(0.0, dE / dt)
            self.consumption_rate = 0.3 * instant_rate + 0.7 * self.consumption_rate
            self.last_energy_snapshot = self.residual_energy
            self.last_time_snapshot = current_time
        return self.consumption_rate
'''

files["V2/simulator/channel.py"] = '''"""
Wireless Channel and Physical Layer Propagation Model.
Implements Log-Distance Path Loss, Log-Normal Shadowing, SINR, PRR, and ETX.
"""

import math
import numpy as np
from typing import Tuple


class WirelessChannel:
    def __init__(
        self,
        path_loss_exponent: float = 2.5,
        ref_distance: float = 1.0,
        ref_path_loss_db: float = 40.0,
        shadowing_std: float = 2.0,
        noise_floor_dbm: float = -95.0,
        tx_power_dbm: float = 0.0,
        sinr_threshold_db: float = 4.0,
        seed: int = 42
    ):
        self.n = float(path_loss_exponent)
        self.d0 = float(ref_distance)
        self.pl_d0 = float(ref_path_loss_db)
        self.sigma = float(shadowing_std)
        self.noise_floor_dbm = float(noise_floor_dbm)
        self.tx_power_dbm = float(tx_power_dbm)
        self.sinr_threshold_db = float(sinr_threshold_db)
        self.rng = np.random.default_rng(seed)

    def calculate_path_loss(self, distance: float, apply_shadowing: bool = True) -> float:
        d = max(distance, self.d0)
        pl = self.pl_d0 + 10.0 * self.n * math.log10(d / self.d0)
        if apply_shadowing and self.sigma > 0:
            shadowing = float(self.rng.normal(0.0, self.sigma))
            pl += shadowing
        return pl

    def compute_received_power_dbm(self, distance: float, apply_shadowing: bool = True) -> float:
        pl = self.calculate_path_loss(distance, apply_shadowing)
        return self.tx_power_dbm - pl

    def compute_snr_db(self, distance: float, apply_shadowing: bool = True) -> float:
        p_rx = self.compute_received_power_dbm(distance, apply_shadowing)
        return p_rx - self.noise_floor_dbm

    def compute_prr(self, distance: float, packet_size_bits: int = 4000, apply_shadowing: bool = True) -> float:
        snr_db = self.compute_snr_db(distance, apply_shadowing)
        if snr_db < self.sinr_threshold_db - 6.0:
            return 0.0
        if snr_db > 25.0:
            return 1.0

        snr_linear = 10.0 ** (snr_db / 10.0)
        ber = 0.5 * math.erfc(math.sqrt(max(0.0, snr_linear)))
        ber = min(0.5, max(1e-9, ber))

        prr = (1.0 - ber) ** packet_size_bits
        return float(np.clip(prr, 0.0, 1.0))

    def compute_etx(self, distance: float, packet_size_bits: int = 4000) -> float:
        prr = self.compute_prr(distance, packet_size_bits, apply_shadowing=False)
        prr_bidir = max(1e-4, prr * prr)
        return float(min(20.0, 1.0 / prr_bidir))

    def test_packet_transmission(self, distance: float, packet_size_bits: int = 4000) -> Tuple[bool, float, float]:
        rssi = self.compute_received_power_dbm(distance, apply_shadowing=True)
        prr = self.compute_prr(distance, packet_size_bits, apply_shadowing=True)
        success = (self.rng.uniform(0.0, 1.0) < prr)
        return success, prr, rssi
'''

files["V2/simulator/node.py"] = '''"""
WSN Sensor Node Implementation.
Manages node lifecycle, FIFO packet queuing, MAC-level counters, and telemetry feature extraction.
"""

import collections
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .energy import FirstOrderRadioModel
from .packet import Packet, DropReason


class SensorNode:
    def __init__(
        self,
        node_id: int,
        pos_x: float,
        pos_y: float,
        is_sink: bool = False,
        initial_energy: float = 5.0,
        queue_capacity: int = 30,
        service_rate: float = 12.0,
        energy_model_params: Optional[Dict[str, Any]] = None
    ):
        self.node_id = int(node_id)
        self.pos_x = float(pos_x)
        self.pos_y = float(pos_y)
        self.is_sink = bool(is_sink)
        self.queue_capacity = int(queue_capacity)
        self.service_rate_capacity = float(service_rate)
        self.hardware_failed = False

        params = energy_model_params or {}
        self.energy = FirstOrderRadioModel(
            initial_energy=initial_energy,
            **params
        )

        self.packet_queue = collections.deque(maxlen=self.queue_capacity)
        self.total_packets_generated = 0
        self.total_packets_forwarded = 0
        self.total_packets_received_dest = 0
        self.total_queue_drops = 0
        self.total_link_drops = 0

        self.epoch_arrivals = 0
        self.epoch_serviced = 0
        self.epoch_drops = 0
        self.epoch_delays: List[float] = []
        self.recent_delays = collections.deque(maxlen=50)

        self.hop_count_to_sink = 999
        self.dist_to_sink = 0.0
        self.neighbors: Dict[int, Dict[str, float]] = {}

    @property
    def is_alive(self) -> bool:
        if self.hardware_failed:
            return False
        if self.is_sink:
            return True
        return self.energy.is_alive

    @property
    def queue_occupancy(self) -> float:
        return len(self.packet_queue) / max(1, self.queue_capacity)

    @property
    def queue_length(self) -> int:
        return len(self.packet_queue)

    def enqueue_packet(self, packet: Packet, current_time: float) -> bool:
        if not self.is_alive:
            packet.mark_dropped(self.node_id, DropReason.NEXT_HOP_DEAD)
            self.total_link_drops += 1
            return False

        self.epoch_arrivals += 1
        if len(self.packet_queue) >= self.queue_capacity:
            packet.mark_dropped(self.node_id, DropReason.QUEUE_OVERFLOW)
            self.total_queue_drops += 1
            self.epoch_drops += 1
            return False

        self.packet_queue.append((packet, current_time))
        return True

    def dequeue_packet(self, current_time: float) -> Optional[Packet]:
        if not self.packet_queue:
            return None
        packet, enqueue_time = self.packet_queue.popleft()
        delay = current_time - enqueue_time
        packet.queuing_delay_accum += delay
        self.epoch_delays.append(delay)
        self.recent_delays.append(delay)
        self.epoch_serviced += 1
        return packet

    def reset_epoch_counters(self) -> None:
        self.epoch_arrivals = 0
        self.epoch_serviced = 0
        self.epoch_drops = 0
        self.epoch_delays.clear()

    def extract_features(
        self,
        current_time: float,
        epoch_duration: float = 1.0,
        max_field_dim: float = 282.84
    ) -> np.ndarray:
        e_res = self.energy.energy_ratio if not self.is_sink else 1.0
        e_rate = self.energy.update_consumption_rate(current_time) if not self.is_sink else 0.0
        q_occ = self.queue_occupancy
        arr_rate = self.epoch_arrivals / max(0.1, epoch_duration)
        srv_rate = self.epoch_serviced / max(0.1, epoch_duration)
        rho = arr_rate / max(0.5, srv_rate)

        total_epoch_pkts = self.epoch_arrivals + self.epoch_drops
        plr = (self.epoch_drops / total_epoch_pkts) if total_epoch_pkts > 0 else 0.0

        avg_delay_s = float(np.mean(self.epoch_delays)) if self.epoch_delays else (
            float(np.mean(self.recent_delays)) if self.recent_delays else 0.0
        )
        avg_delay_ms = avg_delay_s * 1000.0

        if self.neighbors:
            rssi_vals = [n["rssi"] for n in self.neighbors.values()]
            prr_vals = [n["prr"] for n in self.neighbors.values()]
            avg_rssi_dbm = float(np.mean(rssi_vals))
            norm_rssi = float(np.clip((avg_rssi_dbm + 100.0) / 60.0, 0.0, 1.0))
            avg_prr = float(np.mean(prr_vals))
            deg = float(len(self.neighbors))
        else:
            norm_rssi = 0.5
            avg_prr = 0.5
            deg = 0.0

        norm_dist_bs = float(np.clip(self.dist_to_sink / max(1.0, max_field_dim), 0.0, 1.0))

        return np.array([
            e_res, e_rate, q_occ, arr_rate, srv_rate, rho, plr,
            avg_delay_ms, norm_rssi, avg_prr, deg, norm_dist_bs
        ], dtype=np.float32)

    def get_ground_truth_label(
        self,
        queue_congested_ratio: float = 0.70,
        traffic_intensity_threshold: float = 0.90,
        energy_unhealthy_ratio: float = 0.15,
        plr_unhealthy_threshold: float = 0.40,
        prr_unhealthy_threshold: float = 0.65
    ) -> int:
        if self.is_sink:
            return 0
        if not self.is_alive or self.hardware_failed:
            return 2
        if self.energy.energy_ratio <= energy_unhealthy_ratio:
            return 2

        avg_prr = float(np.mean([n["prr"] for n in self.neighbors.values()])) if self.neighbors else 1.0
        total_epoch_pkts = self.epoch_arrivals + self.epoch_drops
        plr = (self.epoch_drops / total_epoch_pkts) if total_epoch_pkts > 0 else 0.0

        if plr >= plr_unhealthy_threshold or avg_prr <= (1.0 - prr_unhealthy_threshold):
            return 2

        arr_rate = self.epoch_arrivals
        srv_rate = max(1, self.epoch_serviced)
        rho = arr_rate / srv_rate

        if self.queue_occupancy >= queue_congested_ratio or rho >= traffic_intensity_threshold:
            return 1

        return 0

    def distance_to(self, other: "SensorNode") -> float:
        return math.hypot(self.pos_x - other.pos_x, self.pos_y - other.pos_y)
'''

files["V2/simulator/network.py"] = '''"""
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
'''

files["V2/simulator/__init__.py"] = '''from .packet import Packet, PacketType, DropReason
from .energy import FirstOrderRadioModel
from .channel import WirelessChannel
from .node import SensorNode
from .network import WSNNetwork

__all__ = ["Packet", "PacketType", "DropReason", "FirstOrderRadioModel", "WirelessChannel", "SensorNode", "WSNNetwork"]
'''

files["V2/preprocessing/scaler.py"] = '''import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Optional

class FeaturePreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names: List[str] = []

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        self.scaler.fit(X)
        self.is_fitted = True
        if feature_names:
            self.feature_names = list(feature_names)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X).astype(np.float32)

    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        self.fit(X, feature_names)
        return self.transform(X)

    def save(self, filepath: str = "V2/preprocessing/scaler.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"scaler": self.scaler, "feature_names": self.feature_names}, filepath)

    def load(self, filepath: str = "V2/preprocessing/scaler.joblib"):
        data = joblib.load(filepath)
        self.scaler = data["scaler"]
        self.feature_names = data.get("feature_names", [])
        self.is_fitted = True
        return self
'''

files["V2/preprocessing/__init__.py"] = '''from .scaler import FeaturePreprocessor
__all__ = ["FeaturePreprocessor"]
'''

for path, content in files.items():
    with open(path, "w") as f:
        f.write(content)
    print(f"Wrote: {path} ({len(content)} chars)")
