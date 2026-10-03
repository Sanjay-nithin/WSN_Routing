"""
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
    def x(self) -> float:
        return self.pos_x

    @property
    def y(self) -> float:
        return self.pos_y

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
