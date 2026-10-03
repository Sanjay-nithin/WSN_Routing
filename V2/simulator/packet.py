"""
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
