"""
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
