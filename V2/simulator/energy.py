"""
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
