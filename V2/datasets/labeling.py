import numpy as np

class GroundTruthLabeler:
    LABEL_NAMES = {0: "Healthy", 1: "Congested", 2: "Unhealthy"}
    def __init__(self, queue_congested_ratio=0.70, traffic_intensity_threshold=0.90, energy_unhealthy_ratio=0.15, plr_unhealthy_threshold=0.40):
        self.queue_thresh = queue_congested_ratio
        self.intensity_thresh = traffic_intensity_threshold
        self.energy_thresh = energy_unhealthy_ratio
        self.plr_thresh = plr_unhealthy_threshold

    def label_vector(self, row: np.ndarray) -> int:
        if row[0] <= self.energy_thresh or row[6] >= self.plr_thresh:
            return 2
        if row[2] >= self.queue_thresh or row[5] >= self.intensity_thresh:
            return 1
        return 0
