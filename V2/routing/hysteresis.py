from typing import Dict, Optional

class RoutingHysteresisFilter:
    def __init__(self, hysteresis_factor: float = 0.15):
        self.hysteresis = float(hysteresis_factor)
        self.active_routes: Dict[int, int] = {}
        self.active_costs: Dict[int, float] = {}
        self.route_changes_count = 0

    def should_update_route(
        self,
        node_id: int,
        candidate_next_hop: int,
        candidate_cost: float,
        is_current_alive: bool = True
    ) -> bool:
        if node_id not in self.active_routes:
            self.active_routes[node_id] = candidate_next_hop
            self.active_costs[node_id] = candidate_cost
            self.route_changes_count += 1
            return True

        current_hop = self.active_routes[node_id]
        current_cost = self.active_costs[node_id]

        if not is_current_alive or candidate_next_hop == current_hop:
            self.active_routes[node_id] = candidate_next_hop
            self.active_costs[node_id] = candidate_cost
            return True

        threshold_cost = current_cost * (1.0 - self.hysteresis)
        if candidate_cost < threshold_cost:
            self.active_routes[node_id] = candidate_next_hop
            self.active_costs[node_id] = candidate_cost
            self.route_changes_count += 1
            return True

        return False

    def get_active_hop(self, node_id: int) -> Optional[int]:
        return self.active_routes.get(node_id)
