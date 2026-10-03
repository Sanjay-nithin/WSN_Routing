from typing import Optional
from V2.simulator.network import WSNNetwork


class ConventionalBaselines:
    @staticmethod
    def get_direct_routing_fn():
        def direct_routing(u_id: int, network: WSNNetwork) -> Optional[int]:
            return network.sink_id
        return direct_routing

    @staticmethod
    def get_min_hop_routing_fn():
        def min_hop_routing(u_id: int, network: WSNNetwork) -> Optional[int]:
            u_node = network.nodes[u_id]
            alive_neighbors = [v for v in u_node.neighbors if network.nodes[v].is_alive or v == network.sink_id]
            if not alive_neighbors:
                return None
            best_v = min(alive_neighbors, key=lambda v: network.nodes[v].dist_to_sink)
            return best_v
        return min_hop_routing

    @staticmethod
    def get_greedy_geographic_router():
        """
        Standard GPSR / localized Greedy Geographic baseline (forwards to neighbor closest to sink).
        """
        class GreedyGeographicRouter:
            def get_next_hop(self, u_id: int, network: WSNNetwork) -> Optional[int]:
                u_node = network.nodes[u_id]
                alive_neighbors = [v for v in u_node.neighbors if network.nodes[v].is_alive or v == network.sink_id]
                if not alive_neighbors:
                    return None
                return min(alive_neighbors, key=lambda v: network.nodes[v].dist_to_sink)
        return GreedyGeographicRouter()
