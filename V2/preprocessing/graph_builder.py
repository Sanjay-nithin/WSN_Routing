import numpy as np
import torch

class GraphBuilder:
    @staticmethod
    def compute_normalized_adjacency(adj_matrix: np.ndarray) -> np.ndarray:
        N = adj_matrix.shape[0]
        A_tilde = adj_matrix + np.eye(N, dtype=np.float32)
        d_tilde = np.sum(A_tilde, axis=1)
        d_inv_sqrt = np.power(np.maximum(d_tilde, 1e-6), -0.5)
        D_inv = np.diag(d_inv_sqrt)
        return (D_inv @ A_tilde @ D_inv).astype(np.float32)
