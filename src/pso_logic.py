import numpy as np
from typing import Tuple

class PSO_KMeans:
    """Particle Swarm Optimization for K-Means Clustering"""
    
    def __init__(self, X_scaled: np.ndarray, k: int = 5):
        self.X_scaled = X_scaled
        self.k = k
        self.n_features = X_scaled.shape[1]
        self.dim = k * self.n_features
    
    def objective_function(self, particle: np.ndarray) -> float:
        """Minimize inertia"""
        centers = particle.reshape((self.k, self.n_features))
        
        distances = np.zeros((len(self.X_scaled), self.k))
        for i in range(self.k):
            distances[:, i] = np.sum((self.X_scaled - centers[i]) ** 2, axis=1)
        
        labels = np.argmin(distances, axis=1)
        
        inertia = 0.0
        for i in range(self.k):
            cluster_points = self.X_scaled[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centers[i]) ** 2)
        
        return inertia
    
    def run_pso(self, lb: np.ndarray, ub: np.ndarray, 
                swarmsize: int = 25, maxiter: int = 80) -> Tuple[np.ndarray, float]:
        """Run PSO optimization"""
        from pyswarm import pso
        print(f" Running PSO with {swarmsize} particles for {maxiter} iterations...")
        
        result = pso(
            func=self.objective_function,
            lb=lb,
            ub=ub,
            swarmsize=swarmsize,
            maxiter=maxiter,
            debug=False
        )
        
        print(f" PSO Completed | Best Inertia: {result.fun:.4f}")
        return result.x, result.fun
    
    def assign_clusters(self, best_position: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Assign clusters using best centroids"""
        centers_scaled = best_position.reshape((self.k, self.n_features))
        
        distances = np.zeros((len(self.X_scaled), self.k))
        for i in range(self.k):
            distances[:, i] = np.sum((self.X_scaled - centers_scaled[i]) ** 2, axis=1)
        
        labels = np.argmin(distances, axis=1)
        return labels, centers_scaled