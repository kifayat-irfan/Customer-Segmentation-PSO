import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from config import *
from pso_logic import PSO_KMeans

def ensure_directories():
    """Create required directories"""
    os.makedirs('data', exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    print(" Starting PSO-Optimized Customer Segmentation Project\n")
    ensure_directories()
    
    # Load Data
    df = pd.read_csv(DATA_FILE)
    print(f" Dataset loaded: {df.shape[0]} customers")
    
    # Preprocessing
    X = df[FEATURE_COLUMNS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(" Features scaled successfully.")
    
    # PSO Optimization
    pso_kmeans = PSO_KMeans(X_scaled, k=DEFAULT_K)
    
    lb = X_scaled.min(axis=0).repeat(DEFAULT_K)
    ub = X_scaled.max(axis=0).repeat(DEFAULT_K)
    
    best_position, best_inertia = pso_kmeans.run_pso(
        lb, ub, PSO_SWARMSIZE, PSO_MAXITER
    )
    
    labels, centers_scaled = pso_kmeans.assign_clusters(best_position)
    centers_original = scaler.inverse_transform(centers_scaled)
    
    df['PSO_Cluster'] = labels
    
    # Evaluation
    sil_score = silhouette_score(X_scaled, labels)
    print(f"\n RESULTS")
    print(f"   Silhouette Score : {sil_score:.4f}")
    print(f"   Best Inertia     : {best_inertia:.4f}")
    print("\n Cluster Distribution:")
    print(df['PSO_Cluster'].value_counts().sort_index())
    
    print("\n Optimized Cluster Centers:")
    for i, center in enumerate(centers_original):
        print(f"   Cluster {i}: Income=${center[0]:.1f}k | Spending={center[1]:.1f}")
    
    # Visualization
    plt.figure(figsize=FIGURE_SIZE)
    sns.scatterplot(
        x=df[FEATURE_COLUMNS[0]], 
        y=df[FEATURE_COLUMNS[1]], 
        hue=df['PSO_Cluster'], 
        palette='viridis', 
        s=100,
        alpha=0.85,
        edgecolor='black'
    )
    
    plt.scatter(centers_original[:, 0], centers_original[:, 1], 
                c='red', s=400, marker='X', 
                label='PSO Centroids', edgecolors='black', linewidth=2)
    
    plt.title('Customer Segmentation using PSO-Optimized K-Means', fontsize=15, pad=20)
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend(title='Cluster')
    plt.grid(True, alpha=0.3)
    
    if SAVE_PLOT:
        plt.savefig(PLOT_FILENAME, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"\n Plot saved: {PLOT_FILENAME}")
    
    plt.show()
    
    # Save Results
    df.to_csv(f'{RESULTS_DIR}/Mall_Customers_PSO_Clusters.csv', index=False)
    print(f" Results saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()