"""
Professional Configuration for PSO Customer Segmentation Project
"""

# Paths
DATA_FILE = 'data/Mall_Customers.csv'
RESULTS_DIR = 'results'

# Features
FEATURE_COLUMNS = ['Annual Income (k$)', 'Spending Score (1-100)']

# Clustering
DEFAULT_K = 5
RANDOM_STATE = 42

# PSO Settings
PSO_SWARMSIZE = 25
PSO_MAXITER = 80

# Visualization
FIGURE_SIZE = (12, 8)
PLOT_DPI = 300
SAVE_PLOT = True
PLOT_FILENAME = 'results/PSO_Customer_Segmentation.png'