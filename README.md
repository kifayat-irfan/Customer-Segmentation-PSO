# Customer Segmentation using PSO-Optimized K-Means

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-FF9F1C?style=for-the-badge&logo=scikit-learn&logoColor=white)
![PSO](https://img.shields.io/badge/Optimization-PSO-blue?style=for-the-badge)

I built this unsupervised learning project to segment shopping-mall customers by **Annual Income** and **Spending Score**, using **Particle Swarm Optimization (PSO)** to find better initial centroids than standard K-Means random seeding.

---

## Project Overview

Standard K-Means is sensitive to its initial centroids. Here, PSO (via `pyswarm`) optimizes all 5 centroids at once — each particle is a flattened set of centroids, and the objective is the K-Means inertia (sum of squared distances) computed on standardized features. The best centroids found by the swarm are then used for the final cluster assignment, and the code reports a **silhouette score** on the scaled data.

### What the code does
- Loads the Mall Customers dataset (`data/Mall_Customers.csv`), uses the two features `Annual Income (k$)` and `Spending Score (1-100)`
- Standardizes both features with `StandardScaler`
- Runs PSO (25 particles, 80 iterations) to minimize inertia over the centroid space
- Assigns final clusters, prints silhouette score + inertia + cluster centers, saves a scatter plot (`results/PSO_Customer_Segmentation.png`) and the labeled dataset (`results/Mall_Customers_PSO_Clusters.csv`)

---

## Tech Stack

- **Language**: Python 3.10+
- **Optimization**: pyswarm (PSO)
- **Evaluation**: scikit-learn (StandardScaler, silhouette_score)
- **Visualization**: Matplotlib, Seaborn
- **Data Handling**: Pandas, NumPy

---

## Project Structure

```bash
Customer-Segmentation-PSO/
├── data/                        # ← not in repo: add Mall_Customers.csv here (git-ignored)
│   └── Mall_Customers.csv
├── results/                     # ← generated on run (git-ignored)
│   ├── PSO_Customer_Segmentation.png
│   └── Mall_Customers_PSO_Clusters.csv
├── src/
│   ├── config.py                # paths, features, k=5, PSO settings, plot settings
│   ├── pso_logic.py             # PSO_KMeans: inertia objective + pso() wrapper
│   └── main.py                  # full pipeline: load → scale → PSO → evaluate → plot → save
├── requirements.txt
├── LICENSE
└── README.md
```

---

## How to Run

> The dataset is not committed to this repo. Download the well-known **Mall Customers** dataset (e.g. from Kaggle) and place it at `data/Mall_Customers.csv`.

### 1. Clone the Repository
```bash
git clone https://github.com/kifayat-irfan/Customer-Segmentation-PSO.git
cd Customer-Segmentation-PSO
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Project
```bash
python src/main.py
```

This prints the silhouette score, best inertia, and cluster centers to the console, and writes the plot + labeled CSV into `results/`.

---

## Results (from an original run)

The example run in my notes reported a **silhouette score of ~0.54** with **5 clusters**. Typical cluster profiles look like this (your exact numbers will vary slightly between runs since PSO is stochastic):

| Cluster | Annual Income (k$) | Spending Score | Interpretation            |
|---------|--------------------|----------------|---------------------------|
| 0       | ~87                | ~16            | High Income, Low Spenders |
| 1       | ~84                | ~82            | Premium Customers         |
| 2       | ~58                | ~48            | Average Customers         |
| 3       | ~32                | ~30            | Low Income, Low Spenders  |
| 4       | ~30                | ~80            | Impulse Buyers            |

## Business Insights

- **Premium Customers** (Cluster 1): Target with loyalty programs and premium offers.
- **Impulse Buyers** (Cluster 4): Focus on promotional campaigns and flash sales.
- **High Income Low Spenders** (Cluster 0): Need strategies to increase engagement.

---

## Future Improvements

- Comparison between Standard K-Means vs PSO-KMeans
- Streamlit Web Application
- Age & Gender based cluster profiling
- Hyperparameter tuning using PSO for `k`

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Author

**Kifayat Irfan**

**⭐ Star this repository if you found it helpful!**
