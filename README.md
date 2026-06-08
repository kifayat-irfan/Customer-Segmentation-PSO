# Customer Segmentation using PSO-Optimized K-Means

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-FF9F1C?style=for-the-badge&logo=scikit-learn&logoColor=white)
![PSO](https://img.shields.io/badge/Optimization-PSO-blue?style=for-the-badge)

A professional **Unsupervised Learning" project that uses **Particle Swarm Optimization (PSO)** to enhance K-Means clustering for customer segmentation.

---

##  Project Overview

This project segments customers of a shopping mall based on their **Annual Income** and **Spending Score** using an advanced approach combining **K-Means Clustering** with **Particle Swarm Optimization (PSO)**.

PSO is used to intelligently optimize the initial cluster centroids, leading to better cluster quality compared to standard K-Means.

### Key Achievements
- **Silhouette Score**: **0.5399**
- Optimal number of clusters: **5**
- Clean, modular, and production-ready code structure
- Beautiful visualizations and saved results

---

## 🛠 Tech Stack

- **Language**: Python 3.10+
- **Machine Learning**: scikit-learn
- **Optimization**: pyswarm (PSO)
- **Visualization**: Matplotlib, Seaborn
- **Data Handling**: Pandas, NumPy

---

##  Project Structure

```bash
customer-segmentation-pso/
├── data/
│   └── Mall_Customers.csv
├── results/
│   ├── PSO_Customer_Segmentation.png
│   └── Mall_Customers_PSO_Clusters.csv
├── src/
│   ├── config.py
│   ├── pso_logic.py
│   └── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

##  How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/customer-segmentation-pso.git
cd customer-segmentation-pso
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Project
```bash
python src/main.py
```

---

##  Results

**Best PSO Optimized Clusters:**

| Cluster | Annual Income (k$) | Spending Score | Interpretation              |
|---------|--------------------|----------------|-----------------------------|
| 0       | ~87.4              | 16.2           | High Income, Low Spenders   |
| 1       | ~84.2              | 82.3           | **Premium Customers**       |
| 2       | ~57.5              | 47.8           | Average Customers           |
| 3       | ~32.3              | 29.8           | Low Income, Moderate Spend  |
| 4       | ~29.6              | 80.1           | **Impulse Buyers**          |

![Customer Segmentation Plot](results/PSO_Customer_Segmentation.png)

---

##  Business Insights

- **Premium Customers** (Cluster 1): Target with loyalty programs and premium offers.
- **Impulse Buyers** (Cluster 4): Focus on promotional campaigns and flash sales.
- **High Income Low Spenders** (Cluster 0): Need strategies to increase engagement.

---

##  Future Improvements

- Comparison between Standard K-Means vs PSO-KMeans
- Streamlit Web Application
- Age & Gender based cluster profiling
- Hyperparameter tuning using PSO for `k`

---

##  License

This project is open source and available under the [MIT License](LICENSE).

---

##  Author

**Kifayat Irfan**  
Machine Learning Enthusiast | Data Science Portfolio

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/YOUR_LINKEDIN)

---

**⭐ Star this repository if you found it helpful!**

```

