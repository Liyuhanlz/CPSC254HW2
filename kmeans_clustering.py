import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load Iris Dataset

iris = load_iris()

X = iris.data

# Labels are only used for cluster quality validation
y = iris.target

# K-Means Clustering

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(X)

clusters = kmeans.labels_

# Cluster Centroids

centroids = kmeans.cluster_centers_

print("\nCluster Centroids:")
print(centroids)

# RMSE for Each Cluster

print("\nRMSE for Each Cluster:")

for cluster_id in range(3):

    cluster_points = X[clusters == cluster_id]

    centroid = centroids[cluster_id]

    squared_errors = np.sum(
        (cluster_points - centroid) ** 2,
        axis=1
    )

    rmse = np.sqrt(
        np.mean(squared_errors)
    )

    print(f"Cluster {cluster_id}: {rmse:.4f}")

# Cluster Quality Validation

print("\nActual Class Distribution:")

for i in range(3):

    print(f"\nCluster {i}")

    cluster_labels = y[clusters == i]

    unique, counts = np.unique(
        cluster_labels,
        return_counts=True
    )

    for label, count in zip(unique, counts):

        print(
            f"Iris Class {label}: {count}"
        )

# Visualization

plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=clusters,
    cmap='viridis'
)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker='X',
    s=200,
    c='red',
    label='Centroids'
)

plt.title("K-Means Clustering of Iris Dataset")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()

plt.show()