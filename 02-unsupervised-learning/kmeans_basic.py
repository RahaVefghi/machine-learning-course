import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def run_kmeans_blobs():
    # 1. Generate synthetic 2D data clusters
    X, y_true = make_blobs(n_samples=300, n_features=2, centers=4, cluster_std=0.6, random_state=42)

    # 2. Fit K-Means clustering model
    kmeans = KMeans(n_clusters=4, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    print("Estimated Cluster Centers:\n", centers)

    # 3. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Raw points
    ax1.scatter(X[:, 0], X[:, 1], s=40, color="gray", alpha=0.7)
    ax1.set_title("Original Unlabeled Data")

    # Clustered points with centroids
    ax2.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=40, cmap="viridis", alpha=0.7)
    ax2.scatter(centers[:, 0], centers[:, 1], c="red", s=180, marker="X", label="Centroids")
    ax2.set_title("K-Means Clustered Data")
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_kmeans_blobs()