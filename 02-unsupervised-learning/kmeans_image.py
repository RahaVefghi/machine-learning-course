import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

def run_kmeans_digits():
    digits = load_digits()
    print(f"Loaded Digits Dataset: {digits.data.shape[0]} samples, {digits.data.shape[1]} features (8x8 pixels)")

    # Fit K-Means with 10 clusters (digits 0 to 9)
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    kmeans.fit(digits.data)

    # Reshape cluster centers back to 8x8 image representations
    centers = kmeans.cluster_centers_.reshape((10, 8, 8))

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    fig.suptitle("K-Means Cluster Centers (8x8 Learned Digits)", fontsize=14)

    for i, ax in enumerate(axes.flat):
        ax.imshow(centers[i], cmap="binary", interpolation="nearest")
        ax.set_title(f"Cluster {i}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_kmeans_digits()