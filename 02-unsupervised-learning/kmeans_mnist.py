import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_mnist_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "mnist.npz")

    if os.path.exists(data_path):
        data = np.load(data_path)
        x_train = data["x_train"]
    else:
        import tensorflow as tf
        (x_train, _), _ = tf.keras.datasets.mnist.load_data()

    # Normalize and flatten
    x_train = (x_train / 255.0).reshape((-1, 784))
    return x_train

def cluster_and_visualize(x_data, n_samples=10000):
    # Cluster on a subset for fast training
    subset = x_data[:n_samples]
    print(f"Clustering {subset.shape[0]} MNIST images with K-Means...")

    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    kmeans.fit(subset)

    # Reconstruct 28x28 centroids
    centers = kmeans.cluster_centers_.reshape((10, 28, 28))

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    fig.suptitle("K-Means MNIST Cluster Centroids (28x28)", fontsize=14)

    for i, ax in enumerate(axes.flat):
        ax.imshow(centers[i], cmap="gray")
        ax.set_title(f"Centroid {i}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = load_mnist_data()
    cluster_and_visualize(data)