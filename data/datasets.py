import os

import graphlearning as gl
import numpy as np
from scipy import sparse


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GL_DATA_DIR = os.path.join(DATA_DIR, "data_gl")
K = 20

gl.datasets.data_dir = GL_DATA_DIR
gl.weightmatrix.knn_dir = os.path.join(DATA_DIR, "knn_data")


def save_dataset(name, X, y, W):
    """Save features, signed labels, and a sparse graph under data/<name>."""
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    W = sparse.csr_matrix(W)

    if X.ndim != 2:
        raise ValueError(f"{name}: X must be two-dimensional; got {X.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{name}: X has {X.shape[0]} rows but y has {y.shape[0]} labels."
        )
    if W.shape != (X.shape[0], X.shape[0]):
        raise ValueError(
            f"{name}: W must have shape {(X.shape[0], X.shape[0])}; got {W.shape}."
        )
    if set(np.unique(y)) != {-1, 1}:
        raise ValueError(f"{name}: y must contain both -1 and +1.")

    output_dir = os.path.join(DATA_DIR, name)
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_dir, "X.npz"), X=X)
    np.save(os.path.join(output_dir, "y.npy"), y, allow_pickle=False)
    sparse.save_npz(os.path.join(output_dir, "W.npz"), W, compressed=True)
    print(f"Created {name}: X={X.shape}, y={y.shape}, W={W.shape}")


def load_dataset(name):
    """Load the graph and labels saved under data/<name> by save_dataset."""
    dataset_dir = os.path.join(DATA_DIR, name)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"No dataset found at {dataset_dir}. Run data/datasets.py to generate it."
        )

    y = np.load(os.path.join(dataset_dir, "y.npy"))
    W = sparse.load_npz(os.path.join(dataset_dir, "W.npz"))
    return W, y


def class_pair(X, labels, W, class1, class2):
    """Restrict features and a graph to two signed classes."""
    condition = (labels == class1) | (labels == class2)
    y = np.where(labels[condition] == class1, 1, -1)
    return X[condition], y, W[condition][:, condition]


def generate_mnist_datasets():
    X, labels = gl.datasets.load("mnist", metric="vae")
    W = gl.weightmatrix.knn(
        "mnist",
        k=K,
        metric="vae",
        kernel="uniform",
    )

    pair_X, y, pair_W = class_pair(X, labels, W, 0, 1)
    save_dataset("MNIST_0_1", pair_X, y, pair_W)

    pair_X, y, pair_W = class_pair(X, labels, W, 4, 9)
    save_dataset("MNIST_4_9", pair_X, y, pair_W)

    pair_X, y, pair_W = class_pair(X, labels, W, 4, 7)
    save_dataset("MNIST_4_7", pair_X, y, pair_W)

    y = np.where(labels.astype(int) % 2 == 0, 1, -1)
    save_dataset("MNIST_EVEN_ODD", X, y, W)


def generate_fashionmnist_dataset():
    X, labels = gl.datasets.load("fashionmnist", metric="vae")
    W = gl.weightmatrix.knn(
        "fashionmnist",
        k=K,
        metric="vae",
        kernel="uniform",
    )
    X, y, W = class_pair(X, labels, W, 5, 7)
    save_dataset("FASHIONMNIST_SANDAL_SNEAKER", X, y, W)


def generate_cifar_dataset():
    X, labels = gl.datasets.load("cifar10", metric="simclr")
    W = gl.weightmatrix.knn(
        "cifar10",
        k=K,
        metric="simclr",
        kernel="uniform",
    )
    X, y, W = class_pair(X, labels, W, 3, 5)
    save_dataset("CIFAR_CAT_DOG", X, y, W)


def generate_clusters_dataset():
    rng = np.random.default_rng(92)
    n_per_cluster = 750
    covariance = [[0.1, 0], [0, 0.1]]
    means = np.array([
        [1.0, 0.4],
        [1.1, -1.0],
        [0, 0.5],
    ])

    points = np.vstack([
        rng.multivariate_normal(mean, covariance, n_per_cluster)
        for mean in means
    ])
    y = np.array([-1] * n_per_cluster + [1] * n_per_cluster + [1] * n_per_cluster)

    neighbor_indices, neighbor_distances = gl.weightmatrix.knnsearch(
        points,
        K + 1,
        method="kdtree",
        similarity="euclidean",
    )
    W = gl.weightmatrix.knn(
        data=points,
        k=K,
        kernel="uniform",
        knn_data=(neighbor_indices, neighbor_distances),
    )
    save_dataset("CLUSTERS", points, y, W)


if __name__ == "__main__":
    generate_mnist_datasets()
    generate_fashionmnist_dataset()
    generate_cifar_dataset()
    generate_clusters_dataset()
