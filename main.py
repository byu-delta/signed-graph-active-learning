import argparse
from collections.abc import Set
import json
import os

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import linalg as splinalg

from laplace import LaplaceLearning
from poisson import PoissonLearning
import data.datasets as datasets
import plot as learner_plot

# LinearOperator allows us to do matrix multiplication and conjugate gradient without building the matrix
class MatrixFreeLaplacian(splinalg.LinearOperator):

    def __init__(
        self,
        W: sparse.csr_matrix,
        variant: str,
        *,
        excluded_edge_pairs: Set[tuple[int, int]] | None = None,
    ) -> None:

        # By forcing CSR storage, MatrixFreeLaplacian.W references the same object as Graph.W
        if not sparse.isspmatrix_csr(W):
            raise TypeError("W must be a CSR matrix")
        if variant not in ("signed", "cut", "unsigned"):
            raise ValueError("variant must be 'signed', 'cut', or 'unsigned'")
        self.W = W
        self.variant = variant
        self.excluded_edge_pairs = (
            excluded_edge_pairs if excluded_edge_pairs is not None else set()
        )

        # Call LinearOperator's constructor
        super().__init__(dtype=np.dtype(float), shape=W.shape)

        (
            self.degree_values,
            self.diagonal_values,
            self.adjustment_rows,
        ) = self.calculate_metadata()

    def unsigned_keep_mask(self, row: int, columns: NDArray) -> NDArray:
        """
        Takes a row index and an array of column indices
        Returns a boolean mask indicating which edges should be kept for the unsigned Laplacian variant.
        """
        if not self.excluded_edge_pairs:
            # Keep everything
            return np.ones(len(columns), dtype=bool)
        
        # fromiter allows us to make a boolean array without creating a temporary list
        return np.fromiter(
            (
                # Check if each edge is in the excluded_edge_pairs
                (min(row, int(column)), max(row, int(column)))
                not in self.excluded_edge_pairs
                for column in columns
            ),
            dtype=bool,
            count=len(columns),
        )

    def effective_row(self, row: int) -> tuple[NDArray, NDArray]:
        """
        Takes a row index and returns the effective columns and values for that row, depending on the Laplacian variant.
        """
        start, end = self.W.indptr[row : row + 2]
        columns = self.W.indices[start:end]
        values = self.W.data[start:end]

        if self.variant == "cut":
            keep = values > 0
            return columns[keep], values[keep]

        # Exclude edges that are in excluded_edge_pairs
        keep = self.unsigned_keep_mask(row, columns)
        return columns[keep], np.abs(values[keep])

    def calculate_metadata(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, ...]]:

        degrees = np.empty(self.shape[0], dtype=float)
        diagonal = np.empty(self.shape[0], dtype=float)
        adjustment_rows = []

        for row in range(self.shape[0]):
            start, end = self.W.indptr[row : row + 2]
            raw_columns = self.W.indices[start:end]
            raw_values = self.W.data[start:end]

            if self.variant == "signed":
                degree = np.abs(raw_values).sum()
                self_loop = raw_values[raw_columns == row].sum()
            else:
                columns, values = self.effective_row(row)
                degree = values.sum()
                self_loop = values[columns == row].sum()

            assert (raw_columns == row).sum() <= 1, (
                f"row {row} has multiple stored diagonal entries"
            )
            degrees[row] = degree
            diagonal[row] = degree - self_loop

            needs_sign_adjustment = np.any(raw_values < 0)
            needs_exclusion_adjustment = (
                self.variant == "unsigned"
                # True if non-empty
                and bool(self.excluded_edge_pairs)
                # True if at least one of the row's edges is in the excluded list
                and not self.unsigned_keep_mask(row, raw_columns).all()
            )
            if self.variant != "signed" and (
                needs_sign_adjustment or needs_exclusion_adjustment
            ):
                adjustment_rows.append(row)
        return degrees, diagonal, tuple(adjustment_rows)

    def _matvec(self, vector: NDArray) -> NDArray[np.float64]:
        """
        For a matrix A, matvec(x) computes A @ x
        """
        vector = np.asarray(vector, dtype=float).ravel()
        result = self.degree_values * vector - self.W @ vector
        for row in self.adjustment_rows:
            start, end = self.W.indptr[row : row + 2]
            columns = self.W.indices[start:end]
            values = self.W.data[start:end]
            negative = values < 0

            if self.variant == "cut":
                result[row] += values[negative] @ vector[columns[negative]]
                continue

            result[row] += 2 * (
                values[negative] @ vector[columns[negative]]
            )
            if self.excluded_edge_pairs:
                excluded = ~self.unsigned_keep_mask(row, columns)
                result[row] += np.abs(values[excluded]) @ vector[columns[excluded]]
        return result

    def _rmatvec(self, vector: NDArray) -> NDArray[np.float64]:
        """
        For a matrix A, rmatvec(x) computes A.conj().T @ x
        """
        return self._matvec(vector)


class Graph:
    def __init__(self, weight_mat, y):
        self.y = np.asarray(y).squeeze()
        self.W = (
            # .tocsr is a no-op when the matrix is already CSR
            # So self.W would share the exact same underlying data/indices/indptr arrays as weight_mat
            # copy=True forces an independent copy
            weight_mat.tocsr(copy=True).astype(float, copy=False)
            if sparse.issparse(weight_mat)
            else sparse.csr_matrix(weight_mat, dtype=float, copy=True)
        )
        self.added_edge_pairs = set()
        self.labeled_mask = np.zeros(self.W.shape[0], dtype=bool)
        # Negative portion of non-zero entries
        self.negative_nnz = int(np.count_nonzero(self.W.data < 0))
        self.frustration_scores = {}

    def Laplacian(self):
        """Return the signed Laplacian without constructing a matrix"""
        return MatrixFreeLaplacian(
            self.W, 
            "signed"
    )

    def UnsignedLaplacian(self):
        """Return the unsigned Laplacian without constructing a matrix"""
        return MatrixFreeLaplacian(
            self.W,
            "unsigned",
            excluded_edge_pairs=self.added_edge_pairs,
        )

    def CutEdgeLaplacian(self):
        """Return a cut-edge Laplacian without constructing a matrix"""
        return MatrixFreeLaplacian(
            self.W, 
            "cut"
        )

    def InitialSample(self, init_label_per_class, rng):
        """Randomly label init_label_per_class vertices from each class."""
        # Flattens and returns nonzero indices
        label1 = np.flatnonzero(self.y == 1)
        label2 = np.flatnonzero(self.y == -1)
        indices1 = rng.choice(label1, size=init_label_per_class, replace=False)
        indices2 = rng.choice(label2, size=init_label_per_class, replace=False)
        indices = np.concatenate((indices1, indices2))
        indices_array = np.asarray(indices, dtype=int).ravel()
        self.labeled_mask[indices_array] = True
        return indices

    def count_mislabeled(self, y_true, y_pred):
        """Count how many predicted labels differ from the true labels."""
        y_true = np.asarray(y_true).squeeze()
        y_pred = np.asarray(y_pred).squeeze()
        return int(np.count_nonzero(y_true != y_pred))

    def SynchronizationScores(self, learner, variant="signed"):
        """ 
        ``learner`` contains the predicted label (-1 or +1) for every vertex.
        It is the output of a trained Laplace or Poisson learner, not the
        learner object itself.

        Returns a frustration score for every unlabeled vertex. 
        A larger score means that more of the vertex's edges disagree with the predicted labels. 
        Labeled vertices receive a score of zero.
        """

        if variant not in ("signed", "unsigned", "cut"):
            raise ValueError(f"Unknown Laplacian variant: {variant!r}")

        # Array of predicted labels
        learner = np.asarray(learner).squeeze()
        scores = np.zeros(self.W.shape[0], dtype=float)

        for row in range(self.W.shape[0]):
            start, end = self.W.indptr[row : row + 2]
            columns = self.W.indices[start:end]
            weights = self.W.data[start:end]
            non_diagonal = columns != row

            if variant == "cut":
                keep = non_diagonal & (weights > 0)
                neighboring_labels = learner[columns[keep]]
            elif variant == "unsigned":
                keep = non_diagonal & (weights != 0)
                if self.added_edge_pairs:
                    # keep = keep & np.fromiter(...)
                    keep &= np.fromiter(
                        (
                            (min(row, int(column)), max(row, int(column)))
                            not in self.added_edge_pairs
                            for column in columns
                        ),
                        dtype=bool,
                        count=len(columns),
                    )
                neighboring_labels = learner[columns[keep]]
            else:
                # signed variant
                keep = non_diagonal & (weights != 0)
                # Positive edge (sign = +1): expected label = neighbor's label (unchanged) — they should match.
                # Negative edge (sign = -1): expected label = -neighbor's label — flipped, since they should disagree.
                neighboring_labels = (
                    # An edge between vertex i and vertex j with weight w is considered "satisfied" (not frustrated) when
                    # sign(w) == label[i] * label[j]
                    np.sign(weights[keep]) * learner[columns[keep]]
                )

            scores[row] = np.count_nonzero(
                learner[row] != neighboring_labels
            )

        scores[self.labeled_mask] = 0

        return scores

    def SynchronizationSamplingProbabilities(self, learner, variant="signed"):
        scores = self.SynchronizationScores(learner, variant=variant)
        total = scores.sum()

        if total == 0:
            return scores, np.zeros_like(scores)

        return scores, scores / total

    def SampleByProbability(self, size, probabilities, rng):
        """
        It is assumed that the probabilities are in order of the vertices
        So probabilities[i] corresponds to vertex i
        """
        unlabeled = np.flatnonzero(~self.labeled_mask)
        if len(unlabeled) == 0:
            return np.array([], dtype=int)

        size = min(size, len(unlabeled))
        probabilities = np.asarray(probabilities, dtype=float)
        # In case you called this function with a probability array that includes labeled vertices
        unlabeled_probabilities = probabilities[unlabeled]
        total = unlabeled_probabilities.sum()

        # If there no positive-probability (frustrated) vertices, sample uniformly
        if total == 0:
            labeled_indices = rng.choice(unlabeled, size=size, replace=False)
        else:
            positive_mask = unlabeled_probabilities > 0
            positive_unlabeled = unlabeled[positive_mask]
            positive_probabilities = unlabeled_probabilities[positive_mask]

            # rng.choice with replace=False requires that there be at least as many nonzero-probability entries as the requested sample size 
            weighted_size = min(size, len(positive_unlabeled))
            weighted_labeled = rng.choice(
                positive_unlabeled,
                size=weighted_size,
                replace=False,
                p=positive_probabilities / positive_probabilities.sum(),
            )

            # If there are too few positive-probability (frustrated) vertices, sample uniformly
            remaining_size = size - weighted_size
            if remaining_size > 0:
                remaining_mask = np.ones(len(unlabeled), dtype=bool)
                # np.searchsorted finds the positions of the weighted labels inside unlabeled
                remaining_mask[np.searchsorted(unlabeled, weighted_labeled)] = False
                remaining_unlabeled = unlabeled[remaining_mask]
                uniform_labeled = rng.choice(
                    remaining_unlabeled,
                    size=remaining_size,
                    replace=False,
                )
                labeled_indices = np.concatenate([weighted_labeled, uniform_labeled])
            else:
                labeled_indices = weighted_labeled

        indices_array = np.asarray(labeled_indices, dtype=int).ravel()
        self.labeled_mask[indices_array] = True
        return labeled_indices

    def AddEdges(self, iter_samples, rng, add_edges_per_node=3):
        """
        For each newly sampled node, add edges between it and previously labeled nodes.

        Parameters:
        - iter_samples: array-like of int
            The indices of the newly sampled nodes.
        - rng: np.random.Generator
            A random number generator for reproducibility.
        - add_edges_per_node: int, optional (default=3)
            The number of positive and negative edges to add for each newly sampled node.
        """

        if not isinstance(add_edges_per_node, (int, np.integer)):
            raise TypeError("add_edges_per_node must be a nonnegative integer")
        if add_edges_per_node < 0:
            raise ValueError("add_edges_per_node must be nonnegative")
        if add_edges_per_node == 0:
            return

        iter_samples = np.asarray(iter_samples, dtype=int).ravel()

        # Reconstruct the set of labeled nodes in the previous iteration
        previous_mask = self.labeled_mask.copy()
        previous_mask[iter_samples] = False
        previous_label_idx = np.flatnonzero(previous_mask)
        previous_labels = self.y[previous_label_idx]

        row_parts, col_parts, val_parts = [], [], []

        for node in iter_samples:
            node_label = self.y[node]
            existing_neighbors = self.W.indices[
                # Find the columns for the nonzero entries in the row corresponding to the current node
                self.W.indptr[node] : self.W.indptr[node + 1]
            ]

            # Finds previously lableled indices that are not neighbors of the current node
            is_nonedge = ~np.isin(previous_label_idx, existing_neighbors)
            candidates = previous_label_idx[is_nonedge]
            candidate_labels = previous_labels[is_nonedge]
            same_label_idx = candidates[candidate_labels == node_label]
            different_label_idx = candidates[candidate_labels != node_label]

            for candidate_idx, edge_weight in (
                (same_label_idx, 1),
                (different_label_idx, -1),
            ):
                if len(candidate_idx) == 0:
                    continue

                k = min(add_edges_per_node, len(candidate_idx))
                nbrs = rng.choice(candidate_idx, size=k, replace=False)
                node_arr = np.full(k, node, dtype=np.int64)
                row_parts.append(node_arr)
                col_parts.append(nbrs)
                row_parts.append(nbrs)
                col_parts.append(node_arr)
                val_parts.append(np.full(2 * k, edge_weight, dtype=self.W.dtype))

        if not row_parts:
            return

        update_rows = np.concatenate(row_parts)
        update_cols = np.concatenate(col_parts)
        update_vals = np.concatenate(val_parts)

        undirected = update_rows < update_cols
        self.added_edge_pairs.update(
            zip(
                update_rows[undirected].tolist(),
                update_cols[undirected].tolist(),
            )
        )

        # Works like update[update_rows[i], update_cols[i]] = update_vals[i]
        update = sparse.csr_matrix(
            (update_vals, (update_rows, update_cols)),
            shape=self.W.shape,
        )
        self.W = (self.W + update).tocsr()
        self.negative_nnz += int(np.count_nonzero(update_vals < 0))

    def csr_data_positions(self, matrix, rows, cols):
        if not sparse.isspmatrix_csr(matrix):
            raise TypeError("matrix must be a CSR matrix")
        if not matrix.has_sorted_indices:
            matrix.sort_indices()

        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
        starts = matrix.indptr[rows]
        ends = matrix.indptr[rows + 1]

        positions = np.array(
            [
                s + np.searchsorted(matrix.indices[s:e], c)
                for s, e, c in zip(starts, ends, cols)
            ]
        )

        invalid = (positions >= ends) | (matrix.indices[positions] != cols)
        if invalid.any():
            bad = np.where(invalid)[0][0]
            raise ValueError(
                f"Entry ({rows[bad]}, {cols[bad]}) is not present in sparse matrix"
            )

        return positions

    def NegateEdges(self, recently_labeled_nodes):
        labeled_idx = np.flatnonzero(self.labeled_mask)
        observed_labels = self.y[labeled_idx]
        recently_labeled_idx = np.asarray(recently_labeled_nodes, dtype=int).ravel()

        # Get the submatrix of W corresponding to edges between recently labeled nodes and previously labeled nodes
        # tocoo() has (row, col, data) instead of (indptr, indices, data)
        K_labeled = self.W[recently_labeled_idx][:, labeled_idx].tocoo()
        k_rows = K_labeled.row
        k_cols = K_labeled.col
        k_weights = K_labeled.data
        recently_labeled_labels = self.y[recently_labeled_idx]
        
        # Contains the original row node ID for every nonzero edge found in K_labeled
        u_all = recently_labeled_idx[k_rows]
        # Same as above, but for columns
        v_all = labeled_idx[k_cols]

        recently_labeled_mask = np.zeros(self.W.shape[0], dtype=bool)
        recently_labeled_mask[recently_labeled_idx] = True

        # If v_all is an older node keep the edge
        # Or if u_all is a newer node, keep the edge when u < v
        handle_once = (~recently_labeled_mask[v_all]) | (u_all < v_all)
        mask = (
            handle_once
            & (k_weights > 0)
            # Ignore self edgees
            & (recently_labeled_labels[k_rows] != observed_labels[k_cols])
        )
        u = u_all[mask]
        v = labeled_idx[k_cols[mask]]

        if len(u) == 0:
            return []

        w_uv = self.csr_data_positions(self.W, u, v)
        w_vu = self.csr_data_positions(self.W, v, u)
        self.W.data[w_uv] *= -1
        self.W.data[w_vu] *= -1
        self.negative_nnz += len(w_uv) + len(w_vu)
        return np.column_stack([u, v]).tolist()

    def RunActiveLearning(
        self,
        classifier="Poisson",
        variant="signed",
        init_label_per_class=5,
        num_iter=75,
        label_per_iter=10,
        add_edges_per_node=3,
        center_poisson_scores=True,
        seed=None,
    ):
        
        rng = np.random.default_rng(seed)

        if variant not in ("signed", "cut", "unsigned"):
            raise ValueError(
                "variant must be 'signed', 'cut', or 'unsigned'"
            )
        if classifier not in ("Laplace", "Poisson"):
            raise ValueError("classifier must be 'Laplace' or 'Poisson'")
        if not isinstance(add_edges_per_node, (int, np.integer)):
            raise TypeError("add_edges_per_node must be a nonnegative integer")
        if add_edges_per_node < 0:
            raise ValueError("add_edges_per_node must be nonnegative")

        initial_samples = self.InitialSample(init_label_per_class, rng)

        history = []

        for iteration in range(num_iter):
            labeled_idx = np.flatnonzero(self.labeled_mask)
            unlabeled_idx = np.flatnonzero(~self.labeled_mask)
            observed_labels = self.y[labeled_idx]

            # Explain these
            num_unlabeled = len(unlabeled_idx)
            #self.frustration_scores = {}

            if variant == "signed":
                laplacian = self.Laplacian()
            elif variant == "cut":
                laplacian = self.CutEdgeLaplacian()
            else:
                laplacian = self.UnsignedLaplacian()

            if classifier == "Laplace":
                learner = LaplaceLearning(laplacian)
                learner.fit(labeled_idx, observed_labels)
                prediction = learner.predict()
            else:
                learner = PoissonLearning(laplacian)
                learner.fit(labeled_idx, observed_labels)
                prediction = learner.predict(
                    center_scores=center_poisson_scores and variant == "signed"
                )

            mislabeled_count = self.count_mislabeled(
                self.y[unlabeled_idx],
                prediction[unlabeled_idx],
            )
            mislabeled_rate = (
                mislabeled_count / num_unlabeled if num_unlabeled else 0.0
            )

            scores, sampling_probabilities = (
                self.SynchronizationSamplingProbabilities(
                    prediction,
                    variant=variant,
                )
            )

            #self.frustration_scores[f"{classifier.lower()}_{variant}"] = scores

            del learner, prediction, laplacian

            #self._active_variant_labeled = {variant: labeled_idx}

            history_row = {
                "iteration": iteration,
                "labeled_count": int(self.labeled_mask.sum()),

                #
                "nnz": self.W.nnz,
                "negative_nnz": self.negative_nnz,

                "mislabeled_count": mislabeled_count,
                "mislabeled_rate": mislabeled_rate,
            }

            iter_samples = self.SampleByProbability(
                label_per_iter,
                sampling_probabilities,
                rng,
            )

            if variant == "signed" and add_edges_per_node > 0:
                self.AddEdges(
                    iter_samples,
                    rng,
                    add_edges_per_node=add_edges_per_node,
                )
            if iteration == 0:
                recently_labeled_nodes = np.concatenate(
                    (initial_samples, iter_samples)
                )
            else:
                recently_labeled_nodes = iter_samples
            num_negated_edges = len(
                self.NegateEdges(recently_labeled_nodes=recently_labeled_nodes)
            )


            # Tracking
            history_row["negated_edges"] = (
                0 if variant == "unsigned" else num_negated_edges
            )
            history.append(history_row)

        return history

    def average_histories(
        self,
        iter,
        classifier="Poisson",
        variant="signed",
        init_label_per_class=5,
        num_iter=75,
        label_per_iter=10,
        add_edges_per_node=3,
        center_poisson_scores=True,
        seed=None,
    ):
        """Run active learning ``iter`` times and average the histories."""
        histories = [
            Graph(self.W.copy(), self.y.copy()).RunActiveLearning(
                classifier=classifier,
                variant=variant,
                init_label_per_class=init_label_per_class,
                num_iter=num_iter,
                label_per_iter=label_per_iter,
                add_edges_per_node=add_edges_per_node,
                center_poisson_scores=center_poisson_scores,
                seed=None if seed is None else seed + trial,
            )
            for trial in range(iter)
        ]
        if not histories:
            return []

        return [
            {
                "iteration": iteration,
                **{
                    metric: np.mean(
                        [history[iteration][metric] for history in histories]
                    )
                    for metric in histories[0][iteration]
                    if metric != "iteration"
                },
            }
            for iteration in range(len(histories[0]))
        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run signed-graph active learning."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "MNIST_0_1",
            "MNIST_4_9",
            "MNIST_EVEN_ODD",
            "FASHIONMNIST_SANDAL_SNEAKER",
            "CIFAR_CAT_DOG",
            "CLUSTERS",
        ],
        help="Prepared dataset to run active learning on",
    )
    parser.add_argument(
        "--classifier",
        default="Poisson",
        choices=["Poisson", "Laplace"],
        help="Which classifier to run",
    )
    parser.add_argument(
        "--variant",
        default="signed",
        choices=["signed", "cut", "unsigned"],
        help="Graph variant used for learning, evaluation, and sampling",
    )
    parser.add_argument(
        "--init-label-per-class",
        type=int,
        default=5,
        help="Number of initially labeled nodes from each class",
    )
    parser.add_argument(
        "--label-per-iter",
        type=int,
        default=10,
        help="Number of new nodes to label per iteration",
    )
    parser.add_argument(
        "--num-iter",
        type=int,
        default=75,
        help="Number of active-learning iterations to run",
    )
    parser.add_argument(
        "--add-edges-per-node",
        type=int,
        default=3,
        help=(
            "Maximum positive and negative edges per new node; defaults to 3, "
            "and zero disables additions"
        ),
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of independent trials to run and average",
    )

    args = parser.parse_args()
    if args.num_trials < 1:
        parser.error("--num-trials must be at least 1")

    figures_dir = os.path.join("figures", args.dataset)
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    W_binary, y = datasets.load_dataset(args.dataset)
    graph = Graph(W_binary, y)
    run_settings = {
        "classifier": args.classifier,
        "variant": args.variant,
        "init_label_per_class": args.init_label_per_class,
        "label_per_iter": args.label_per_iter,
        "num_iter": args.num_iter,
        "add_edges_per_node": args.add_edges_per_node,
    }

    if args.num_trials == 1:
        history = graph.RunActiveLearning(**run_settings)
    else:
        history = graph.average_histories(args.num_trials, **run_settings)

    with open(os.path.join(results_dir, f"{args.dataset}.json"), "w") as f:
        json.dump(history, f, indent=2)

    learner_plot.save_plot(
        history,
        os.path.join(figures_dir, "error_rate.png"),
    )
