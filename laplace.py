"""Binary graph Laplace learning.

Given labeled vertices ``L`` and unlabeled vertices ``U``, Laplace learning
extends binary boundary values harmonically by solving

.. math:: L_{UU} u_U = -L_{UL} u_L.

Labels are encoded as ``-1`` and ``+1``, so the sign of the solution gives the
predicted class. See Zhu, Ghahramani, and Lafferty, *Semi-Supervised Learning
Using Gaussian Fields and Harmonic Functions* (ICML, 2003).
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import linalg as splinalg


FloatArray: TypeAlias = NDArray[np.float64]


class LaplaceLearning:
    """Binary harmonic-extension classifier for a graph Laplacian."""

    def __init__(
        self,
        L: splinalg.LinearOperator,
        *,
        rtol: float = 1e-5,
        max_iter: int = 100_000,
    ) -> None:
        if not isinstance(L, splinalg.LinearOperator):
            raise TypeError("L must be a matrix-free scipy LinearOperator")
        if not hasattr(L, "diagonal_values"):
            raise TypeError("L must provide diagonal_values")
        self.L = L
        self.rtol = float(rtol)
        self.max_iter = int(max_iter)
        self.scores: FloatArray | None = None
        self.classes: NDArray | None = None
        self.cg_info: int | None = None
        self.converged: bool | None = None
        self.cg_iterations: int | None = None
        self.residual_norm: float | None = None

    def fit(
        self,
        train_ind: ArrayLike,
        train_labels: ArrayLike,
        *,
        x0: ArrayLike | None = None,
    ) -> LaplaceLearning:

        train_ind = np.asarray(train_ind, dtype=int).ravel()
        train_labels = np.asarray(train_labels).ravel()
        classes = np.unique(train_labels)

        unlabeled_mask = np.ones(self.L.shape[0], dtype=bool)
        unlabeled_mask[train_ind] = False
        unlabeled_ind = np.flatnonzero(unlabeled_mask)
        boundary = np.zeros(self.L.shape[0], dtype=float)
        boundary[train_ind] = train_labels

        return self._solve_prepared(
            train_ind,
            train_labels,
            classes,
            unlabeled_ind,
            boundary,
            x0=x0,
        )

    def _solve_prepared(
        self,
        train_ind: ArrayLike,
        label_values: ArrayLike,
        classes: ArrayLike,
        unlabeled_ind: ArrayLike,
        boundary: ArrayLike,
        *,
        x0: ArrayLike | None = None,
    ) -> LaplaceLearning:
        
        """
        Fundamental Laplace learning condition is that the Laplacian 
        applied to the learned function must be 0 on the unlabeled vertices

            (Lu)_U = 0

        Conceptually we can think of Laplace learning as solving
        
            L_UU u_U = -L_UL u_L, 
        
        where:

            L = [[L_LL, L_LU],
                [L_UL, L_UU]]

            u = [u_L, u_U]

        and the full boundary vector is:

            boundary = [u_L, 0]

        The implementation partitions the problem implicitly using labeled and
        unlabeled index arrays. It does not explicitly construct the block
        matrices L_UU and L_UL.

        
        """
        train_ind = np.asarray(train_ind, dtype=int).ravel()
        label_values = np.asarray(label_values, dtype=float).ravel()
        unlabeled_ind = np.asarray(unlabeled_ind, dtype=int).ravel()
        boundary = np.asarray(boundary, dtype=float).ravel()

        # Can be thought of as -L_UL u_L
        rhs = np.asarray(-(self.L @ boundary)).ravel()[unlabeled_ind]

        num_vertices = self.L.shape[0]

        def restricted_matvec(unlabeled_vector: FloatArray) -> FloatArray:
            full_vector = np.zeros(num_vertices, dtype=float)
            full_vector[unlabeled_ind] = unlabeled_vector
            return np.asarray(self.L @ full_vector).ravel()[unlabeled_ind]

        # Can be thought of as L_UU
        A = splinalg.LinearOperator(
            shape=(len(unlabeled_ind), len(unlabeled_ind)),
            matvec=restricted_matvec,
            rmatvec=restricted_matvec,
            dtype=float,
        )
        full_diagonal = self.L.diagonal_values
        # Used for Jacobi preconditioning of the conjugate gradient solver
        inverse_diagonal = 1.0 / np.maximum(
            np.abs(full_diagonal[unlabeled_ind]), 1e-10
        )
        preconditioner = splinalg.LinearOperator(
            shape=A.shape,
            matvec=lambda vector: inverse_diagonal * vector,
            rmatvec=lambda vector: inverse_diagonal * vector,
            dtype=float,
        )
        initial = (
            None if x0 is None else np.asarray(x0, dtype=float).ravel()[unlabeled_ind]
        )

        iterations = 0

        def count_iteration(_: FloatArray) -> None:
            nonlocal iterations
            iterations += 1

        unlabeled_scores, info = splinalg.cg(
            A,
            rhs,
            M=preconditioner,
            x0=initial,
            rtol=self.rtol,
            atol=0.0,
            maxiter=self.max_iter,
            callback=count_iteration,
        )

        scores = np.zeros(self.L.shape[0], dtype=float)
        scores[unlabeled_ind] = unlabeled_scores
        scores[train_ind] = label_values
        self.scores = scores
        self.classes = np.asarray(classes)
        self.cg_info = int(info)
        self.converged = info == 0
        self.cg_iterations = iterations
        self.residual_norm = float(np.linalg.norm(rhs - A @ unlabeled_scores))
        return self

    def predict(self) -> NDArray:
        """Return original class labels for every graph vertex."""
        if self.scores is None or self.classes is None:
            raise ValueError("Call fit before predict")
        return self.classes[(self.scores >= 0).astype(np.int8)]
