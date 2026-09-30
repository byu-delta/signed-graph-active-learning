r"""Graph Poisson learning for binary semi-supervised classification.

Poisson learning solves the regularized graph equation

.. math:: (L + \varepsilon I)u = f,

where ``f`` is supported on labeled vertices and centered to have zero sum.
See Calder, Cook, Thorpe, and Slepčev, *Poisson Learning: Graph Based
Semi-Supervised Learning at Very Low Label Rates* (ICML, 2020).

Centering makes the source orthogonal to the constant null vector of an
unsigned graph Laplacian and corrects for imbalance in the observed labels.
Regularization makes the finite system nonsingular, but retaining centering
preserves the intended equation as ``eps`` tends to zero. For a signed
Laplacian the constant vector need not be null, so centering is an explicit
modeling choice and should be held fixed when comparing experiments.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.sparse.linalg as splinalg


FloatArray: TypeAlias = NDArray[np.float64]


class PoissonLearning:
    """Binary Poisson-learning classifier.

    Parameters
    ----------
    L:
        Matrix-free graph Laplacian. It must also provide a
        ``diagonal_values`` attribute for Jacobi preconditioning.
    eps:
        Positive diagonal regularization in ``L + eps*I``. It removes
        Laplacian singularity and improves conditioning, but also damps the
        solution and can move the zero classification boundary. Report this
        value in experiments and test sensitivity when it is not negligible
        relative to the nonzero spectrum of ``L``.
    rtol:
        Relative residual tolerance passed to SciPy CG.
    max_iter:
        Maximum CG iterations.
    Attributes
    ----------
    scores:
        Fitted scalar decision score for each graph vertex.
    classes:
        Sorted original class labels used to decode score signs.
    cg_info:
        SciPy's convergence status for the most recent solve.
    """

    def __init__(
        self,
        L: splinalg.LinearOperator,
        *,
        eps: float = 1e-6,
        rtol: float = 1e-5,
        max_iter: int = 100_000,
    ) -> None:
        if not isinstance(L, splinalg.LinearOperator):
            raise TypeError("L must be a matrix-free scipy LinearOperator")
        if not hasattr(L, "diagonal_values"):
            raise TypeError("L must provide diagonal_values")
        self.L = L
        self.eps = float(eps)
        self.rtol = float(rtol)
        self.max_iter = int(max_iter)

        num_vertices = self.L.shape[0]
        self.regularized_laplacian_operator = splinalg.LinearOperator(
            shape=(num_vertices, num_vertices),
            matvec=lambda vector: self.L @ vector + self.eps * vector,
            rmatvec=lambda vector: self.L.T @ vector + self.eps * vector,
            dtype=float,
        )
        regularized_diagonal = self.L.diagonal_values + self.eps
        # # Used for Jacobi preconditioning of the conjugate gradient solver
        inverse_diagonal = 1.0 / np.maximum(np.abs(regularized_diagonal), 1e-10)
        self._jacobi_preconditioner_operator = splinalg.LinearOperator(
            shape=(num_vertices, num_vertices),
            matvec=lambda vector: inverse_diagonal * vector,
            rmatvec=lambda vector: inverse_diagonal * vector,
            dtype=float,
        )
        self.scores: FloatArray | None = None
        self.classes: NDArray | None = None
        self.cg_info: int | None = None
        self.converged: bool | None = None
        self.residual_norm: float | None = None

    def fit(
        self,
        train_ind: ArrayLike,
        train_labels: ArrayLike,
        *,
        x0: ArrayLike | None = None,
    ) -> PoissonLearning:
        """Fit the Poisson-learning model.

        Center the supplied -1/+1 labels, place them in the source vector at the
        labeled vertices, and solve the Poisson equation. x0 can provide a
        previous solution as the solver's starting point.
        """
        train_ind = np.asarray(train_ind, dtype=int).ravel()
        train_labels = np.asarray(train_labels).ravel()
        classes = np.unique(train_labels)
        source = np.zeros(self.L.shape[0], dtype=float)
        source[train_ind] = train_labels - train_labels.mean()
        return self._solve_prepared(source, classes, x0=x0)

    def _solve_prepared(
        self,
        source: ArrayLike,
        classes: ArrayLike,
        *,
        x0: ArrayLike | None = None,
    ) -> PoissonLearning:
        """Solve a prepared Poisson problem and return ``self``.
        This internal entry point avoids rebuilding a graph-independent source
        vector when an active-learning loop compares graph variants. Public
        callers should normally use :meth:`fit`.
        """

        source = np.asarray(source, dtype=float).ravel()
        initial = None if x0 is None else np.asarray(x0, dtype=float).ravel()
        scores, info = splinalg.cg(
            self.regularized_laplacian_operator,
            source,
            x0=initial,
            M=self._jacobi_preconditioner_operator,
            rtol=self.rtol,
            atol=0.0,
            maxiter=self.max_iter,
        )
        self.scores = np.asarray(scores, dtype=float)
        self.classes = classes
        self.cg_info = int(info)
        self.converged = info == 0
        self.residual_norm = float(
            np.linalg.norm(source - self.regularized_laplacian_operator @ self.scores)
        )
        return self

    def predict(self, *, center_scores: bool = True) -> NDArray:
        """Convert fitted scores into class labels.

        Negative scores are assigned to the first class, and scores greater than or
        equal to zero are assigned to the second class. If ``center_scores=True``,
        the mean score is subtracted before classification to remove a global shift
        in the scores that could move the classification boundary. This changes the
        classification threshold but does not modify the stored scores.
        """
        if self.scores is None or self.classes is None:
            raise ValueError("Call fit before predict")

        decision_scores = self.scores
        if center_scores:
            decision_scores = decision_scores - decision_scores.mean()
        return self.classes[(decision_scores >= 0).astype(int)]
