import scipy as sp
import scipy.sparse
from scipy.linalg import cho_factor, cho_solve
import warnings


def get_dense_cholesky_solver(h, h_chol=None):
    """Return a function solving :math:`h^{-1} v` with a dense
    Cholesky factorization.

    Parameters
    -------------
    h : `numpy.ndarray`
        A dense symmetric positive definite matrix.
    h_chol : A Cholesky factorization
        Optional, a Cholesky factorization created with `sp.linalg.cho_factor`.
        If this is specified, the argument `h` is ignored.  If `None` (the
        default), the factoriztion of `h` is calculated.

    Returns
    --------------
    solve : `callable`
        A function of a single vector argument, `v` that returns
        :math:`h^{-1} v`.
    """
    if h_chol is None:
        h_chol = sp.linalg.cho_factor(h)
    def solve(v):
        return sp.linalg.cho_solve(h_chol, v)
    return solve


def get_sparse_cholesky_solver(h):
    """Return a function solving :math:`h^{-1} v` for a sparse `h`.

    Parameters
    -------------
    h : A sparse invertible matrix.

    Returns
    --------------
    solve : `callable`
        A function of a single vector argument, `v` that returns
        :math:`h^{-1} v`.
    """
    if not sp.sparse.issparse(h):
        raise ValueError('`h` must be sparse.')
    return sp.sparse.linalg.factorized(h)


def get_cholesky_solver(h):
    """Return a function solving :math:`h^{-1} v` for a matrix `h`.

    Parameters
    -------------
    h : A dense or sparse matrix.

    Returns
    --------------
    solve : `callable`
        A function of a single vector argument, `v` that returns
        :math:`h^{-1} v`.
    """
    if sp.sparse.issparse(h):
        return get_sparse_cholesky_solver(h)
    else:
        return get_dense_cholesky_solver(h)


def get_cg_solver(mat_times_vec, dim, cg_opts={}):
    """Return a function solving :math:`h^{-1} v` for a matrix `h` using
    conjugate gradient.

    Parameters
    -------------
    mat_times_vec : `callable`
        A function of a single vector argument, `v` that returns the product
        `h v` for some invertible matrix `h`.
    dim : `int`
        The dimension of the vector `v`.
    cg_opts : `dict`
        Optional, a dictionary of named arguments for the solver
        `sp.sparse.linalg.cg`.

    Returns
    ---------
    solve : `callable`
        A function of a single vector argument, `v` that returns the
        conjugate gradient approximation to :math:`h^{-1} v`.
    """
    linop = sp.sparse.linalg.LinearOperator((dim, dim), mat_times_vec)
    def solve(v):
        cg_result = sp.sparse.linalg.cg(linop, v, **cg_opts, atol='legacy')
        if cg_result[1] != 0:
            warnings.warn(
                'CG exited with error code {}'.format(cg_result[1]))
        return cg_result[0]
    return solve
