import scipy as sp
import scipy.sparse
from scipy.linalg import cho_factor, cho_solve
import warnings


def get_dense_cholesky_solver(h, h_chol=None):
    if h_chol is None:
        h_chol = sp.linalg.cho_factor(h)
    def solve(v):
        return sp.linalg.cho_solve(h_chol, v)
    return solve


def get_sparse_cholesky_solver(h):
    if not sp.sparse.issparse(h):
        raise ValueError('`h` must be sparse.')
    return sp.sparse.linalg.factorized(h)


def get_cholesky_solver(h):
    if sp.sparse.issparse(h):
        return get_sparse_cholesky_solver(h)
    else:
        return get_dense_cholesky_solver(h)


def get_cg_solver(mat_times_vec, dim, cg_opts={}):
    linop = sp.sparse.linalg.LinearOperator((dim, dim), mat_times_vec)
    def solve(v):
        cg_result = sp.sparse.linalg.cg(linop, v, **cg_opts, atol='legacy')
        if cg_result[1] != 0:
            warnings.warn(
                'CG exited with error code {}'.format(cg_result[1]))
        return cg_result[0]
    return solve
