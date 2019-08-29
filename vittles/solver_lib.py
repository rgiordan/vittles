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


# # TODO: deprecate this, and just pass in a function rather than a class.
# class SystemSolver:
#     """A class to provide a common interface for solving :math:`H^{-1} g`.
#     """
#     def __init__(self, h, method):
#         """
#         Parameters
#         -------------
#         h : `numpy.ndarray` or `scipy.sparse` matrix
#             The "Hessian" matrix for sensitivity analysis.
#         method : {'factorization', 'cg'}
#             How to solve the system.  `factorization` uses a Cholesky
#             decomposition, and `cg` uses conjugate gradient.
#         """
#         self.__valid_methods = [ 'factorization', 'cg' ]
#         if method not in self.__valid_methods:
#             raise ValueError('method must be one of {}'.format(
#                 self.__valid_methods))
#         self._method = method
#         self.set_h(h)
#         self.set_cg_options({})
#
#     def set_h(self, h):
#         """Set the Hessian matrix.
#         """
#         self._h = h
#         self._sparse = sp.sparse.issparse(h)
#         if self._method == 'factorization':
#             if self._sparse:
#                 self._solve_h = sp.sparse.linalg.factorized(self._h)
#             else:
#                 self._h_chol = sp.linalg.cho_factor(self._h)
#         elif self._method == 'cg':
#             self._linop = sp.sparse.linalg.aslinearoperator(self._h)
#         else:
#             raise ValueError('Unknown method {}'.format(self._method))
#
#     def set_cg_options(self, cg_opts):
#         """Set the cg options as a dictionary.
#
#         Parameters
#         -------------
#         cg_opts : `dict`
#             A dictionary of keyword options to be passed to
#             `scipy.sparse.linalg.cg`.  If ``method`` is not ``cg``, these
#             will be ignored.
#         """
#         self._cg_opts = cg_opts
#
#     def solve(self, v):
#         """Solve the linear system :math:`H{-1} v`.
#
#         Parameters
#         ------------
#         v : `numpy.ndarray`
#             A numpy array.
#
#         Returns
#         --------
#         h_inv_v : `numpy.ndarray`
#             The value of :math:`H{-1} v`.
#         """
#         if self._method == 'factorization':
#             if self._sparse:
#                 return self._solve_h(v)
#             else:
#                 return sp.linalg.cho_solve(self._h_chol, v)
#         elif self._method == 'cg':
#             cg_result = sp.sparse.linalg.cg(
#                 self._linop, v, **self._cg_opts, atol='legacy')
#             if cg_result[1] != 0:
#                 warnings.warn(
#                     'CG exited with error code {}'.format(cg_result[1]))
#             return cg_result[0]
#
#         else:
#             raise ValueError('Unknown method {}'.format(self._method))
