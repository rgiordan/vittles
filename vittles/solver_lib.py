import scipy as sp
import scipy.sparse
from scipy.linalg import cho_factor, cho_solve
import warnings


class SystemSolver:
    """A class to provide a common interface for solving :math:`H^{-1} g`.
    """
    def __init__(self, h, method):
        """
        Parameters
        -------------
        h : `numpy.ndarray` or `scipy.sparse` matrix
            The "Hessian" matrix for sensitivity analysis.
        method : {'factorization', 'cg'}
            How to solve the system.  `factorization` uses a Cholesky decomposition,
            and `cg` uses conjugate gradient.
        """
        self.__valid_methods = [ 'factorization', 'cg' ]
        if method not in self.__valid_methods:
            raise ValueError('method must be one of {}'.format(self.__valid_methods))
        self._method = method
        self.set_h(h)
        self.set_cg_options({})

    def set_h(self, h):
        """Set the Hessian matrix.
        """
        self._h = h
        self._sparse = sp.sparse.issparse(h)
        if self._method == 'factorization':
            if self._sparse:
                self._solve_h = sp.sparse.linalg.factorized(self._h)
            else:
                self._h_chol = sp.linalg.cho_factor(self._h)
        elif self._method == 'cg':
            self._linop = sp.sparse.linalg.aslinearoperator(self._h)
        else:
            raise ValueError('Unknown method {}'.format(self._method))

    def set_cg_options(self, cg_opts):
        """Set the cg options as a dictionary.

        Parameters
        -------------
        cg_opts : `dict`
            A dictionary of keyword options to be passed to
            `scipy.sparse.linalg.cg`.  If ``method`` is not ``cg``, these will be
            ignored.
        """
        self._cg_opts = cg_opts

    def solve(self, v):
        """Solve the linear system :math:`H{-1} v`.

        Parameters
        ------------
        v : `numpy.ndarray`
            A numpy array.

        Returns
        --------
        h_inv_v : `numpy.ndarray`
            The value of :math:`H{-1} v`.
        """
        if self._method == 'factorization':
            if self._sparse:
                return self._solve_h(v)
            else:
                return sp.linalg.cho_solve(self._h_chol, v)
        elif self._method == 'cg':
            cg_result = sp.sparse.linalg.cg(self._linop, v, **self._cg_opts)
            if cg_result[1] != 0:
                warnings.warn('CG exited with error code {}'.format(cg_result[1]))
            return cg_result[0]

        else:
            raise ValueError('Unknown method {}'.format(self._method))
