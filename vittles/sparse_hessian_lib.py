import autograd
import numpy as np
from .sensitivity_lib import _append_jvp

from copy import deepcopy
import scipy as sp
import scipy.sparse
from scipy.sparse import coo_matrix


class SparseBlockHessian():
    """Efficiently calculate block-sparse Hessians.

        The objective function is expected to be of the form

        .. math ::
            x = (x_1 , ... , x_G) \\textrm{ (or some permutation thereof)}

            f(x) = \sum_{g=1}^{G} f_g(x_g)

        Each :math:`x_g` is
        expected to have the same dimension.  Consequently, the Hessian
        matrix of ``f`` with respect to ``x``, is block diagonal with
        ``G`` blocks, up to a permutation of the order of ``x``.
        The purpose of this class is to efficiently calculate
        this Hessian when the block structure (i.e., the partition of ``x``)
        is known.

    """
    def __init__(self, objective_function, sparsity_array):
        """In terms of the class description, ``objective_function = f``,
        ``opt_par = x``, and ``sparsity_array`` describes
        the partition of ``x`` into :math:`(x_1, ..., x_G)`.

        Parameters
        ------------
        objective_function : `callable`
            An objective function of which to calculate a Hessian.   The
            argument should be

            - ``opt_par``: `numpy.ndarray` (N,) The optimization parameter.

        sparsity_array : `numpy.ndarray` (G, M)
            An array containing the indices of rows and columns of each block.
            The Hessian should contain ``G`` dense blocks, each of which
            is ``M`` by ``M``.  Each row of ``sparsity_array`` should contain
            the indices of the corresponding block.  There must be no repeated
            indices, and each block must be the same size.
        """
        self._fun = objective_function
        self._sparsity_array = sparsity_array
        self._num_blocks = self._sparsity_array.shape[0]
        self._block_size = self._sparsity_array.shape[1]

        if len(np.unique(sparsity_array)) != len(sparsity_array.flatten()):
            raise ValueError(
                'The indices in ``sparsity array`` must be unique.')

        self._f_grad = autograd.grad(self._fun, argnum=0)
        self._f_fwd_hess = _append_jvp(self._f_grad, num_base_args=1)

    def _hess_summed_term(self, opt_par, ib):
        """``ib`` is the index within the block.
        """
        v = np.zeros_like(opt_par)
        v[self._sparsity_array[:, ib]] = 1
        return self._f_fwd_hess(opt_par, v)

    def get_block_hessian(self, opt_par, print_every=0):
        """Get the block Hessian at ``opt_par`` and ``weights``.

        Parmeters
        ----------
        opt_par : `numpy.ndarray`
            The argument to ``objective_function`` at which to evaluate
            the Hessian matrix.
        print_every : `int`, optional.
            How often to display progress.  If ``0``, nothing is printed.

        Returns
        --------
        hessian : `scipy.sparse.coo_matrix` (N, N)
            The block-sparse Hessian given by and ``sparsity_array``.
        """
        opt_par = np.atleast_1d(opt_par)
        if opt_par.ndim != 1:
            raise ValueError('``opt_par`` must be a vector.')

        mat_vals = [] # These will be the entries of the Hessian
        mat_rows = [] # These will be the row indices
        mat_cols = [] # These will be the column indices

        for ib in range(self._block_size):
            if print_every > 0:
                if ib % print_every == 0:
                    print('Block index {} of {}.'.format(ib, self._block_size))
            hess_prod = self._hess_summed_term(opt_par, ib)
            for b in range(self._num_blocks):
                hess_inds = self._sparsity_array[b, :]
                mat_vals.extend(hess_prod[hess_inds])
                mat_rows.extend(hess_inds)
                mat_cols.extend(np.full(self._block_size, hess_inds[ib]))
        if print_every > 0:
            print('Done differentiating.')

        d = len(opt_par)
        h_sparse = coo_matrix((mat_vals, (mat_rows, mat_cols)), (d, d))
        return h_sparse

    def get_global_hessian(self, opt_par, global_inds=None, print_every=0):
        """Get the dense Hessian terms for the global parameters, which
        are, by default, indexed by any indices not in ``_sparsity_array``.
        """
        local_inds = np.hstack(self._sparsity_array)
        if global_inds is None:
            global_inds = np.setdiff1d(np.arange(len(opt_par)), local_inds)

        global_local_intersection = np.intersect1d(global_inds, local_inds)
        if len(global_local_intersection) > 0:
            raise ValueError(
                'The global and local indices must be disjoint.  {}'.format(
                    global_local_intersection))

        mat_vals = [] # These will be the entries of the Hessian
        mat_rows = [] # These will be the row indices
        mat_cols = [] # These will be the column indices

        v = np.zeros_like(opt_par)
        count = 0
        for ig in global_inds:
            if print_every > 0:
                if count % print_every == 0:
                    print('Global index {} of {}.'.format(
                        count, len(global_inds)))
            v[ig] = 1
            hess_row = self._f_fwd_hess(opt_par, v)
            for il in local_inds:
                mat_vals.append(hess_row[il])
                mat_cols.append(ig)
                mat_rows.append(il)

                mat_vals.append(hess_row[il])
                mat_cols.append(il)
                mat_rows.append(ig)

            for ig2 in global_inds:
                mat_vals.append(0.5 * hess_row[ig2])
                mat_cols.append(ig)
                mat_rows.append(ig2)

                mat_vals.append(0.5 * hess_row[ig2])
                mat_cols.append(ig2)
                mat_rows.append(ig)

            v[ig] = 0
            count += 1

        if print_every > 0:
            print('Done differentiating.')

        d = len(opt_par)
        h_sparse = coo_matrix((mat_vals, (mat_rows, mat_cols)), (d, d))
        return h_sparse

    def get_hessian(self, opt_par, print_every=0):
        local_hessian = self.get_block_hessian(opt_par, print_every=print_every)
        global_hessian = self.get_global_hessian(opt_par, print_every=print_every)
        return local_hessian + global_hessian
