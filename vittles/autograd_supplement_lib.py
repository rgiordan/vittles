# Define some forward-diff functions for np.linalg that are currently excluded
# from autogra.d
#
# Most of these are copied with minimal modification from
# https://github.com/HIPS/autograd/blob/65c21e2/autograd/numpy/linalg.py

import autograd
import autograd.numpy as np
import autograd.scipy as sp
from autograd.core import primitive, defvjp, defjvp

from autograd.numpy.linalg import slogdet, solve, inv
from functools import partial

# transpose by swapping last two dimensions
def T(x): return np.swapaxes(x, -1, -2)

def inv_jvp(g, ans, x):
    dot = np.dot if ans.ndim == 2 else partial(np.einsum, '...ij,...jk->...ik')
    return -dot(dot(ans, g), ans)

defjvp(inv, inv_jvp)

def jvp_solve(argnum, g, ans, a, b):
    def broadcast_matmul(a, b):
        return \
            np.matmul(a, b) if b.ndim == a.ndim \
            else np.matmul(a, b[..., None])[..., 0]
    if argnum == 0:
        foo = np.linalg.solve(a, g)
        return -broadcast_matmul(np.linalg.solve(a, g), ans)
    else:
        return np.linalg.solve(a, g)

defjvp(solve, partial(jvp_solve, 0), partial(jvp_solve, 1))


def slogdet_jvp(g, ans, x):
    # Due to https://github.com/HIPS/autograd/issues/115
    # and https://github.com/HIPS/autograd/blob/65c21e/tests/test_numpy.py#L302
    # it does not seem easy to take the trace of the last two dimensions of
    # a multi-dimensional array at this time.
    if len(x.shape) > 2:
        raise ValueError('JVP is only supported for 2d input.')
    return 0, np.trace(np.linalg.solve(x.T, g.T))

defjvp(slogdet, slogdet_jvp)


def get_sparse_product(z_mat):
    """
    Return an autograd-compatible function that calculates
    ``z_mat @ a`` and ``z_mat.T @ a`` when ``z_mat`` is a sparse matrix.

    Parameters
    ------------
    z_mat: A 2d matrix
        The matrix by which to multiply.  The matrix can be dense, but the only
        reason to use ``get_sparse_product`` is with a sparse matrix since
        dense matrix multiplication is supported natively by ``autograd``.

    Returns
    -----------
    z_mult:
        A function such that ``z_mult(b) = z_mat @ b``.
    zt_mult:
        A function such that ``zt_mult(b) = z_mat.T @ b``.
    Unlike standard sparse matrix multiplication, ``z_mult`` and ``zt_mult``
    can be used with ``autograd``.
    """

    if z_mat.ndim != 2:
        raise ValueError(
            'get_sparse_product can only be used with 2d arrays.')

    @primitive
    def z_mult(b):
        return z_mat @ b

    @primitive
    def zt_mult(b):
        return z_mat.T @ b

    def z_mult_jvp(g, ans, b):
        return z_mult(g) # z_mat @ g

    defjvp(z_mult, z_mult_jvp)

    def z_mult_vjp(ans, b):
        def vjp(g):
            return zt_mult(g).T # g.T @ z_mat
        return vjp
    defvjp(z_mult, z_mult_vjp)

    def zt_mult_jvp(g, ans, b):
        return zt_mult(g) # z_mat.T @ g
    defjvp(zt_mult, zt_mult_jvp)

    def zt_mult_vjp(ans, b):
        def vjp(g):
            return z_mult(g).T # g.T @ z_mat.T
        return vjp
    defvjp(zt_mult, zt_mult_vjp)

    return z_mult, zt_mult
