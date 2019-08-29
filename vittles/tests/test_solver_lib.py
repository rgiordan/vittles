#!/usr/bin/env python3

import numpy as np
from numpy.testing import assert_array_almost_equal
import scipy as sp
import unittest
from vittles import solver_lib


class TestSystemSolver(unittest.TestCase):
    def test_solver(self):
        np.random.seed(101)
        d = 10
        h_dense = np.random.random((d, d))
        h_dense = h_dense + h_dense.T + d * np.eye(d)
        h_sparse = sp.sparse.csc_matrix(h_dense)
        v = np.random.random(d)
        h_inv_v = np.linalg.solve(h_dense, v)

        assert_array_almost_equal(
            solver_lib.get_dense_cholesky_solver(h_dense)(v), h_inv_v)
        assert_array_almost_equal(
            solver_lib.get_cholesky_solver(h_dense)(v), h_inv_v)
        h_chol = sp.linalg.cho_factor(h_dense)
        assert_array_almost_equal(
            solver_lib.get_dense_cholesky_solver(None, h_chol)(v), h_inv_v)

        assert_array_almost_equal(
            solver_lib.get_cholesky_solver(h_sparse)(v), h_inv_v)
        assert_array_almost_equal(
            solver_lib.get_sparse_cholesky_solver(h_sparse)(v), h_inv_v)

        assert_array_almost_equal(
            solver_lib.get_cg_solver(lambda v: h_dense @ v, d)(v),
            h_inv_v)
        assert_array_almost_equal(
            solver_lib.get_cg_solver(lambda v: h_sparse @ v, d)(v), h_inv_v)

        # With only one iteration, the CG should fail and raise a warning.
        h_solver = solver_lib.get_cg_solver(
            lambda v: h_sparse @ v, d, cg_opts={'maxiter': 1})
        with self.assertWarns(UserWarning):
            h_solver(v)


if __name__ == '__main__':
    unittest.main()
