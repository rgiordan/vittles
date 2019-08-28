#!/usr/bin/env python3

import numpy as np
from numpy.testing import assert_array_almost_equal
#import paragami
import scipy as sp
#from test_utils import QuadraticModel
import unittest
#import time
#import warnings

# import vittles
# from vittles import sensitivity_lib
from vittles import solver_lib
# from vittles.sensitivity_lib import \
#     _append_jvp, _evaluate_term_fwd, DerivativeTerm, \
#     ReverseModeDerivativeArray, ForwardModeDerivativeArray


class TestSystemSolver(unittest.TestCase):
    def test_solver(self):
        np.random.seed(101)
        d = 10
        h_dense = np.random.random((d, d))
        h_dense = h_dense + h_dense.T + d * np.eye(d)
        h_sparse = sp.sparse.csc_matrix(h_dense)
        v = np.random.random(d)
        h_inv_v = np.linalg.solve(h_dense, v)

        for h in [h_dense, h_sparse]:
            for method in ['factorization', 'cg']:
                h_solver = solver_lib.SystemSolver(h, method)
                assert_array_almost_equal(h_solver.solve(v), h_inv_v)

        h_solver = solver_lib.SystemSolver(h_dense, 'cg')
        h_solver.set_cg_options({'maxiter': 1})
        with self.assertWarns(UserWarning):
            # With only one iteration, the CG should fail and raise a warning.
            h_solver.solve(v)


if __name__ == '__main__':
    unittest.main()
