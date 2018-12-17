#!/usr/bin/env python3

import autograd
import autograd.numpy as np
from autograd.test_util import check_grads
import itertools
from numpy.testing import assert_array_almost_equal
import vittles
import scipy as sp
from test_utils import QuadraticModel, captured_output
import unittest


class TestPreconditionedFunction(unittest.TestCase):
    def test_preconditioned_function(self):
        model = QuadraticModel(dim=3)

        # Define a function of theta alone.
        f = vittles.Functor(model.get_objective, argnums=0)
        f.cache_args(None, model.lam)
        f_grad = autograd.grad(f)
        f_hessian = autograd.hessian(f)

        f_c = vittles.PreconditionedFunction(f)
        f_c_grad = autograd.grad(f_c)
        f_c_hessian = autograd.hessian(f_c)

        dim = model.theta_pattern.shape()[0]
        theta = np.arange(0, dim) / 5.0

        # Raise an error if we have not set the preconditioner.
        with self.assertRaises(ValueError):
            f_c(theta)

        def test_f_c_values(a):
            a_inv = np.linalg.inv(a)
            assert_array_almost_equal(
                a_inv @ theta, f_c.precondition(theta))
            assert_array_almost_equal(
                theta, f_c.unprecondition(a_inv @ theta))
            assert_array_almost_equal(f(theta), f_c(a_inv @ theta))
            assert_array_almost_equal(
                a @ f_grad(theta), f_c_grad(a_inv @ theta))
            assert_array_almost_equal(
                a @ f_hessian(theta) @ a.T, f_c_hessian(a_inv @ theta))
            assert_array_almost_equal(a, f_c.get_preconditioner())
            assert_array_almost_equal(a_inv, f_c.get_preconditioner_inv())

        # Test with an ordinary matrix.
        a = 2 * np.eye(dim) + np.full((dim, dim), 0.1)
        f_c.set_preconditioner(a)
        test_f_c_values(a)

        f_c.set_preconditioner(a, np.linalg.inv(a))
        test_f_c_values(a)

        # Test with the Hessian.
        hess = f_hessian(theta)

        for ev_min in [None, 0.01]:
            for ev_max in [None, 10.0]:
                h_inv_sqrt, h_sqrt, h = \
                    vittles.optimization_lib._get_sym_matrix_inv_sqrt(
                        hess, ev_min=ev_min, ev_max=ev_max)

                f_c.set_preconditioner_with_hessian(
                    x=theta, ev_min=ev_min, ev_max=ev_max)
                test_f_c_values(h_inv_sqrt)

                f_c.set_preconditioner_with_hessian(
                    hessian=h, ev_min=ev_min, ev_max=ev_max)
                test_f_c_values(h_inv_sqrt)

        # Check that optimizing the two functions is equivalent.
        opt_result = sp.optimize.minimize(
            fun=f,
            jac=f_grad,
            hess=f_hessian,
            x0=theta,
            method='Newton-CG',
            options={'maxiter': 100, 'disp': False })

        theta_c = f_c.precondition(theta)
        opt_result_c = sp.optimize.minimize(
            fun=f_c,
            jac=f_c_grad,
            hess=f_c_hessian,
            x0=theta_c,
            method='Newton-CG',
            options={'maxiter': 100, 'disp': False })

        assert_array_almost_equal(
            opt_result.x, f_c.unprecondition(opt_result_c.x))

        # Check that at the optimum, where the gradient is zero,
        # preconditioning with the Hessian makes the Hessian of f_c
        # into the identity.
        theta_opt = opt_result.x
        theta_c_opt = opt_result_c.x
        f_c.set_preconditioner_with_hessian(x=theta_opt)
        assert_array_almost_equal(np.eye(dim), f_c_hessian(theta_c_opt))





    def _test_matrix_sqrt(self, mat):
        id_mat = np.eye(mat.shape[0])
        eig_vals = np.linalg.eigvals(mat)
        ev_min = np.min(eig_vals)
        ev_max = np.max(eig_vals)
        ev0 = np.real(ev_min + (ev_max - ev_min) / 3)
        ev1 = np.real(ev_min + 2 * (ev_max - ev_min) / 3)

        for test_ev_min in [None, ev0]:
            for test_ev_max in [None, ev1]:
                h_inv_sqrt, h_sqrt, h = \
                    vittles.optimization_lib._get_sym_matrix_inv_sqrt(
                        mat, test_ev_min, test_ev_max)
                assert_array_almost_equal(id_mat, h_inv_sqrt @ h_sqrt)
                assert_array_almost_equal(
                    id_mat, h_inv_sqrt @ h @ h_inv_sqrt.T)
                eig_vals_test = np.linalg.eigvals(h)
                if test_ev_min is not None:
                    self.assertTrue(np.min(eig_vals_test) >=
                                    test_ev_min - 1e-8)
                else:
                    assert_array_almost_equal(ev_min, np.min(eig_vals_test))
                if test_ev_max is not None:
                    self.assertTrue(np.max(eig_vals_test) <=
                                    test_ev_max + 1e-8)
                else:
                    assert_array_almost_equal(ev_max, np.max(eig_vals_test))

    def test_matrix_sqrt(self):
        dim = 5
        mat = dim * np.eye(dim)
        vec = np.random.random(dim)
        mat = mat + np.outer(vec, vec)
        self._test_matrix_sqrt(mat)


class TestOptimizationObjective(unittest.TestCase):
    def test_optimization_objective(self):
        def objective_fun(x):
            return np.sum(x ** 4)

        x0 = np.random.random(5)
        obj = vittles.OptimizationObjective(
            objective_fun, print_every=0)
        assert_array_almost_equal(objective_fun(x0), obj.f(x0))
        assert_array_almost_equal(
            autograd.grad(objective_fun)(x0), obj.grad(x0))
        assert_array_almost_equal(
            autograd.hessian(objective_fun)(x0), obj.hessian(x0))
        assert_array_almost_equal(
            autograd.hessian_vector_product(objective_fun)(
                x0, x0),
            obj.hessian_vector_product(x0, x0))

        def test_print_and_log(num_evals, expected_prints, expected_logs):
            with captured_output() as (out, err):
                init_num_iterations = obj.num_iterations()
                for iter in range(num_evals):
                    # Funtion evaluations should be printed and logged.
                    obj.f(x0)

                    # Derivatives should not count towards printing or logging.
                    obj.grad(x0)
                    obj.hessian(x0)
                    obj.hessian_vector_product(x0, x0)

            lines = out.getvalue().splitlines()
            self.assertEqual(init_num_iterations + num_evals,
                             obj.num_iterations())
            self.assertEqual(len(lines), expected_prints)
            self.assertEqual(len(obj.optimization_log), expected_logs)

        # Test reset.
        obj.set_print_every(1)
        obj.set_log_every(1)
        obj.reset()
        test_print_and_log(num_evals=1, expected_prints=1, expected_logs=1)
        obj.reset()
        test_print_and_log(num_evals=1, expected_prints=1, expected_logs=1)

        # Test that the first iteration prints and logs no matter what.
        obj.set_print_every(2)
        obj.set_log_every(2)
        obj.reset()
        test_print_and_log(num_evals=1, expected_prints=1, expected_logs=1)
        test_print_and_log(num_evals=1, expected_prints=0, expected_logs=1)

        # Test combinations of print and log.
        for print_every, log_every in itertools.product([0, 1], [0, 1]):
            obj.set_print_every(print_every)
            obj.set_log_every(log_every)
            obj.reset()
            test_print_and_log(
                num_evals=3,
                expected_prints=3 * print_every,
                expected_logs=3 * log_every)

        for print_every, log_every in itertools.product([0, 3], [0, 3]):
            obj.set_print_every(print_every)
            obj.set_log_every(log_every)
            obj.reset()
            test_print_and_log(
                num_evals=6,
                expected_prints=(print_every != 0) * 2,
                expected_logs=(log_every != 0) * 2)

        # Test reset only printing or logging.
        obj.set_print_every(2)
        obj.set_log_every(1)

        obj.reset()
        test_print_and_log(num_evals=1, expected_prints=1, expected_logs=1)
        test_print_and_log(num_evals=1, expected_prints=0, expected_logs=2)

        obj.reset()
        test_print_and_log(num_evals=1, expected_prints=1, expected_logs=1)
        obj.reset_iteration_count()
        test_print_and_log(num_evals=1, expected_prints=1, expected_logs=2)

        obj.reset()
        test_print_and_log(num_evals=1, expected_prints=1, expected_logs=1)
        obj.reset_log()
        test_print_and_log(num_evals=1, expected_prints=0, expected_logs=1)


if __name__ == '__main__':
    unittest.main()
