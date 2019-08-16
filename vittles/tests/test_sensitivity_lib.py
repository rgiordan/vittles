#!/usr/bin/env python3

import autograd
import autograd.numpy as np

from copy import deepcopy
import itertools
from numpy.testing import assert_array_almost_equal
import paragami
import scipy as sp
from test_utils import QuadraticModel
import unittest
import warnings

import vittles
from vittles import sensitivity_lib
from vittles import solver_lib
from vittles.sensitivity_lib import \
    _append_jvp, _evaluate_term_fwd, DerivativeTerm, \
    ReverseModeDerivativeArray

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


class TestReverseModeDerivativeArray(unittest.TestCase):

    def test_evaluate_directional_derivative(self):
        def f(x1, x2):
            return np.sin(np.sum(x1) + np.sum(x2))

        g = autograd.grad(f, argnum=0)
        dim1 = 3
        dim2 = 4

        x1 = np.random.random(dim1)
        x2 = np.random.random(dim2)

        max_order1 = 2
        max_order2 = 2
        deriv_array = ReverseModeDerivativeArray(
            fun=g, order1=max_order1, order2=max_order2)
        deriv_array.set_evaluation_location(x1, x2)

        self.assertEqual(
            max_order1 + 1,
            len(deriv_array._eval_deriv_arrays))
        for i in range(max_order1 + 1):
            self.assertEqual(
                max_order2 + 1,
                len(deriv_array._eval_deriv_arrays[i]))

        # Check the first couple deriv_arrays by hand.
        assert_array_almost_equal(
            g(x1, x2),
            deriv_array.deriv_arrays[0][0])

        assert_array_almost_equal(
            autograd.jacobian(g, argnum=0)(x1, x2),
            deriv_array.deriv_arrays[1][0])

        assert_array_almost_equal(
            autograd.jacobian(g, argnum=1)(x1, x2),
            deriv_array.deriv_arrays[0][1])

        # Check eval_directional_derivative.
        dx1s = [ np.random.random(dim1) for _ in range(2) ]
        dx2s = [ np.random.random(dim2) for _ in range(2) ]

        deriv_array.eval_directional_derivative(x1, x2, dx1s, dx2s)

class TestAppendJVP(unittest.TestCase):
    def test_append_jvp(self):
        eta_is_free = True
        eps_is_free = True
        model = QuadraticModel(dim=3)

        objective = model.get_flat_objective(eta_is_free, eps_is_free)
        eta0, eps0 = model.get_default_flat_values(eta_is_free, eps_is_free)

        # Use the reverse mode derivatives as ground truth.
        obj_eta_grad = autograd.grad(objective, argnum=0)
        obj_eps_grad = autograd.grad(objective, argnum=1)
        obj_eta_hessian = autograd.hessian(objective, argnum=0)
        obj_eps_hessian = autograd.hessian(objective, argnum=1)
        get_dobj_deta_deps = autograd.jacobian(
            autograd.jacobian(objective, argnum=0), argnum=1)

        dobj_deta = sensitivity_lib._append_jvp(
            objective, num_base_args=2, argnum=0)
        d2obj_deta_deta = sensitivity_lib._append_jvp(
            dobj_deta, num_base_args=2, argnum=0)

        v1 = np.random.random(len(eta0))
        v2 = np.random.random(len(eta0))
        v3 = np.random.random(len(eta0))
        w1 = np.random.random(len(eps0))
        w2 = np.random.random(len(eps0))
        w3 = np.random.random(len(eps0))

        hess0 = obj_eta_hessian(eta0, eps0)

        # Check the first argument
        assert_array_almost_equal(
            np.einsum('i,i', obj_eta_grad(eta0, eps0), v1),
            dobj_deta(eta0, eps0, v1))
        assert_array_almost_equal(
            np.einsum('ij,i,j', obj_eta_hessian(eta0, eps0), v1, v2),
            d2obj_deta_deta(eta0, eps0, v1, v2))

        # Check the second argument
        dobj_deps = sensitivity_lib._append_jvp(
            objective, num_base_args=2, argnum=1)
        d2obj_deps_deps = sensitivity_lib._append_jvp(
            dobj_deps, num_base_args=2, argnum=1)

        assert_array_almost_equal(
            np.einsum('i,i', obj_eps_grad(eta0, eps0), w1),
            dobj_deps(eta0, eps0, w1))

        assert_array_almost_equal(
            np.einsum('ij,i,j', obj_eps_hessian(eta0, eps0), w1, w2),
            d2obj_deps_deps(eta0, eps0, w1, w2))

        # Check mixed arguments
        d2obj_deps_deta = sensitivity_lib._append_jvp(
            dobj_deps, num_base_args=2, argnum=0)
        d2obj_deta_deps = sensitivity_lib._append_jvp(
            dobj_deta, num_base_args=2, argnum=1)

        assert_array_almost_equal(
            d2obj_deps_deta(eta0, eps0, v1, w1),
            d2obj_deta_deps(eta0, eps0, w1, v1))

        assert_array_almost_equal(
            np.einsum('ij,i,j', get_dobj_deta_deps(eta0, eps0), v1, w1),
            d2obj_deps_deta(eta0, eps0, v1, w1))

        # Check derivatives of vectors.
        dg_deta = sensitivity_lib._append_jvp(
            obj_eta_grad, num_base_args=2, argnum=0)

        assert_array_almost_equal(
            hess0 @ v1, dg_deta(eta0, eps0, v1))


class TestDerivativeTerm(unittest.TestCase):
    def test_generate_two_term_fwd_derivative_array(self):
        model = QuadraticModel(dim=3)

        eta0, eps0 = model.get_default_flat_values(True, True)
        objective = model.get_flat_objective(True, True)
        obj_eta_grad = autograd.grad(objective, argnum=0)

        max_order1 = 2
        max_order2 = 3
        eval_g_derivs = \
            sensitivity_lib._generate_two_term_fwd_derivative_array(
                obj_eta_grad, order1=max_order1, order2=max_order2)

        # n^th order derivatives require n + 1 entries, because the
        # 0-th order is included as well.
        self.assertEqual(max_order1 + 1, len(eval_g_derivs))
        for i in range(len(eval_g_derivs)):
            self.assertEqual(max_order2 + 1, len(eval_g_derivs[i]))

        # Use autodiff's forward mode to check.
        dg_deta = sensitivity_lib._append_jvp(
            obj_eta_grad, num_base_args=2, argnum=0)
        dg_deps = sensitivity_lib._append_jvp(
            obj_eta_grad, num_base_args=2, argnum=1)
        d2g_deta_deta = sensitivity_lib._append_jvp(
            dg_deta, num_base_args=2, argnum=0)
        d2g_deta_deps = sensitivity_lib._append_jvp(
            dg_deta, num_base_args=2, argnum=1)
        d2g_deps_deta = sensitivity_lib._append_jvp(
            dg_deps, num_base_args=2, argnum=0)
        d2g_deps_deps = sensitivity_lib._append_jvp(
            dg_deps, num_base_args=2, argnum=1)

        # Directions in eta
        v1 = np.random.random(len(eta0))
        v2 = np.random.random(len(eta0))

        # Directions in eps
        w1 = np.random.random(len(eps0))
        w2 = np.random.random(len(eps0))

        # Test the array entries
        assert_array_almost_equal(
            obj_eta_grad(eta0, eps0),
            eval_g_derivs[0][0](eta0, eps0))

        assert_array_almost_equal(
            dg_deta(eta0, eps0, v1),
            eval_g_derivs[1][0](eta0, eps0, v1))

        assert_array_almost_equal(
            dg_deps(eta0, eps0, w1),
            eval_g_derivs[0][1](eta0, eps0, w1))

        assert_array_almost_equal(
            d2g_deta_deta(eta0, eps0, v1, v2),
            eval_g_derivs[2][0](eta0, eps0, v1, v2))

        assert_array_almost_equal(
            d2g_deta_deps(eta0, eps0, v1, w1),
            eval_g_derivs[1][1](eta0, eps0, v1, w1))

        assert_array_almost_equal(
            d2g_deps_deps(eta0, eps0, w1, w2),
            eval_g_derivs[0][2](eta0, eps0, w1, w2))

    def test_differentiate(self):
        dterm = DerivativeTerm(
            eps_order=2,
            eta_orders=[1, 0, 0],
            prefactor=1.7)

        dterms = dterm.differentiate()

        # We expect three terms:
        # Order: 4	1.7 * eta[1, 0, 0, 0] * eps[3]
        # Order: 4	1.7 * eta[2, 0, 0, 0] * eps[2]
        # Order: 4	1.7 * eta[0, 1, 0, 0] * eps[2]
        self.assertEqual(3, len(dterms))
        term1_found = False
        term2_found = False
        term3_found = False
        for term in dterms:
            if term.check_similarity(
                DerivativeTerm(3, [1, 0, 0, 0], 1.7)):
                term1_found = True
            elif term.check_similarity(
                DerivativeTerm(2, [2, 0, 0, 0], 1.7)):
                term2_found = True
            elif term.check_similarity(
                DerivativeTerm(2, [0, 1, 0, 0], 1.7)):
                term3_found = True
        self.assertTrue(term1_found and term2_found and term3_found)


    def test_consolidate_terms(self):
        dterms = [
            DerivativeTerm(
                eps_order=2,
                eta_orders=[0, 0],
                prefactor=1.5),
            DerivativeTerm(
                eps_order=1,
                eta_orders=[1, 0],
                prefactor=2),
            DerivativeTerm(
                eps_order=1,
                eta_orders=[1, 0],
                prefactor=3) ]

        dterms_combined = sensitivity_lib._consolidate_terms(dterms)
        self.assertEqual(3, len(dterms))
        self.assertEqual(2, len(dterms_combined))

    def test_evaluate_derivative_term(self):
        model = QuadraticModel(dim=3)

        eta0, eps0 = model.get_default_flat_values(True, True)
        objective = model.get_flat_objective(True, True)
        get_true_optimal_flat_theta = \
            model.get_flat_true_optimal_theta(True, True)

        obj_eta_grad = autograd.grad(objective, argnum=0)
        eval_g_derivs = \
            sensitivity_lib._generate_two_term_fwd_derivative_array(
                obj_eta_grad, order1=2, order2=2)

        eps1 = eps0 + 1e-1
        eta1 = get_true_optimal_flat_theta(eps1)
        deps = eps1 - eps0

        # Get true derivatives.
        true_deta_deps = autograd.jacobian(get_true_optimal_flat_theta)

        # Use autodiff's forward mode to check.
        dg_deta = sensitivity_lib._append_jvp(
            obj_eta_grad, num_base_args=2, argnum=0)
        dg_deps = sensitivity_lib._append_jvp(
            obj_eta_grad, num_base_args=2, argnum=1)
        d2g_deta_deps = sensitivity_lib._append_jvp(
            dg_deta, num_base_args=2, argnum=1)

        dterm = DerivativeTerm(
            eps_order=1,
            eta_orders=[1, 0],
            prefactor=1.5)

        eta_derivs = [ true_deta_deps(eps0) @ deps ]
        assert_array_almost_equal(
            dterm.prefactor * d2g_deta_deps(
                eta0, eps0, true_deta_deps(eps0) @  deps, deps),
            _evaluate_term_fwd(
                dterm, eta0, eps0, deps, eta_derivs, eval_g_derivs))

        dterms1 = sensitivity_lib._get_taylor_base_terms()
        deriv_terms = [ true_deta_deps(eps0) @ deps ]
        assert_array_almost_equal(
            dg_deps(eta0, eps0, deps),
            _evaluate_term_fwd(
                dterms1[0], eta0, eps0, deps, deriv_terms, eval_g_derivs))


class TestHyperparameterSensitivityLinearApproximation(unittest.TestCase):
    def _test_linear_approximation(self, dim,
                                   theta_free, lambda_free,
                                   use_hessian_at_opt,
                                   use_cross_hessian_at_opt,
                                   use_hyper_par_objective_fun):
        model = QuadraticModel(dim=dim)
        lam0 = model.lambda_pattern.flatten(
            model.get_default_lambda(), free=lambda_free)

        get_objective_flat = model.get_flat_objective(theta_free, lambda_free)
        get_objective_for_opt = lambda x: get_objective_flat(x, lam0)
        get_objective_for_opt_grad = autograd.grad(get_objective_for_opt)
        get_objective_for_opt_hessian = autograd.hessian(get_objective_for_opt)

        get_objective_for_sens_grad = \
            autograd.grad(get_objective_flat, argnum=0)
        get_objective_for_sens_cross_hess = \
            autograd.jacobian(get_objective_for_sens_grad, argnum=1)

        opt_output = sp.optimize.minimize(
            fun=get_objective_for_opt,
            jac=get_objective_for_opt_grad,
            x0=np.zeros(model.dim),
            method='BFGS')

        get_flat_true_optimal_theta = \
            model.get_flat_true_optimal_theta(theta_free, lambda_free)
        theta0 = get_flat_true_optimal_theta(lam0)
        assert_array_almost_equal(theta0, opt_output.x)

        # Instantiate the sensitivity object.
        if use_hessian_at_opt:
            hess0 = get_objective_for_opt_hessian(theta0)
        else:
            hess0 = None

        if use_cross_hessian_at_opt:
            cross_hess0 = get_objective_for_sens_cross_hess(theta0, lam0)
        else:
            cross_hess0 = None

        if use_hyper_par_objective_fun:
            hyper_par_objective_fun = \
                paragami.FlattenFunctionInput(
                    model.get_hyper_par_objective,
                    free=[theta_free, lambda_free],
                    argnums=[0, 1],
                    patterns=[model.theta_pattern, model.lambda_pattern])
        else:
            hyper_par_objective_fun = None

        parametric_sens = \
            vittles.HyperparameterSensitivityLinearApproximation(
                objective_fun=get_objective_flat,
                opt_par_value=theta0,
                hyper_par_value=lam0,
                hessian_at_opt=hess0,
                cross_hess_at_opt=cross_hess0,
                hyper_par_objective_fun=hyper_par_objective_fun,
                validate_optimum=True)

        epsilon = 0.001
        lam1 = lam0 + epsilon
        lam_folded1 = model.lambda_pattern.fold(lam1, free=lambda_free)

        # Check the optimal parameters
        pred_diff = \
            parametric_sens.predict_opt_par_from_hyper_par(lam1) - theta0
        true_theta_folded1 = model.get_true_optimal_theta(lam_folded1)
        true_theta1 = \
            model.theta_pattern.flatten(true_theta_folded1, free=theta_free)
        true_diff = true_theta1 - theta0

        if (not theta_free) and (not lambda_free):
            # The optimum is linear in lambda, so the prediction
            # should be exact.
            assert_array_almost_equal(pred_diff, true_diff)
        else:
            # Check the relative error.
            error = np.abs(pred_diff - true_diff)
            tol = 0.01 * np.max(np.abs(true_diff))
            if not np.all(error < tol):
                print('Error in linear approximation: ',
                      error, tol, pred_diff, true_diff)
            self.assertTrue(np.all(error < tol))

        # Test the Jacobian.
        get_true_optimal_theta_lamflat = \
            paragami.FlattenFunctionInput(
                model.get_true_optimal_theta,
                patterns=model.lambda_pattern,
                free=lambda_free, argnums=0)
        def get_true_optimal_theta_flat(lam_flat):
            theta_folded = get_true_optimal_theta_lamflat(lam_flat)
            return model.theta_pattern.flatten(theta_folded, free=theta_free)

        get_dopt_dhyper = autograd.jacobian(get_true_optimal_theta_flat)
        assert_array_almost_equal(
            get_dopt_dhyper(lam0),
            parametric_sens.get_dopt_dhyper())

        ########################################
        # Test the differentiable objective.
        get_opt_par = parametric_sens.get_opt_par_function()
        assert_array_almost_equal(theta0, get_opt_par(lam0))

        # Check that you cannot evaluate at other hyperparameters
        with self.assertRaises(ValueError):
            get_opt_par(lam0 + 1)

        with self.assertRaises(ValueError):
            autograd.grad(get_opt_par)(lam0 + 1)

        delta = np.random.random(dim)
        get_opt_par_fwd = _append_jvp(get_opt_par)
        with self.assertRaises(ValueError):
            get_opt_par_fwd(lam0 + 1, delta)

        def fun_of_opt(lam):
            return np.exp(np.sum(get_opt_par(lam) + 0.1))

        # You cannot use check_grads because you cannot evaluate
        # `get_opt_par` at different values for finite differences.
        dopt_dhyper = parametric_sens.get_dopt_dhyper()

        # Test reverse mode
        assert_array_almost_equal(
            dopt_dhyper.T @ np.full(dim, fun_of_opt(lam0)),
            autograd.grad(fun_of_opt)(lam0))

        # Test forward mode
        delta = np.random.random(dim)
        fun_of_opt_fwd = _append_jvp(fun_of_opt)
        assert_array_almost_equal(
            delta.T @ dopt_dhyper.T @ np.full(dim, fun_of_opt(lam0)),
            fun_of_opt_fwd(lam0, delta))

        # Check that higher orders fail.
        with self.assertRaises(NotImplementedError):
            autograd.hessian(fun_of_opt)(lam0)

        with self.assertRaises(NotImplementedError):
            fun_of_opt_fwd2 = _append_jvp(fun_of_opt_fwd)
            fun_of_opt_fwd2(lam0, delta, delta)


    def test_quadratic_model(self):
        ft_vec = [False, True]
        dim = 3
        for (theta_free, lambda_free, use_hess, use_hyperobj, use_cross_hess) in \
            itertools.product(ft_vec, ft_vec, ft_vec, ft_vec, ft_vec):

            print(('theta_free: {}, lambda_free: {}, ' +
                   'use_hess: {}, use_hyperobj: {}').format(
                   theta_free, lambda_free, use_hess, use_hyperobj))
            self._test_linear_approximation(
                dim, theta_free, lambda_free,
                use_hess, use_cross_hess, use_hyperobj)


class TestTaylorExpansion(unittest.TestCase):
    def test_taylor_series(self):
        #################################
        # Set up the ground truth.

        # Perhaps confusingly, in the notation of the
        # ParametricSensitivityTaylorExpansion
        # and QuadraticModel class respectively,
        # eta = flattened theta
        # epsilon = flattened lambda.
        model = QuadraticModel(dim=3)

        eta_is_free = True
        eps_is_free = True
        eta0, eps0 = model.get_default_flat_values(eta_is_free, eps_is_free)
        objective = model.get_flat_objective(eta_is_free, eps_is_free)

        obj_eta_grad = autograd.grad(objective, argnum=0)
        obj_eps_grad = autograd.grad(objective, argnum=1)
        obj_eta_hessian = autograd.hessian(objective, argnum=0)
        obj_eps_hessian = autograd.hessian(objective, argnum=1)
        get_dobj_deta_deps = autograd.jacobian(
            autograd.jacobian(objective, argnum=0), argnum=1)

        hess0 = obj_eta_hessian(eta0, eps0)

        eps1 = eps0 + 1e-1
        eta1 = model.get_true_optimal_theta(eps1)

        deps = eps1 - eps0

        v1 = np.random.random(len(eta0))
        v2 = np.random.random(len(eta0))
        v3 = np.random.random(len(eta0))
        w1 = np.random.random(len(eps0))
        w2 = np.random.random(len(eps0))
        w3 = np.random.random(len(eps0))

        # Get the exact derivatives using the closed-form optimum.
        get_true_optimal_flat_theta = \
            model.get_flat_true_optimal_theta(eta_is_free, eps_is_free)

        true_deta_deps = autograd.jacobian(get_true_optimal_flat_theta)
        true_d2eta_deps2 = autograd.jacobian(true_deta_deps)
        true_d3eta_deps3 = autograd.jacobian(true_d2eta_deps2)
        true_d4eta_deps4 = autograd.jacobian(true_d3eta_deps3)

        # Sanity check using standard first-order approximation.
        d2f_deta_deps = get_dobj_deta_deps(eta0, eps0)
        assert_array_almost_equal(
            true_deta_deps(eps0),
            -1 * np.linalg.solve(hess0, d2f_deta_deps))

        ###################################
        # Test the Taylor series itself.

        test_order = 3
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            taylor_expansion = \
                sensitivity_lib.ParametricSensitivityTaylorExpansion(
                    objective_function=objective,
                    input_val0=eta0,
                    hyper_val0=eps0,
                    order=test_order,
                    hess0=hess0)

        self.assertEqual(test_order, taylor_expansion.get_max_order())
        taylor_expansion.print_terms(k=3)

        d1 = np.einsum('ij,j', true_deta_deps(eps0), deps)
        d2 = np.einsum('ijk,j,k', true_d2eta_deps2(eps0), deps, deps)
        d3 = np.einsum('ijkl,j,k,l', true_d3eta_deps3(eps0), deps, deps, deps)

        input_derivs = taylor_expansion.evaluate_input_derivs(deps)

        assert_array_almost_equal(d1, input_derivs[0])

        assert_array_almost_equal(d2, input_derivs[1])

        assert_array_almost_equal(d3, input_derivs[2])

        assert_array_almost_equal(
            eta0 + d1,
            taylor_expansion.evaluate_taylor_series(eps1, max_order=1))

        assert_array_almost_equal(
            eta0 + d1 + 0.5 * d2,
            taylor_expansion.evaluate_taylor_series(eps1, max_order=2))

        assert_array_almost_equal(
            eta0 + d1 + d2 / 2 + d3 / 6,
            taylor_expansion.evaluate_taylor_series(eps1, max_order=3))

        assert_array_almost_equal(
            eta0 + d1 + d2 / 2 + d3 / 6,
            taylor_expansion.evaluate_taylor_series(eps1))

        terms = taylor_expansion.evaluate_taylor_series_terms(
            eps1, max_order=3)
        assert_array_almost_equal(
            taylor_expansion.evaluate_taylor_series(eps1, max_order=3),
            np.sum(terms, axis=0))


if __name__ == '__main__':
    unittest.main()
