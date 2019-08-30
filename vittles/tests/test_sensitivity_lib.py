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
import time
import warnings

import vittles
from vittles import sensitivity_lib
from vittles import solver_lib
from vittles.sensitivity_lib import \
    _append_jvp, _evaluate_term_fwd, DerivativeTerm, \
    ReverseModeDerivativeArray, ForwardModeDerivativeArray, \
    ReorderedReverseModeDerivativeArray, \
    ParametricSensitivityTaylorExpansion, \
    _contract_tensor


class TestContractTensor(unittest.TestCase):
    def test_contract_tensor(self):
        x1dim = 3
        x2dim = 5

        def xs(n, dim):
            return [ np.random.random(dim) for _ in range(n) ]

        def tensor(n1, n2):
            dims = (x1dim, ) + \
                   tuple(x1dim for _ in range(n1)) + \
                   tuple(x2dim for _ in range(n2))
            return np.random.random(dims)

        deriv_array = [[ tensor(n1, n2) for n2 in range(3) ]
                         for n1 in range(3) ]

        assert_array_almost_equal(
            deriv_array[0][0],
            _contract_tensor(deriv_array[0][0], [], []))

        x1s = xs(2, x1dim)
        x2s = xs(2, x2dim)

        assert_array_almost_equal(
            np.einsum('abc,b,c->a', deriv_array[2][0], x1s[0], x1s[1]),
            _contract_tensor(deriv_array[2][0], x1s, []))

        assert_array_almost_equal(
            np.einsum('abc,b,c->a', deriv_array[0][2], x2s[0], x2s[1]),
            _contract_tensor(deriv_array[0][2], [], x2s))

        assert_array_almost_equal(
            np.einsum('abcde,b,c,d,e->a',
                      deriv_array[2][2], x1s[0], x1s[1], x2s[0], x2s[1]),
            _contract_tensor(deriv_array[2][2], x1s, x2s))


class TestForwardModederivativeArray(unittest.TestCase):
    def test_fwd_derivative_array(self):
        model = QuadraticModel(dim=3)

        eta0, eps0 = model.get_default_flat_values(True, True)
        objective = model.get_flat_objective(True, True)
        obj_eta_grad = autograd.grad(objective, argnum=0)

        max_order1 = 2
        max_order2 = 3
        deriv_array = ForwardModeDerivativeArray(
            obj_eta_grad, max_order1, max_order2)

        # n^th order derivatives require n + 1 entries, because the
        # 0-th order is included as well.
        self.assertEqual(max_order1 + 1, len(deriv_array._eval_fun_derivs))
        for i in range(len(deriv_array._eval_fun_derivs)):
            self.assertEqual(
                max_order2 + 1,
                len(deriv_array._eval_fun_derivs[i]))

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
        v = [ np.random.random(len(eta0)) for _ in range(max_order1) ]

        # Directions in eps
        w = [ np.random.random(len(eps0)) for _ in range(max_order2) ]

        # Test the array entries
        assert_array_almost_equal(
            obj_eta_grad(eta0, eps0),
            deriv_array.eval_directional_derivative(eta0, eps0, [], []))

        assert_array_almost_equal(
            dg_deta(eta0, eps0, v[0]),
            deriv_array.eval_directional_derivative(
                eta0, eps0, [v[0]], []))

        assert_array_almost_equal(
            dg_deps(eta0, eps0, w[0]),
            deriv_array.eval_directional_derivative(
                eta0, eps0, [], [w[0]]))

        assert_array_almost_equal(
            d2g_deta_deta(eta0, eps0, v[0], v[1]),
            deriv_array.eval_directional_derivative(
                eta0, eps0, [v[0], v[1]], []))

        assert_array_almost_equal(
            d2g_deta_deps(eta0, eps0, v[0], w[0]),
            deriv_array.eval_directional_derivative(
                eta0, eps0, [v[0]], [w[0]]))

        assert_array_almost_equal(
            d2g_deps_deps(eta0, eps0, w[0], w[1]),
            deriv_array.eval_directional_derivative(
                eta0, eps0, [], [w[0], w[1]]))


class TestReverseModeDerivativeArray(unittest.TestCase):
    def get_test_fun(self, dim1, dim2, eqdim):
        a1 = np.random.random(dim1)
        a2 = np.random.random(dim2)

        # Typically, we will be using the derivative array with a gradient.
        # But, to make sure we are using the dimensions correctly, we will
        # test it with a vector function whose dimension does not match either
        # regressor.
        def g(x1, x2):
            val = np.sin(np.dot(a1, x1) + np.dot(a2, x2))
            return (0.5 + np.arange(eqdim)) * val

        x1 = np.random.random(dim1)
        x2 = np.random.random(dim2)

        return g, x1, x2

    def _test_warning(self, RMDA, swapped=False):
        def g(x1, x2):
            return 0.

        deriv_array = RMDA(fun=g, order1=2, order2=2)

        # Swap so that it doesn't take a long time when running with force=True.
        if swapped:
            x1 = np.zeros(1000)
            x2 = np.zeros(1)
        else:
            x1 = np.zeros(1)
            x2 = np.zeros(1000)

        with self.assertRaises(ValueError):
            # 100^2 * 100^2 > 1e6, and should throw an error when you try to
            # set the initial point.
            deriv_array.set_evaluation_location(x1, x2)

        # Check that it works with force.
        deriv_array.set_evaluation_location(x1, x2, force=True, verbose=True)

        deriv_array = RMDA(fun=g, order1=3, order2=3)
        x1 = np.zeros(2)
        x2 = np.zeros(2)
        with self.assertRaises(ValueError):
            # Both orders are greater than two.
            deriv_array.set_evaluation_location(x1, x2)

        # Check that it works with force.
        deriv_array.set_evaluation_location(x1, x2, force=True)

    def _test_derivative_arrays(self, RMDA):
        x1dim = 2
        x2dim = 4
        gdim = 3
        g, x1, x2 = self.get_test_fun(x1dim, x2dim, gdim)

        max_order1 = 2
        max_order2 = 2
        deriv_array = RMDA(fun=g, order1=max_order1, order2=max_order2)
        deriv_array.set_evaluation_location(x1, x2)

        # This does not make sense with a swapped array.
        # TODO: replace this with another test.
        # self.assertEqual(
        #     max_order1 + 1,
        #     len(deriv_array._eval_deriv_arrays))
        # for i in range(max_order1 + 1):
        #     self.assertEqual(
        #         max_order2 + 1,
        #         len(deriv_array._eval_deriv_arrays[i]))

        # Check the first couple deriv_arrays by hand.
        assert_array_almost_equal(
            g(x1, x2),
            deriv_array.deriv_arrays(0, 0))

        assert_array_almost_equal(
            autograd.jacobian(g, argnum=0)(x1, x2),
            deriv_array.deriv_arrays(1, 0))

        assert_array_almost_equal(
            autograd.jacobian(g, argnum=1)(x1, x2),
            deriv_array.deriv_arrays(0, 1))

        for k1, k2 in itertools.product(
                range(max_order1 + 1), range(max_order2 + 1)):
            expected_dim = (gdim, ) + \
                           tuple(x1dim for _ in range(k1)) + \
                           tuple(x2dim for _ in range(k2))
            self.assertEqual(
                expected_dim, deriv_array.deriv_arrays(k1, k2).shape)

    def _test_evaluate_directional_derivative(self, RMDA):
        g, x1, x2 = self.get_test_fun(2, 4, 3)

        max_order1 = 2
        max_order2 = 2
        deriv_array = RMDA(fun=g, order1=max_order1, order2=max_order2)
        deriv_array.set_evaluation_location(x1, x2)

        dx1s = [ np.random.random(len(x1)) \
                 for _ in range(max_order1) ]
        dx2s = [ np.random.random(len(x2)) \
                 for _ in range(max_order2 - 1) ]

        deriv = deriv_array.eval_directional_derivative(x1, x2, dx1s, dx2s)

        # Check accuracy against _append_jvp.
        eval_forward_deriv = g
        for _ in range(len(dx1s)):
            eval_forward_deriv = \
                _append_jvp(eval_forward_deriv, num_base_args=2, argnum=0)
        for _ in range(len(dx2s)):
            eval_forward_deriv = \
                _append_jvp(eval_forward_deriv, num_base_args=2, argnum=1)
        assert_array_almost_equal(
            eval_forward_deriv(x1, x2, *(dx1s + dx2s)),
            deriv)

        # Test the errors.
        with self.assertRaises(ValueError):
            deriv_array.eval_directional_derivative(x1 + 0.1, x2, dx1s, dx2s)

        with self.assertRaises(ValueError):
            deriv_array.eval_directional_derivative(x1, x2 + 0.1, dx1s, dx2s)


    def test_classes(self):
        # Currently this test only makes sense with the original class.
        self._test_derivative_arrays(ReverseModeDerivativeArray)
        self._test_warning(ReverseModeDerivativeArray)
        self._test_evaluate_directional_derivative(ReverseModeDerivativeArray)

        for swapped in [False, True]:
            def RMDA(fun, order1, order2):
                return ReorderedReverseModeDerivativeArray(
                    fun, order1, order2, swapped)
            self._test_derivative_arrays(RMDA)
            self._test_warning(RMDA, swapped=swapped)
            self._test_evaluate_directional_derivative(RMDA)


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
        deriv_array = \
            ForwardModeDerivativeArray(obj_eta_grad, order1=2, order2=2)
        eval_directional_derivative = \
            deriv_array.eval_directional_derivative

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
                dterm, eta0, eps0, deps, eta_derivs,
                eval_directional_derivative))

        dterms1 = sensitivity_lib._get_taylor_base_terms()
        deriv_terms = [ true_deta_deps(eps0) @ deps ]
        assert_array_almost_equal(
            dg_deps(eta0, eps0, deps),
            _evaluate_term_fwd(
                dterms1[0], eta0, eps0, deps, deriv_terms,
                eval_directional_derivative))


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
        self._test_taylor_series(use_hess=True, custom_solver=False)
        self._test_taylor_series(use_hess=False, custom_solver=False)
        self._test_taylor_series(use_hess=False, custom_solver=True)

    def _test_taylor_series(self, use_hess, custom_solver):
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

        obj_eta_hessian = autograd.hessian(objective, argnum=0)
        hess0 = obj_eta_hessian(eta0, eps0)

        test_order = 3
        if custom_solver:
            estimating_equation = autograd.grad(objective, argnum=0)
            solver = solver_lib.get_cg_solver(
                lambda v: hess0 @ v, dim=3)
            taylor_expansion = \
                ParametricSensitivityTaylorExpansion(
                    estimating_equation=estimating_equation,
                    input_val0=eta0,
                    hyper_val0=eps0,
                    order=test_order,
                    hess_solver=solver)
        else:
            if use_hess:
                hess0_arg = hess0
            else:
                hess0_arg = None

            taylor_expansion = \
                ParametricSensitivityTaylorExpansion.optimization_objective(
                        objective_function=objective,
                        input_val0=eta0,
                        hyper_val0=eps0,
                        order=test_order,
                        hess0=hess0_arg)

        self.assertEqual(test_order, taylor_expansion.get_max_order())
        taylor_expansion.print_terms(k=3)

        # Get the exact derivatives using the closed-form optimum.

        eps1 = eps0 + 1e-1
        eta1 = model.get_true_optimal_theta(eps1)

        deps = eps1 - eps0

        get_true_optimal_flat_theta = \
            model.get_flat_true_optimal_theta(eta_is_free, eps_is_free)

        true_deta_deps = autograd.jacobian(get_true_optimal_flat_theta)
        true_d2eta_deps2 = autograd.jacobian(true_deta_deps)
        true_d3eta_deps3 = autograd.jacobian(true_d2eta_deps2)
        true_d4eta_deps4 = autograd.jacobian(true_d3eta_deps3)

        # Sanity check using standard first-order approximation.
        get_dobj_deta_deps = autograd.jacobian(
            autograd.jacobian(objective, argnum=0), argnum=1)
        d2f_deta_deps = get_dobj_deta_deps(eta0, eps0)
        assert_array_almost_equal(
            true_deta_deps(eps0),
            -1 * np.linalg.solve(hess0, d2f_deta_deps))

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

    def _test_max_order(self, eta_order, eps_order, test_order):
        # TODO: test this with reverse mode and swapping.

        # Partial derivative of the gradient higher than eta_order and
        # eps_order are zero.
        def objective(eta, eps):
            eta_sum = np.sum(eta)
            eps_sum = np.sum(eps)
            return (eta_sum ** (eta_order + 1)) * (eps_sum ** eps_order)

        # These need to be nonzero for the test to be valid.
        # Note that this test doesn't require an actual optimum,
        # nor does it require the real Hessian.
        eta0 = 0.01 * np.arange(2)
        eps0 = 0.02 * np.arange(3)
        eps1 = eps0 + 1

        # We don't actually need the real Hessian for this test.
        hess0 = np.diag(np.array([2.1, 4.5]))

        taylor_expansion_truth = \
            ParametricSensitivityTaylorExpansion.optimization_objective(
                objective_function=objective,
                input_val0=eta0,
                hyper_val0=eps0,
                hess0=hess0,
                order=test_order)

        taylor_expansion_test = \
            ParametricSensitivityTaylorExpansion.optimization_objective(
                objective_function=objective,
                input_val0=eta0,
                hyper_val0=eps0,
                hess0=hess0,
                max_input_order=eta_order,
                max_hyper_order=eps_order,
                order=test_order)

        assert_array_almost_equal(
            taylor_expansion_truth.evaluate_taylor_series(eps1),
            taylor_expansion_test.evaluate_taylor_series(eps1))

    def test_max_orders(self):
        self._test_max_order(1, 1, 4)
        self._test_max_order(1, 2, 4)
        self._test_max_order(2, 1, 4)
        self._test_max_order(2, 2, 4)
        self._test_max_order(1, 3, 4)
        self._test_max_order(3, 1, 4)

    def test_reverse_mode_swapping(self):
        dim1 = 3
        dim2 = 6

        a = (np.random.random(dim1) - 0.5)  / dim1
        b = (np.random.random(dim2) - 0.5) / dim2

        def objective12(x1, x2):
            return np.array([np.exp(np.dot(a, x1) + np.dot(b, x2)), 0])

        def objective21(x2, x1):
            return objective12(x1, x2)

        x1 = np.random.random(dim1)
        dx1 = np.random.random(dim1)
        solver1 = solver_lib.get_cholesky_solver(np.eye(dim1))

        x2 = np.random.random(dim2)
        dx2 = np.random.random(dim2)
        solver2 = solver_lib.get_cholesky_solver(np.eye(dim2))

        order = 2
        taylor_12 = \
            ParametricSensitivityTaylorExpansion(
                estimating_equation=objective12,
                forward_mode=False,
                input_val0=x1,
                hyper_val0=x2,
                hess_solver=solver1,
                order=order)

        taylor_21 = \
            ParametricSensitivityTaylorExpansion(
                estimating_equation=objective21,
                forward_mode=False,
                input_val0=x2,
                hyper_val0=x1,
                hess_solver=solver2,
                order=order)

        for k1, k2 in itertools.product(range(order + 1), range(order + 1)):
            print(k1, k2)
            dx1s = [dx1 for _ in range(k1)]
            dx2s = [dx2 for _ in range(k2)]

            # Just check that these evaluate.  They are not comparable
            # with one another.
            d12 = taylor_12._deriv_array.eval_directional_derivative(
                x1, x2, dx1s, dx2s)
            d21 = taylor_21._deriv_array.eval_directional_derivative(
                x2, x1, dx2s, dx1s)

            # These derivatives should be comparable.
            deriv12 = taylor_12._deriv_array.deriv_arrays(k1, k2)
            deriv21 = taylor_21._deriv_array.deriv_arrays(k2, k1)

            assert_array_almost_equal(
                _contract_tensor(deriv12, dx1s, dx2s),
                _contract_tensor(deriv21, dx2s, dx1s))

    def test_weighted_linear_regression(self):
        # Test with weighted linear regression, which has only one partial
        # derivative with respect to the hyperparameter.
        n_obs = 10
        dim = 2
        theta_true = np.array([0.5, -0.1])
        x = np.random.random((n_obs, dim))
        y = x @ theta_true + np.random.normal(n_obs)
        def objective(theta, w):
            resid = y - x @ theta
            return np.sum(w * (resid ** 2))

        def run_regression(w):
            xtx = np.einsum('n,ni,nj->ij', w, x, x)
            xty = np.einsum('n,ni,n->i', w, x, y)
            return np.linalg.solve(xtx, xty)

        w1 = np.ones(n_obs)
        theta0 = run_regression(w1)
        dw = np.random.random(n_obs) - 0.5

        objective_grad = autograd.grad(objective, argnum=0)
        self.assertTrue(
            np.linalg.norm(objective_grad(theta0, w1)) < 1e-8)
        self.assertTrue(
            np.linalg.norm(objective_grad(
                run_regression(w1 + dw), w1 + dw)) < 1e-8)

        taylor_expansion = \
            ParametricSensitivityTaylorExpansion.optimization_objective(
                objective_function=objective,
                input_val0=theta0,
                hyper_val0=w1,
                order=4,
                max_hyper_order=1,
                max_input_order=2,
                forward_mode=False)

        # Get exact derivatives using the closed form.
        dtheta_dw = _append_jvp(run_regression)
        d2theta_dw2 = _append_jvp(dtheta_dw)
        d3theta_dw3 = _append_jvp(d2theta_dw2)
        d4theta_dw4 = _append_jvp(d3theta_dw3)

        d1 = dtheta_dw(w1, dw)
        d2 = d2theta_dw2(w1, dw, dw)
        d3 = d3theta_dw3(w1, dw, dw, dw)
        d4 = d4theta_dw4(w1, dw, dw, dw, dw)

        assert_array_almost_equal(
            taylor_expansion.evaluate_taylor_series(w1 + dw, max_order=1),
            theta0 + d1)

        assert_array_almost_equal(
            taylor_expansion.evaluate_taylor_series(w1 + dw, max_order=2),
            theta0 + d1 + d2 / 2.0)

        assert_array_almost_equal(
            taylor_expansion.evaluate_taylor_series(w1 + dw, max_order=3),
            theta0 + d1 + d2 / 2.0 + d3 / 6.0)

        assert_array_almost_equal(
            taylor_expansion.evaluate_taylor_series(w1 + dw, max_order=4),
            theta0 + d1 + d2 / 2.0 + d3 / 6.0 + d4 / 24.0)


if __name__ == '__main__':
    unittest.main()
