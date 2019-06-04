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


class TestLinearResponseCovariances(unittest.TestCase):
    def test_lr(self):
        np.random.seed(42)

        dim = 4
        mfvb_par_pattern = paragami.PatternDict()
        mfvb_par_pattern['mean'] = paragami.NumericArrayPattern((dim, ))
        mfvb_par_pattern['var'] = paragami.NumericArrayPattern((dim, ))

        mfvb_par = mfvb_par_pattern.empty(valid=True)

        true_mean = np.arange(0, dim)
        true_cov = dim * np.eye(dim) + np.outer(true_mean, true_mean)
        true_info = np.linalg.inv(true_cov)

        def get_kl(mfvb_par):
            """
            This is :math:`KL(q(\\theta) || p(\\theta))`` where
            :math:`p(\\theta)` is normal with mean ``true_mean``
            and inverse covariance ``ture_info`` and the variational
            distribution :math:`q` is given by ``mfvb_par``.
            The result is only up to constants that do not depend on
            :math:`q`.
            """

            t_centered = mfvb_par['mean'] - true_mean
            e_log_p = -0.5 * (
                np.trace(true_info @ np.diag(mfvb_par['var'])) +
                t_centered.T @ true_info @ t_centered)
            q_ent = 0.5 * np.sum(np.log(mfvb_par['var']))

            return -1 * (q_ent + e_log_p)

        par_free = True
        init_hessian = True

        for par_free, init_hessian in \
            itertools.product([False, True], [False, True]):

            get_kl_flat = paragami.FlattenFunctionInput(
                original_fun=get_kl, patterns=mfvb_par_pattern, free=par_free)
            get_kl_flat_grad = autograd.grad(get_kl_flat, argnum=0)
            get_kl_flat_hessian = autograd.hessian(get_kl_flat, argnum=0)

            # This is the optimum.
            mfvb_par['mean'] = true_mean
            mfvb_par['var'] = 1 / np.diag(true_info)
            mfvb_par_flat = mfvb_par_pattern.flatten(mfvb_par, free=par_free)

            hess0 = get_kl_flat_hessian(mfvb_par_flat)

            # Sanity check.the optimum.
            assert_array_almost_equal(
                0., np.linalg.norm(get_kl_flat_grad(mfvb_par_flat)))

            if init_hessian:
                lr_covs = vittles.LinearResponseCovariances(
                    objective_fun=get_kl_flat,
                    opt_par_value=mfvb_par_flat,
                    validate_optimum=True,
                    hessian_at_opt=hess0,
                    grad_tol=1e-15)
            else:
                lr_covs = vittles.LinearResponseCovariances(
                    objective_fun=get_kl_flat,
                    opt_par_value=mfvb_par_flat,
                    validate_optimum=True,
                    grad_tol=1e-15)

            assert_array_almost_equal(hess0, lr_covs.get_hessian_at_opt())

            get_mean_flat = paragami.FlattenFunctionInput(
                lambda mfvb_par: mfvb_par['mean'],
                patterns=mfvb_par_pattern,
                free=par_free)
            theta_lr_cov = lr_covs.get_lr_covariance(get_mean_flat)

            # The LR covariance is exact for the multivariate normal.
            assert_array_almost_equal(true_cov, theta_lr_cov)
            moment_jac = lr_covs.get_moment_jacobian(get_mean_flat)
            assert_array_almost_equal(
                theta_lr_cov,
                lr_covs.get_lr_covariance_from_jacobians(
                    moment_jac, moment_jac))

            # Check cross-covariances.
            get_mean01_flat = paragami.FlattenFunctionInput(
                lambda mfvb_par: mfvb_par['mean'][0:2],
                patterns=mfvb_par_pattern,
                free=par_free)
            get_mean23_flat = paragami.FlattenFunctionInput(
                lambda mfvb_par: mfvb_par['mean'][2:4],
                patterns=mfvb_par_pattern,
                free=par_free)
            moment01_jac = lr_covs.get_moment_jacobian(get_mean01_flat)
            moment23_jac = lr_covs.get_moment_jacobian(get_mean23_flat)
            assert_array_almost_equal(
                theta_lr_cov[0:2, 2:4],
                lr_covs.get_lr_covariance_from_jacobians(
                    moment01_jac, moment23_jac))

            # Check that you get an error when passing in a Jacobian with the
            # wrong dimension.
            self.assertRaises(
                ValueError,
                lambda: lr_covs.get_lr_covariance_from_jacobians(
                                    moment_jac.T, moment_jac))

            self.assertRaises(
                ValueError,
                lambda: lr_covs.get_lr_covariance_from_jacobians(
                                    moment_jac, moment_jac.T))

            self.assertRaises(
                ValueError,
                lambda: lr_covs.get_lr_covariance_from_jacobians(
                                    moment_jac[:, :, None], moment_jac))
            self.assertRaises(
                ValueError,
                lambda: lr_covs.get_lr_covariance_from_jacobians(
                                    moment_jac, moment_jac[:, :, None]))


class TestHyperparameterSensitivityLinearApproximation(unittest.TestCase):
    def _test_linear_approximation(self, dim,
                                   theta_free, lambda_free,
                                   use_hessian_at_opt,
                                   use_cross_hessian_at_opt,
                                   use_hyper_par_objective_fun):
        model = QuadraticModel(dim=dim)
        lam_folded0 = deepcopy(model.lam)
        lam0 = model.lambda_pattern.flatten(lam_folded0, free=lambda_free)

        # Sanity check that the optimum is correct.
        get_objective_flat = paragami.FlattenFunctionInput(
            model.get_objective,
            free=[theta_free, lambda_free],
            argnums=[0, 1],
            patterns=[model.theta_pattern, model.lambda_pattern])
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

        theta_folded_0 = model.get_true_optimal_theta(model.lam)
        theta0 = model.theta_pattern.flatten(theta_folded_0, free=theta_free)
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
    def test_everything(self):
        # TODO: split some of these out into standalone tests.

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
        eta0 = model.theta_pattern.flatten(
            model.get_true_optimal_theta(model.lam), free=eta_is_free)
        eps0 = model.lambda_pattern.flatten(model.lam, free=eps_is_free)

        objective = paragami.FlattenFunctionInput(
            original_fun=model.get_objective,
            patterns=[model.theta_pattern, model.lambda_pattern],
            free=[eta_is_free, eps_is_free],
            argnums=[0, 1])

        obj_eta_grad = autograd.grad(objective, argnum=0)
        obj_eps_grad = autograd.grad(objective, argnum=1)
        obj_eta_hessian = autograd.hessian(objective, argnum=0)
        obj_eps_hessian = autograd.hessian(objective, argnum=1)
        get_dobj_deta_deps = autograd.jacobian(
            autograd.jacobian(objective, argnum=0), argnum=1)

        hess0 = obj_eta_hessian(eta0, eps0)

        eps1 = eps0 + 1e-1
        eta1 = model.get_true_optimal_theta(eps1)

        # Get the exact derivatives using the closed-form optimum.
        def get_true_optimal_flat_theta(lam):
            theta = model.get_true_optimal_theta(lam)
            return model.theta_pattern.flatten(theta, free=eta_is_free)

        get_true_optimal_flat_theta = paragami.FlattenFunctionInput(
            original_fun=get_true_optimal_flat_theta,
            patterns=model.lambda_pattern,
            free=eps_is_free,
            argnums=0)
        true_deta_deps = autograd.jacobian(get_true_optimal_flat_theta)
        true_d2eta_deps2 = autograd.jacobian(true_deta_deps)
        true_d3eta_deps3 = autograd.jacobian(true_d2eta_deps2)
        true_d4eta_deps4 = autograd.jacobian(true_d3eta_deps3)

        # Sanity check using standard first-order approximation.
        d2f_deta_deps = get_dobj_deta_deps(eta0, eps0)
        assert_array_almost_equal(
            true_deta_deps(eps0),
            -1 * np.linalg.solve(hess0, d2f_deta_deps))

        ########################
        # Test append_jvp.
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

        ########################
        # Test derivative terms.

        # Again, first some ground truth.
        def eval_deta_deps(eta, eps, v1):
            assert np.max(np.sum(eps - eps0)) < 1e-8
            assert np.max(np.sum(eta - eta0)) < 1e-8
            return -1 * np.linalg.solve(hess0, d2f_deta_deps @ v1)

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

        # This is a manual version of the second derivative.
        def eval_d2eta_deps2(eta, eps, delta_eps):
            assert np.max(np.sum(eps - eps0)) < 1e-8
            assert np.max(np.sum(eta - eta0)) < 1e-8

            deta_deps = -1 * np.linalg.solve(
                hess0, dg_deps(eta, eps, delta_eps))

            # Then the terms in the second derivative.
            d2_terms = \
                d2g_deps_deps(eta, eps, delta_eps, delta_eps) + \
                d2g_deps_deta(eta, eps, delta_eps, deta_deps) + \
                d2g_deta_deps(eta, eps, deta_deps, delta_eps) + \
                d2g_deta_deta(eta, eps, deta_deps, deta_deps)
            d2eta_deps2 = -1 * np.linalg.solve(hess0, d2_terms)
            return d2eta_deps2

        eval_g_derivs = \
            sensitivity_lib._generate_two_term_fwd_derivative_array(
                obj_eta_grad, order=5)

        assert_array_almost_equal(
            hess0 @ v1,
            eval_g_derivs[1][0](eta0, eps0, v1))

        d2g_deta_deta(eta0, eps0, v1, v2)
        eval_g_derivs[2][0](eta0, eps0, v1, v2)

        assert_array_almost_equal(
            d2g_deta_deta(eta0, eps0, v1, v2),
            eval_g_derivs[2][0](eta0, eps0, v1, v2))

        assert_array_almost_equal(
            d2g_deta_deps(eta0, eps0, v1, v2),
            eval_g_derivs[1][1](eta0, eps0, v1, v2))

        # Test the DerivativeTerm.

        dterm = sensitivity_lib.DerivativeTerm(
            eps_order=1,
            eta_orders=[1, 0],
            prefactor=1.5,
            eval_g_derivs=eval_g_derivs)

        deps = eps1 - eps0

        eta_derivs = [ eval_deta_deps(eta0, eps0, deps) ]
        assert_array_almost_equal(
            dterm.prefactor * d2g_deta_deps(
                eta0, eps0, eval_deta_deps(eta0, eps0, deps), deps),
            dterm.evaluate(eta0, eps0, deps,eta_derivs))

        dterms = [
            sensitivity_lib.DerivativeTerm(
                eps_order=2,
                eta_orders=[0, 0],
                prefactor=1.5,
                eval_g_derivs=eval_g_derivs),
            sensitivity_lib.DerivativeTerm(
                eps_order=1,
                eta_orders=[1, 0],
                prefactor=2,
                eval_g_derivs=eval_g_derivs),
            sensitivity_lib.DerivativeTerm(
                eps_order=1,
                eta_orders=[1, 0],
                prefactor=3,
                eval_g_derivs=eval_g_derivs) ]

        dterms_combined = sensitivity_lib._consolidate_terms(dterms)
        self.assertEqual(3, len(dterms))
        self.assertEqual(2, len(dterms_combined))

        # TODO: test dterm.differentiate() explicity.

        dterms1 = sensitivity_lib._get_taylor_base_terms(eval_g_derivs)

        deriv_terms = [ true_deta_deps(eps0) @ deps ]
        assert_array_almost_equal(
            dg_deps(eta0, eps0, deps),
            dterms1[0].evaluate(eta0, eps0, deps, deriv_terms))

        hess_solver = solver_lib.SystemSolver(hess0, 'factorization')

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


class TestBlockHessian(unittest.TestCase):
    def test_block_hessian(self):
        np.random.seed(42)

        group_size = 3
        num_groups = 10
        d = group_size * num_groups

        pattern = paragami.PatternDict()
        pattern['array'] = \
            paragami.NumericArrayPattern((num_groups, group_size))
        mat_pattern = paragami.PSDSymmetricMatrixPattern(size=group_size)
        pattern['mats'] = paragami.PatternArray((num_groups,), mat_pattern)

        def f(x_dict):
            return 0.5 * np.einsum(
                'nij,ni,nj', x_dict['mats'], x_dict['array'], x_dict['array'])

        f_flat = paragami.FlattenFunctionInput(
            f, argnums=0, free=True, patterns=pattern)

        x = pattern.random()
        x_flat = pattern.flatten(x, free=True)
        f(x)

        f_hess = autograd.hessian(f_flat, argnum=0)
        h0 = f_hess(x_flat)

        inds = []
        for g in range(num_groups):
            x_bool = pattern.empty_bool(False)
            x_bool['array'][g, :] = True
            x_bool['mats'][g, :, :] = True
            inds.append(pattern.flat_indices(x_bool, free=True))
        inds = np.array(inds)

        sparse_hess = vittles.SparseBlockHessian(f_flat, inds)
        block_hess = sparse_hess.get_block_hessian(x_flat)

        assert_array_almost_equal(np.array(block_hess.todense()), h0)

    def test_full_hessian(self):
        np.random.seed(42)

        group_size = 3
        num_groups = 10
        d = group_size * num_groups

        pattern = paragami.PatternDict()
        pattern['array'] = \
            paragami.NumericArrayPattern((num_groups, group_size))
        mat_pattern = paragami.PSDSymmetricMatrixPattern(size=group_size)
        pattern['mats'] = paragami.PatternArray((num_groups,), mat_pattern)
        pattern['scales'] = paragami.NumericVectorPattern(length=2, lb=0.0)

        def f(x_dict):
            scale = np.prod(x_dict['scales'])
            scale_prior = np.exp(-1 * scale)
            return 0.5 * scale * np.einsum(
                'nij,ni,nj', x_dict['mats'], x_dict['array'], x_dict['array'])

        f_flat = paragami.FlattenFunctionInput(
            f, argnums=0, free=True, patterns=pattern)

        x = pattern.random()
        x_flat = pattern.flatten(x, free=True)
        f(x)

        group_inds = []
        x_bool = pattern.empty_bool(False)
        for g in range(num_groups):
            x_bool['array'][g, :] = True
            x_bool['mats'][g, :, :] = True
            group_inds.append(pattern.flat_indices(x_bool, free=True))
            x_bool['array'][g, :] = False
            x_bool['mats'][g, :, :] = False
        group_inds = np.array(group_inds)

        f_hess = autograd.hessian(f_flat, argnum=0)
        h0 = f_hess(x_flat)

        x_bool['scales'][:] = True
        global_inds_paragami = pattern.flat_indices(x_bool, free=True)
        x_bool['scales'][:] = False

        sparse_hess = vittles.SparseBlockHessian(f_flat, group_inds)

        block_hess = \
            sparse_hess.get_block_hessian(x_flat) + \
            sparse_hess.get_global_hessian(x_flat)
        assert_array_almost_equal(h0, block_hess.todense())

        block_hess = \
            sparse_hess.get_block_hessian(x_flat) + \
            sparse_hess.get_global_hessian(
                x_flat, global_inds=global_inds_paragami)
        assert_array_almost_equal(h0, block_hess.todense())

        block_hess = sparse_hess.get_hessian(x_flat, print_every=1)
        assert_array_almost_equal(h0, block_hess.todense())

if __name__ == '__main__':
    unittest.main()
