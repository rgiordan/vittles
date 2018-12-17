#!/usr/bin/env python3

import autograd
import autograd.numpy as np
import itertools
from numpy.testing import assert_array_almost_equal
import vittles
from vittles import sensitivity_lib
import scipy as sp
from test_utils import QuadraticModel
import unittest
import warnings


class TestLinearResponseCovariances(unittest.TestCase):
    def test_lr(self):
        np.random.seed(42)

        dim = 4
        mfvb_par_pattern = vittles.PatternDict()
        mfvb_par_pattern['mean'] = vittles.NumericArrayPattern((dim, ))
        mfvb_par_pattern['var'] = vittles.NumericArrayPattern((dim, ))

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

            get_kl_flat = vittles.FlattenedFunction(
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
            # Just check that you can get the cholesky decomposition.
            lr_covs.get_hessian_cholesky_at_opt()

            get_mean_flat = vittles.FlattenedFunction(
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
            get_mean01_flat = vittles.FlattenedFunction(
                lambda mfvb_par: mfvb_par['mean'][0:2],
                patterns=mfvb_par_pattern,
                free=par_free)
            get_mean23_flat = vittles.FlattenedFunction(
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


class HyperparameterSensitivityLinearApproximation(unittest.TestCase):
    def _test_linear_approximation(self, dim,
                                   theta_free, lambda_free,
                                   use_hessian_at_opt,
                                   use_hyper_par_objective_fun):
        model = QuadraticModel(dim=dim)

        # Sanity check that the optimum is correct.
        get_objective_flat = vittles.FlattenedFunction(
            model.get_objective, free=theta_free, argnums=0,
            patterns=model.theta_pattern)
        get_objective_for_opt = vittles.Functor(
            get_objective_flat, argnums=0)
        get_objective_for_opt.cache_args(None, model.lam)
        get_objective_for_opt_grad = autograd.grad(get_objective_for_opt)
        get_objective_for_opt_hessian = autograd.hessian(get_objective_for_opt)

        opt_output = sp.optimize.minimize(
            fun=get_objective_for_opt,
            jac=get_objective_for_opt_grad,
            x0=np.zeros(model.dim),
            method='BFGS')

        theta0 = model.get_true_optimal_theta(model.lam)
        theta_flat = model.theta_pattern.flatten(theta0, free=theta_free)
        assert_array_almost_equal(theta_flat, opt_output.x)

        # Instantiate the sensitivity object.
        if use_hessian_at_opt:
            hess0 = get_objective_for_opt_hessian(theta_flat)
        else:
            hess0 = None

        if use_hyper_par_objective_fun:
            hyper_par_objective_fun = model.get_hyper_par_objective
        else:
            hyper_par_objective_fun = None

        parametric_sens = \
            vittles.HyperparameterSensitivityLinearApproximation(
                objective_fun=model.get_objective,
                opt_par_pattern=model.theta_pattern,
                hyper_par_pattern=model.lambda_pattern,
                opt_par_folded_value=theta0,
                hyper_par_folded_value=model.lam,
                opt_par_is_free=theta_free,
                hyper_par_is_free=lambda_free,
                hessian_at_opt=hess0,
                hyper_par_objective_fun=hyper_par_objective_fun)

        epsilon = 0.01
        lambda1 = model.lam + epsilon

        # Check the optimal parameters
        pred_diff = \
            parametric_sens.predict_opt_par_from_hyper_par(lambda1) - theta0
        true_diff = model.get_true_optimal_theta(lambda1) - theta0

        if (not theta_free) and (not lambda_free):
            # The model is linear in lambda, so the prediction should be exact.
            assert_array_almost_equal(pred_diff, true_diff)
        else:
            # Check the relative error.
            error = np.abs(pred_diff - true_diff)
            tol = epsilon * np.mean(np.abs(true_diff))
            if not np.all(error < tol):
                print('Error in linear approximation: ', error, tol)
            self.assertTrue(np.all(error < tol))

        # Test the Jacobian.
        get_true_optimal_theta_lamflat = vittles.FlattenedFunction(
            model.get_true_optimal_theta, patterns=model.lambda_pattern,
            free=lambda_free, argnums=0)
        def get_true_optimal_theta_flat(lam_flat):
            theta0 = get_true_optimal_theta_lamflat(lam_flat)
            return model.theta_pattern.flatten(theta0, free=theta_free)

        get_dopt_dhyper = autograd.jacobian(get_true_optimal_theta_flat)
        lambda_flat = model.lambda_pattern.flatten(model.lam, free=lambda_free)
        assert_array_almost_equal(
            get_dopt_dhyper(lambda_flat),
            parametric_sens.get_dopt_dhyper())

    def test_quadratic_model(self):
        ft_vec = [False, True]
        dim = 3
        for (theta_free, lambda_free, use_hess, use_hyperobj) in \
            itertools.product(ft_vec, ft_vec, ft_vec, ft_vec):

            print(('theta_free: {}, lambda_free: {}, ' +
                   'use_hess: {}, use_hyperobj: {}').format(
                   theta_free, lambda_free, use_hess, use_hyperobj))
            self._test_linear_approximation(
                dim, theta_free, lambda_free,
                use_hess, use_hyperobj)


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

        objective = vittles.FlattenedFunction(
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

        get_true_optimal_flat_theta = vittles.FlattenedFunction(
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
            eval_eta_derivs=[ eval_deta_deps ],
            eval_g_derivs=eval_g_derivs)

        deps = eps1 - eps0

        assert_array_almost_equal(
            dterm.prefactor * d2g_deta_deps(
                eta0, eps0, eval_deta_deps(eta0, eps0, deps), deps),
            dterm.evaluate(eta0, eps0, deps))

        dterms = [
            sensitivity_lib.DerivativeTerm(
                eps_order=2,
                eta_orders=[0, 0],
                prefactor=1.5,
                eval_eta_derivs=[ eval_deta_deps ],
                eval_g_derivs=eval_g_derivs),
            sensitivity_lib.DerivativeTerm(
                eps_order=1,
                eta_orders=[1, 0],
                prefactor=2,
                eval_eta_derivs=[ eval_deta_deps ],
                eval_g_derivs=eval_g_derivs),
            sensitivity_lib.DerivativeTerm(
                eps_order=1,
                eta_orders=[1, 0],
                prefactor=3,
                eval_eta_derivs=[ eval_deta_deps ],
                eval_g_derivs=eval_g_derivs) ]


        dterms_combined = sensitivity_lib._consolidate_terms(dterms)
        self.assertEqual(3, len(dterms))
        self.assertEqual(2, len(dterms_combined))

        assert_array_almost_equal(
            sensitivity_lib.evaluate_terms(dterms, eta0, eps0, deps),
            sensitivity_lib.evaluate_terms(dterms_combined, eta0, eps0, deps))

        dterms1 = sensitivity_lib._get_taylor_base_terms(eval_g_derivs)

        assert_array_almost_equal(
            dg_deps(eta0, eps0, deps),
            dterms1[0].evaluate(eta0, eps0, deps))

        assert_array_almost_equal(
            np.einsum('ij,j', true_deta_deps(eps0), deps),
            sensitivity_lib.evaluate_dketa_depsk(
                hess0, dterms1, eta0, eps0, deps))

        assert_array_almost_equal(
            eval_deta_deps(eta0, eps0, deps),
            sensitivity_lib.evaluate_dketa_depsk(
                hess0, dterms1, eta0, eps0, deps))

        dterms2 = sensitivity_lib.differentiate_terms(hess0, dterms1)
        self.assertTrue(np.linalg.norm(sensitivity_lib.evaluate_dketa_depsk(
            hess0, dterms2, eta0, eps0, deps)) > 0)
        assert_array_almost_equal(
            np.einsum('ijk,j, k', true_d2eta_deps2(eps0), deps, deps),
            sensitivity_lib.evaluate_dketa_depsk(
                hess0, dterms2, eta0, eps0, deps))

        dterms3 = sensitivity_lib.differentiate_terms(hess0, dterms2)
        self.assertTrue(np.linalg.norm(sensitivity_lib.evaluate_dketa_depsk(
            hess0, dterms3, eta0, eps0, deps)) > 0)

        assert_array_almost_equal(
            np.einsum('ijkl,j,k,l', true_d3eta_deps3(eps0), deps, deps, deps),
            sensitivity_lib.evaluate_dketa_depsk(
                hess0, dterms3, eta0, eps0, deps))

        ###################################
        # Test the Taylor series itself.

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            taylor_expansion = \
                sensitivity_lib.ParametricSensitivityTaylorExpansion(
                    objective_function=objective,
                    input_val0=eta0,
                    hyper_val0=eps0,
                    order=3,
                    hess0=hess0)

        taylor_expansion.print_terms(k=3)

        d1 = np.einsum('ij,j', true_deta_deps(eps0), deps)
        d2 = np.einsum('ijk,j,k', true_d2eta_deps2(eps0), deps, deps)
        d3 = np.einsum('ijkl,j,k,l', true_d3eta_deps3(eps0), deps, deps, deps)

        assert_array_almost_equal(
            d1, taylor_expansion.evaluate_dkinput_dhyperk(deps, k=1))

        assert_array_almost_equal(
            d2, taylor_expansion.evaluate_dkinput_dhyperk(deps, k=2))

        assert_array_almost_equal(
            d3, taylor_expansion.evaluate_dkinput_dhyperk(deps, k=3))

        assert_array_almost_equal(
            eta0 + d1, taylor_expansion.evaluate_taylor_series(
                deps, max_order=1))

        assert_array_almost_equal(
            eta0 + d1 + 0.5 * d2,
            taylor_expansion.evaluate_taylor_series(deps, max_order=2))

        assert_array_almost_equal(
            eta0 + d1 + d2 / 2 + d3 / 6,
            taylor_expansion.evaluate_taylor_series(deps, max_order=3))

        assert_array_almost_equal(
            eta0 + d1 + d2 / 2 + d3 / 6,
            taylor_expansion.evaluate_taylor_series(deps))


if __name__ == '__main__':
    unittest.main()
