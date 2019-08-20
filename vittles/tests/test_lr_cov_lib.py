#!/usr/bin/env python3

import autograd
import autograd.numpy as np

import itertools
from numpy.testing import assert_array_almost_equal
import paragami
import unittest
import warnings

import vittles


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


if __name__ == '__main__':
    unittest.main()
