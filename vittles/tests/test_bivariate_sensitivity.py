#!/usr/bin/env python3

import autograd
import autograd.numpy as np
import scipy as sp

from numpy.testing import assert_array_almost_equal
import unittest

import vittles
from vittles.bivariate_sensitivity import CrossSensitivity

class CrossSensitivityTest(unittest.TestCase):
    def test_cross(self):
        # For now, just dump the python notebook I used for development.
        dim = 10
        n_obs = 200
        y_var = 1

        theta_true = np.random.random(dim) - 0.5
        x = np.random.random((n_obs, dim))
        x = x - np.mean(x, axis=0)
        y_mean = np.exp(x @ theta_true)
        y = np.array([ np.random.normal(mu, np.sqrt(y_var)) for mu in y_mean ])

        def log_lik(theta, w, x, y, y_var):
            resid = y - np.exp(x @ theta)
            return -0.5 * np.sum(w * (resid ** 2)) / y_var

        def get_objective(w):
            return lambda theta: -1 * log_lik(theta, w, x, y, y_var)

        def optimize(theta_init, w, maxiter):
            obj = get_objective(w)
            return sp.optimize.minimize(
                fun=obj, x0=theta_init, method='BFGS',
                options={'maxiter': maxiter})

        def optimize_well(theta_init, w):
            obj = get_objective(w)
            res = sp.optimize.minimize(
                fun=obj,
                jac=autograd.jacobian(obj),
                hess=autograd.hessian(obj),
                x0=theta_init,
                method='trust-exact')
            print(res.message)
            return res

        w_base = np.ones(n_obs)
        obj = get_objective(w_base)
        theta_init = np.zeros(dim)
        opt_result_0 = optimize_well(theta_init, w_base)
        opt_result_1 = optimize(theta_init, w_base, 5)

        def vecsize(x):
            return np.max(np.abs(x))

        print('Iters for opt:\t', opt_result_1.nit)

        print('With limit:\t', opt_result_0.message)
        print('Without limit:\t', opt_result_1.message)

        obj_grad = autograd.grad(obj)
        obj_hess = autograd.hessian(obj)
        grad_0 = obj_grad(opt_result_0.x)
        grad_1 = obj_grad(opt_result_1.x)

        print('With limit:\t', vecsize(grad_0))
        print('Without limit:\t', vecsize(grad_1))

        # Ok, let's check it out.

        def get_dtheta(theta, new_w):
            hess_base = obj_hess(theta)
            def w_obj(theta, w):
                return -1 * log_lik(theta, w, x, y, y_var)
            w_sens = (vittles.ParametricSensitivityTaylorExpansion.
                optimization_objective)(
                objective_function=w_obj,
                input_val0=theta,
                hyper_val0=w_base,
                order=1,
                hess0=hess_base,
                forward_mode=False,
                max_input_order=None,
                max_hyper_order=1,
                force=True)
            return w_sens.evaluate_taylor_series(new_w) - theta

        new_w = np.ones(n_obs)
        new_w[1] = 0
        dw = new_w - w_base

        dtheta_0 = get_dtheta(opt_result_0.x, new_w)
        dtheta_1 = get_dtheta(opt_result_1.x, new_w)

        new_theta = optimize_well(opt_result_0.x, new_w).x

        # Look at one of the two optima.
        opt_result = opt_result_1

        def pert_obj(theta, lam, w):
            return -1 * log_lik(theta, w, x, y, y_var) - np.dot(lam, theta)

        theta_base = opt_result.x
        g = autograd.jacobian(pert_obj, argnum=0)
        lam_base = obj_grad(opt_result.x)

        print('Base lambda:\t', vecsize(g(theta_base, lam_base, w_base)))
        print('Zero lambda:\t', vecsize(g(theta_base, np.zeros(dim), w_base)))

        # Get the Hessian
        hess_base = obj_hess(theta_base)
        print(np.linalg.eigvals(hess_base))

        grad_base = obj_grad(theta_base)
        newton_step = -1 * np.linalg.solve(hess_base, grad_base)
        print('ns size:\t', vecsize(newton_step))
        print('g without step:\t',
            vecsize(g(theta_base, np.zeros(dim), w_base)))
        print('g after step:\t',
            vecsize(g(theta_base + newton_step, np.zeros(dim), w_base)))

        def get_dtheta_from_lam(theta, lam, new_w):
            hess_base = autograd.hessian(
                lambda theta: pert_obj(theta, lam, w_base))(theta)
            w_sens = (vittles.ParametricSensitivityTaylorExpansion.
                optimization_objective(
                objective_function=lambda theta, w: pert_obj(theta, lam, w),
                input_val0=theta,
                hyper_val0=w_base,
                order=1,
                hess0=hess_base,
                forward_mode=False,
                max_input_order=None,
                max_hyper_order=1,
                force=True))
            return w_sens.evaluate_taylor_series(new_w) - theta

        def optimize_lam(theta_init, lam, w):
            obj = lambda theta: pert_obj(theta, lam, w)
            res = sp.optimize.minimize(
                fun=obj,
                jac=autograd.jacobian(obj),
                hess=autograd.hessian(obj),
                x0=theta_init,
                method='trust-exact',
                options={'maxiter': 10000, 'gtol': 1e-12})
            print(res.message)
            return res.x

        dtheta = get_dtheta_from_lam(theta_base, lam_base, new_w)
        new_theta = optimize_lam(theta_base, lam_base, new_w)
        theta_diff = new_theta - theta_base

        solver = lambda v: np.linalg.solve(hess_base, v)
        cross_sens = CrossSensitivity(
            estimating_equation=g,
            solver=solver,
            input_base=theta_base,
            hyper1_base=lam_base,
            hyper2_base=w_base)

        # If lambda is zero, you are unconstrained.  This is equivalent to a
        # Newton step.
        dlambda = -1 * lam_base

        dtheta_correction = cross_sens.evaluate(dlambda, dw)
        print('Correction:\t',  vecsize(dtheta_correction))
        print('Rel. crctn.:\t', vecsize(dtheta_correction)  / vecsize(dtheta))


        # The target of the improved approximation is the sensitivity at the
        # new optimum.

        print('Uncorrected:\t', np.sum(np.abs(dtheta_0 - dtheta)))
        print('Corrected:\t',
            np.sum(np.abs(dtheta_0 - (dtheta + dtheta_correction))))


        # Let's try a different tack.  Many specifications of the constraint
        # will do; indeed, any invertible function of the gradient will do.

        constrain_grad = True
        theta_base = opt_result.x
        hess_base = obj_hess(theta_base)
        grad_base = obj_grad(theta_base)

        if constrain_grad:
            log_lik_grad = autograd.grad(log_lik, argnum=0)
            def pert_obj(theta, lam, w):
                return -1 * log_lik(theta, w, x, y, y_var) + \
                    np.dot(lam, log_lik_grad(theta, w, x, y, y_var))

            g = autograd.jacobian(pert_obj, argnum=0)
            lam_base = np.linalg.solve(hess_base, grad_base)
        else:
            def pert_obj(theta, lam, w):
                return -1 * log_lik(theta, w, x, y, y_var) + np.dot(lam, theta)
            lam_base = -1 * obj_grad(theta_base)
            g = autograd.jacobian(pert_obj, argnum=0)

        print('Base lambda (should be zero):\t\t',
            vecsize(g(theta_base, lam_base, w_base)))
        print('Zero lambda (should be nonzero):\t',
            vecsize(g(theta_base, np.zeros(dim), w_base)))

        solver = lambda v: np.linalg.solve(hess_base, v)
        cross_sens = CrossSensitivity(
            estimating_equation=g,
            solver=solver,
            input_base=theta_base,
            hyper1_base=lam_base,
            hyper2_base=w_base)

        dlambda = -1 * lam_base

        dtheta_correction = cross_sens.evaluate(dlambda, dw)
        print('Correction:\t',    vecsize(dtheta_correction))
        print('Rel. crctn.:\t', vecsize(dtheta_correction)  / vecsize(dtheta))


        # The target of the improved approximation is the sensitivity at the
        # new optimum.

        print('Uncorrected:\t', np.sum(np.abs(dtheta_0 - dtheta)))
        print('Corrected:\t',
            np.sum(np.abs(dtheta_0 - (dtheta + dtheta_correction))))


if __name__ == '__main__':
    unittest.main()
