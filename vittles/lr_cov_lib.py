##############
# LRVB class #
##############

import autograd
from copy import deepcopy
import numpy as np
from . import solver_lib

class LinearResponseCovariances:
    """
    Calculate linear response covariances of a variational distribution.

    Let :math:`q(\\theta | \\eta)` be a class of probability distribtions on
    :math:`\\theta` where the class is parameterized by the real-valued vector
    :math:`\\eta`.  Suppose that we wish to approximate a distribution
    :math:`q(\\theta | \\eta^*) \\approx p(\\theta)` by solving an optimization
    problem :math:`\\eta^* = \\mathrm{argmin} f(\\eta)`.  For example, :math:`f`
    might be a measure of distance between :math:`q(\\theta | \\eta)` and
    :math:`p(\\theta)`.  This class uses the sensitivity of the optimal
    :math:`\\eta^*` to estimate the covariance
    :math:`\\mathrm{Cov}_p(g(\\theta))`. This covariance estimate is called the
    "linear response covariance".

    In this notation, the arguments to the class mathods are as follows.
    :math:`f` is ``objective_fun``, :math:`\\eta^*` is ``opt_par_value``, and
    the function ``calculate_moments`` evaluates :math:`\\mathbb{E}_{q(\\theta |
    \\eta)}[g(\\theta)]` as a function of :math:`\\eta`.

    Methods
    ------------
    set_base_values:
        Set the base values, :math:`\\eta^*` that optimizes the
        objective function.
    get_hessian_at_opt:
        Return the Hessian of the objective function evaluated at the optimum.
    get_hessian_cholesky_at_opt:
        Return the Cholesky decomposition of the Hessian of the objective
        function evaluated at the optimum.
    get_lr_covariance:
        Return the linear response covariance of a given moment.
    """
    def __init__(
        self,
        objective_fun,
        opt_par_value,
        validate_optimum=False,
        hessian_at_opt=None,
        factorize_hessian=True,
        grad_tol=1e-8):
        """
        Parameters
        --------------
        objective_fun: Callable function
            A callable function whose optimum parameterizes an approximate
            Bayesian posterior.  The function must take as a single
            argument a numeric vector, ``opt_par``.
        opt_par_value:
            The value of ``opt_par`` at which ``objective_fun`` is optimized.
        validate_optimum: Boolean
            When setting the values of ``opt_par``, check
            that ``opt_par`` is, in fact, a critical point of
            ``objective_fun``.
        hessian_at_opt: Numeric matrix (optional)
            The Hessian of ``objective_fun`` at the optimum.  If not specified,
            it is calculated using automatic differentiation.
        factorize_hessian: Boolean
            If ``True``, solve the required linear system using a Cholesky
            factorization.  If ``False``, use the conjugate gradient algorithm
            to avoid forming or inverting the Hessian.
        grad_tol: Float
            The tolerance used to check that the gradient is approximately
            zero at the optimum.
        """

        self._obj_fun = objective_fun
        self._obj_fun_grad = autograd.grad(self._obj_fun, argnum=0)
        self._obj_fun_hessian = autograd.hessian(self._obj_fun, argnum=0)
        self._obj_fun_hvp = autograd.hessian_vector_product(
            self._obj_fun, argnum=0)

        self._grad_tol = grad_tol

        self.set_base_values(
            opt_par_value, hessian_at_opt,
            factorize_hessian, validate=validate_optimum)

    def set_base_values(self,
                        opt_par_value,
                        hessian_at_opt,
                        factorize_hessian=True,
                        validate=True,
                        grad_tol=None):
        if grad_tol is None:
            grad_tol = self._grad_tol

        # Set the values of the optimal parameters.
        self._opt0 = deepcopy(opt_par_value)

        # Set the values of the Hessian at the optimum.
        if hessian_at_opt is None:
            self._hess0 = self._obj_fun_hessian(self._opt0)
        else:
            self._hess0 = hessian_at_opt

        self.hess_solver = solver_lib.get_cholesky_solver(self._hess0)

        if validate:
            # Check that the gradient of the objective is zero at the optimum.
            grad0 = self._obj_fun_grad(self._opt0)
            newton_step = -1 * self.hess_solver(grad0)

            newton_step_norm = np.linalg.norm(newton_step)
            if newton_step_norm > grad_tol:
                err_msg = \
                    'The gradient is not zero at the proposed optimal ' + \
                    'values.  ||newton_step|| = {} > {} = grad_tol'.format(
                        newton_step_norm, grad_tol)
                raise ValueError(err_msg)

    def get_hessian_at_opt(self):
        return self._hess0

    def get_lr_covariance_from_jacobians(self,
                                         moment_jacobian1,
                                         moment_jacobian2):
        """
        Get the linear response covariance between two vectors of moments.

        Parameters
        ------------
        moment_jacobian1: 2d numeric array.
            The Jacobian matrix of a map from a value of
            ``opt_par`` to a vector of moments of interest.  Following
            standard notation for Jacobian matrices, the rows should
            correspond to moments and the columns to elements of
            a flattened ``opt_par``.
        moment_jacobian2: 2d numeric array.
            Like ``moment_jacobian1`` but for the second vector of moments.

        Returns
        ------------
        Numeric matrix
            If ``moment_jacobian1(opt_par)`` is the Jacobian
            of :math:`\mathbb{E}_q[g_1(\\theta)]` and
            ``moment_jacobian2(opt_par)``
            is the Jacobian of  :math:`\mathbb{E}_q[g_2(\\theta)]` then this
            returns the linear response estimate of
            :math:`\\mathrm{Cov}_p(g_1(\\theta), g_2(\\theta))`.
        """

        if moment_jacobian1.ndim != 2:
            raise ValueError('moment_jacobian1 must be a 2d array.')

        if moment_jacobian2.ndim != 2:
            raise ValueError('moment_jacobian2 must be a 2d array.')

        if moment_jacobian1.shape[1] != len(self._opt0):
            err_msg = ('The number of rows of moment_jacobian1 must match' +
                       'the dimension of the optimization parameter. ' +
                       'Expected {} rows, but got shape = {}').format(
                         len(self._opt0), moment_jacobian1.shape)
            raise ValueError(err_msg)

        if moment_jacobian2.shape[1] != len(self._opt0):
            err_msg = ('The number of rows of moment_jacobian2 must match' +
                       'the dimension of the optimization parameter. ' +
                       'Expected {} rows, but got shape = {}').format(
                         len(self._opt0), moment_jacobian2.shape)
            raise ValueError(err_msg)

        return moment_jacobian1 @ self.hess_solver(moment_jacobian2.T)

    def get_moment_jacobian(self, calculate_moments):
        """
        The Jacobian matrix of a map from ``opt_par`` to a vector of
        moments of interest.

        Parameters
        ------------
        calculate_moments: Callable function
            A function that takes the folded ``opt_par`` as a single argument
            and returns a numeric vector containing posterior moments of
            interest.

        Returns
        ----------
        Numeric matrix
            The Jacobian of the moments.
        """
        calculate_moments_jacobian = autograd.jacobian(calculate_moments)
        return calculate_moments_jacobian(self._opt0)

    def get_lr_covariance(self, calculate_moments):
        """
        Get the linear response covariance of a vector of moments.

        Parameters
        ------------
        calculate_moments: Callable function
            A function that takes the folded ``opt_par`` as a single argument
            and returns a numeric vector containing posterior moments of
            interest.

        Returns
        ------------
        Numeric matrix
            If ``calculate_moments(opt_par)`` returns
            :math:`\\mathbb{E}_q[g(\\theta)]`
            then this returns the linear response estimate of
            :math:`\\mathrm{Cov}_p(g(\\theta))`.
        """

        moment_jacobian = self.get_moment_jacobian(calculate_moments)
        return self.get_lr_covariance_from_jacobians(
            moment_jacobian, moment_jacobian)
