##########################################################################
# Functions for evaluating the sensitivity of optima to hyperparameters. #
##########################################################################

import autograd
import autograd.numpy as np
from copy import deepcopy
from math import factorial
import scipy as sp
import scipy.sparse
from scipy.sparse import coo_matrix
import warnings

from paragami import FlattenFunctionInput
from .solver_lib import SystemSolver


##############
# LRVB class #
##############

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

        method = 'factorization' if factorize_hessian else 'cg'
        self.hess_solver = SystemSolver(self._hess0, method)

        if validate:
            # Check that the gradient of the objective is zero at the optimum.
            grad0 = self._obj_fun_grad(self._opt0)
            newton_step = -1 * self.hess_solver.solve(grad0)

            newton_step_norm = np.linalg.norm(newton_step)
            if newton_step_norm > grad_tol:
                err_msg = \
                    'The gradient is not zero at the proposed optimal ' + \
                    'values.  ||newton_step|| = {} > {} = grad_tol'.format(
                        newton_step_norm, grad_tol)
                raise ValueError(err_msg)

    # Methods:
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

        # return moment_jacobian1 @ cho_solve(
        #     self._hess0_chol, moment_jacobian2.T)
        return moment_jacobian1 @ self.hess_solver.solve(moment_jacobian2.T)

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


class HyperparameterSensitivityLinearApproximation:
    """
    Linearly approximate dependence of an optimum on a hyperparameter.

    Suppose we have an optimization problem in which the objective
    depends on a hyperparameter:

    .. math::

        \hat{\\theta} = \mathrm{argmin}_{\\theta} f(\\theta, \\lambda).

    The optimal parameter, :math:`\hat{\\theta}`, is a function of
    :math:`\\lambda` through the optimization problem.  In general, this
    dependence is complex and nonlinear.  To approximate this dependence,
    this class uses the linear approximation:

    .. math::

        \hat{\\theta}(\\lambda) \\approx \hat{\\theta}(\\lambda_0) +
            \\frac{d\hat{\\theta}}{d\\lambda}|_{\\lambda_0}
                (\\lambda - \\lambda_0).

    In terms of the arguments to this function,
    :math:`\\theta` corresponds to ``opt_par``,
    :math:`\\lambda` corresponds to ``hyper_par``,
    and :math:`f` corresponds to ``objective_fun``.

    Methods
    ------------
    set_base_values:
        Set the base values, :math:`\\lambda_0` and
        :math:`\\theta_0 := \hat\\theta(\\lambda_0)`, at which the linear
        approximation is evaluated.
    get_dopt_dhyper:
        Return the Jacobian matrix
        :math:`\\frac{d\hat{\\theta}}{d\\lambda}|_{\\lambda_0}` in flattened
        space.
    get_hessian_at_opt:
        Return the Hessian of the objective function in the
        flattened space.
    predict_opt_par_from_hyper_par:
        Use the linear approximation to predict
        the value of ``opt_par`` from a value of ``hyper_par``.
    """
    def __init__(
        self,
        objective_fun,
        opt_par_value,
        hyper_par_value,
        validate_optimum=False,
        hessian_at_opt=None,
        cross_hess_at_opt=None,
        factorize_hessian=True,
        hyper_par_objective_fun=None,
        grad_tol=1e-8):
        """
        Parameters
        --------------
        objective_fun : `callable`
            The objective function taking two positional arguments,
            - ``opt_par``: The parameter to be optimized (`numpy.ndarray` (N,))
            - ``hyper_par``: A hyperparameter (`numpy.ndarray` (N,))
            and returning a real value to be minimized.
        opt_par_value :  `numpy.ndarray` (N,)
            The value of ``opt_par`` at which ``objective_fun`` is
            optimized for the given value of ``hyper_par_value``.
        hyper_par_value : `numpy.ndarray` (M,)
            The value of ``hyper_par`` at which ``opt_par`` optimizes
            ``objective_fun``.
        validate_optimum : `bool`, optional
            When setting the values of ``opt_par`` and ``hyper_par``, check
            that ``opt_par`` is, in fact, a critical point of
            ``objective_fun``.
        hessian_at_opt : `numpy.ndarray` (N,N), optional
            The Hessian of ``objective_fun`` at the optimum.  If not specified,
            it is calculated using automatic differentiation.
        cross_hess_at_opt : `numpy.ndarray`  (N, M)
            Optional.  The second derivative of the objective with respect to
            ``input_val`` then ``hyper_val``.
            If not specified it is calculated at initialization.
        factorize_hessian : `bool`, optional
            If ``True``, solve the required linear system using a Cholesky
            factorization.  If ``False``, use the conjugate gradient algorithm
            to avoid forming or inverting the Hessian.
        hyper_par_objective_fun : `callable`, optional
            The part of ``objective_fun`` depending on both ``opt_par`` and
            ``hyper_par``.  The arguments must be the same as
            ``objective_fun``:
            - ``opt_par``: The parameter to be optimized (`numpy.ndarray` (N,))
            - ``hyper_par``: A hyperparameter (`numpy.ndarray` (N,))
            This can be useful if only a small part of the objective function
            depends on both ``opt_par`` and ``hyper_par``.  If not specified,
            ``objective_fun`` is used.
        grad_tol : `float`, optional
            The tolerance used to check that the gradient is approximately
            zero at the optimum.
        """

        self._objective_fun = objective_fun
        self._obj_fun_grad = autograd.grad(self._objective_fun, argnum=0)
        self._obj_fun_hessian = autograd.hessian(self._objective_fun, argnum=0)
        self._obj_fun_hvp = autograd.hessian_vector_product(
            self._objective_fun, argnum=0)

        if hyper_par_objective_fun is None:
            self._hyper_par_objective_fun = self._objective_fun
        else:
            self._hyper_par_objective_fun = hyper_par_objective_fun

        # TODO: is this the right default order?  Make this flexible.
        self._hyper_obj_fun_grad = \
            autograd.grad(self._hyper_par_objective_fun, argnum=0)
        self._hyper_obj_cross_hess = autograd.jacobian(
            self._hyper_obj_fun_grad, argnum=1)

        self._grad_tol = grad_tol

        self.set_base_values(
            opt_par_value, hyper_par_value,
            hessian_at_opt, cross_hess_at_opt,
            factorize_hessian,
            validate_optimum=validate_optimum,
            grad_tol=self._grad_tol)

    def set_base_values(self,
                        opt_par_value, hyper_par_value,
                        hessian_at_opt, cross_hess_at_opt,
                        factorize_hessian,
                        validate_optimum=True, grad_tol=None):

        # Set the values of the optimal parameters.
        self._opt0 = deepcopy(opt_par_value)
        self._hyper0 = deepcopy(hyper_par_value)

        # Set the values of the Hessian at the optimum.
        if hessian_at_opt is None:
            self._hess0 = self._obj_fun_hessian(self._opt0, self._hyper0)
        else:
            self._hess0 = hessian_at_opt
        if self._hess0.shape != (len(self._opt0), len(self._opt0)):
            raise ValueError('``hessian_at_opt`` is the wrong shape.')

        method = 'factorization' if factorize_hessian else 'cg'
        self.hess_solver = SystemSolver(self._hess0, method)

        if validate_optimum:
            if grad_tol is None:
                grad_tol = self._grad_tol

            # Check that the gradient of the objective is zero at the optimum.
            grad0 = self._obj_fun_grad(self._opt0, self._hyper0)
            newton_step = -1 * self.hess_solver.solve(grad0)

            newton_step_norm = np.linalg.norm(newton_step)
            if newton_step_norm > grad_tol:
                err_msg = \
                    'The gradient is not zero at the proposed optimal ' + \
                    'values.  ||newton_step|| = {} > {} = grad_tol'.format(
                        newton_step_norm, grad_tol)
                raise ValueError(err_msg)

        if cross_hess_at_opt is None:
            self._cross_hess = self._hyper_obj_cross_hess(self._opt0, self._hyper0)
        else:
            self._cross_hess = cross_hess_at_opt
        if self._cross_hess.shape != (len(self._opt0), len(self._hyper0)):
            raise ValueError('``cross_hess_at_opt`` is the wrong shape.')

        self._sens_mat = -1 * self.hess_solver.solve(self._cross_hess)


    # Methods:
    def get_dopt_dhyper(self):
        return self._sens_mat

    def get_hessian_at_opt(self):
        return self._hess0

    def predict_opt_par_from_hyper_par(self, new_hyper_par_value):
        """
        Predict ``opt_par`` using the linear approximation.

        Parameters
        ------------
        new_hyper_par_value: `numpy.ndarray` (M,)
            The value of ``hyper_par`` at which to approximate ``opt_par``.
        """
        return \
            self._opt0 + \
            self._sens_mat @ (new_hyper_par_value - self._hyper0)


################################
# Higher-order approximations. #
################################

def _append_jvp(fun, num_base_args=1, argnum=0):
    """
    Append a jacobian vector product to a function.

    This function is designed to be used recursively to calculate
    higher-order Jacobian-vector products.

    Parameters
    --------------
    fun: Callable function
        The function to be differentiated.
    num_base_args: integer
        The number of inputs to the base function, i.e.,
        to the function before any differentiation.
     argnum: inteeger
        Which argument should be differentiated with respect to.
        Must be between 0 and num_base_args - 1.

    Returns
    ------------
    Denote the base args x1, ..., xB, where B == num_base_args.
    Let argnum = k.  Then _append_jvp returns a function,
    fun_jvp(x1, ..., xB, ..., v) =
    \sum_i (dfun_dx_{ki}) v_i | (x1, ..., xB).
    That is, it returns the Jacobian vector product where the Jacobian
    is taken with respect to xk, and the vector product is with the
    final argument.
    """
    assert argnum < num_base_args

    fun_jvp = autograd.make_jvp(fun, argnum=argnum)
    def obj_jvp_wrapper(*argv):
        # These are the base arguments -- the points at which the
        # Jacobians are evaluated.
        base_args = argv[0:num_base_args]

        # The rest of the arguments are the vectors, with which inner
        # products are taken in the order they were passed to
        # _append_jvp.
        vec_args = argv[num_base_args:]

        if (len(vec_args) > 1):
            # Then this is being applied to an existing Jacobian
            # vector product.  The last will be the new vector, and
            # we need to evaluate the function at the previous vectors.
            # The new jvp will be appended to the end of the existing
            # list.
            old_vec_args = vec_args[:-1]
            return fun_jvp(*base_args, *old_vec_args)(vec_args[-1])[1]
        else:
            return fun_jvp(*base_args)(*vec_args)[1]

    return obj_jvp_wrapper


class DerivativeTerm:
    """
    A single term in a Taylor expansion of a two-parameter objective with
    methods for computing its derivatives.

    .. note::
        This class is intended for internal use.  Most users should not
        use ``DerivativeTerm`` directly, and should rather use
        ``ParametricSensitivityTaylorExpansion``.

    Let :math:`\hat{\\eta}(\\epsilon)` be such that
    :math:`g(\hat{\\eta}(\\epsilon), \\epsilon) = 0`.
    The nomenclature assumes that
    the term arises from calculating total derivatives of
    :math:`g(\hat{\\eta}(\\epsilon), \\epsilon)`,
    with respect to :math:`\\epsilon`, so such a term arose from repeated
    applications of the chain and product rule of differentiation with respect
    to :math:`\\epsilon`.

    In the ``ParametricSensitivityTaylorExpansion`` class, such terms are
    then used to calculate

    .. math::
        \\frac{d^k\hat{\\eta}}{d\\epsilon^k} |_{\\eta_0, \\epsilon_0}.

    We assume the term will only be calculated summed against a single value
    of :math:`\\Delta\\epsilon`, so we do not need to keep track of the
    order in which the derivatives are evaluated.

    Every term arising from differentiation of :math:`g(\hat{\\eta}(\\epsilon),
    \\epsilon)` with respect to :math:`\\epsilon` is a product the following
    types of terms.

    First, there are the partial derivatives of :math:`g` itself.

    .. math::
        \\frac{\\partial^{m+n} g(\\eta, \\epsilon)}
              {\\partial \\eta^m \\epsilon^n}

    In the preceding display, ``m``
    is the total number of :math:`\\eta` derivatives, i.e.
    ``m = np.sum(eta_orders)``, and ``n = eps_order``.

    Each partial derivative of :math:`g` with respect to :math:`\\epsilon`
    will multiply one :math:`\\Delta \\epsilon` term directly.  Each
    partial derivative with respect to :math:`\\eta` will multiply a term
    of the form

    .. math::
        \\frac{d^p \hat{\\eta}}{d \\epsilon^p}

    which will in turn multiply :math:`p` different :math:`\\Delta \\epsilon`
    terms. The number of such terms of order :math:`p` are given by the entry
    ``eta_orders[p - 1]``.  Each such terms arises from a single partial
    derivative of :math:`g` with respect to :math:`\\eta`, which is why
    the above ``m = np.sum(eta_orders)``.

    Finally, the term is multiplied by the constant ``prefactor``.

    For example, suppose that ``eta_orders = [1, 0, 2]``, ``prefactor = 1.5``,
    and ``epsilon_order = 2``.  Then the derivative term is

    .. math::
        1.5 \\cdot
        \\frac{\\partial^{5} g(\hat{\\eta}, \\epsilon)}
              {\\partial \\eta^3 \\epsilon^2} \\cdot
        \\frac{d \hat{\\eta}}{d \\epsilon} \\cdot
        \\frac{d^3 \hat{\\eta}}{d \\epsilon^3} \\cdot
        \\frac{d^3 \hat{\\eta}}{d \\epsilon^3} \\cdot

    ...which will multiply a total of
    ``9 = epsilon_order + np.sum(eta_orders * [1, 2, 3])``
    :math:`\\Delta \\epsilon` terms.  Such a term would arise in
    the 9-th order Taylor expansion of :math:`g(\hat{\\eta}(\\epsilon),
    \\epsilon)` in :math:`\\epsilon`.

    Attributes
    -----------------
    eps_order:
        The total number of epsilon derivatives of g.
    eta_orders:
        A vector of length order - 1.  Entry i contains the number
        of terms of the form d\eta^{i + 1} / d\epsilon^{i + 1}.
    prefactor:
        The constant multiple in front of this term.

    Methods
    ------------
    evaluate:
        Get the value of the current derivative term.
    differentiate:
        Get a list of derivatives terms resulting from differentiating this
        term.
    check_similarity:
        Return a boolean indicating whether this term is equivalent to another
        term in the order of its derivative.
    combine_with:
        Return the sum of this term and another term.
    """
    def __init__(self, eps_order, eta_orders, prefactor,
                 eval_g_derivs):
        """
        Parameters
        -------------
        eps_order:
            The total number of epsilon derivatives of g.
        eta_orders:
            A vector of length order - 1.  Entry i contains the number
            of terms :math:`d\\eta^{i + 1} / d\\epsilon^{i + 1}`.
        prefactor:
            The constant multiple in front of this term.
        eval_g_derivs:
            A list of lists of g jacobian vector product functions.
            The array should be such that
            eval_g_derivs[i][j](eta0, eps0, v1 ... vi, w1 ... wj)
            evaluates d^{i + j} G / (deta^i)(deps^j)(v1 ... vi)(w1 ... wj).
        """
        # Base properties.
        self.eps_order = eps_order
        self.eta_orders = eta_orders
        self.prefactor = prefactor
        #self._eval_eta_derivs = eval_eta_derivs
        self._eval_g_derivs = eval_g_derivs

        # Derived quantities.

        # The order is the total number of epsilon derivatives.
        self._order = int(
            self.eps_order + \
            np.sum(self.eta_orders * np.arange(1, len(self.eta_orders) + 1)))

        # The derivative of g needed for this particular term.
        self.eval_g_deriv = \
            eval_g_derivs[np.sum(eta_orders)][self.eps_order]

        # Sanity checks.
        # The rules of differentiation require that these assertions be true
        # -- that is, if terms are generated using the differentiate()
        # method from other well-defined terms, these assertions should always
        # be sastisfied.
        assert isinstance(self.eps_order, int)
        assert len(self.eta_orders) == self._order
        assert self.eps_order >= 0 # Redundant
        for eta_order in self.eta_orders:
            assert eta_order >= 0
            assert isinstance(eta_order, int)
        # assert len(self._eval_eta_derivs) >= self._order - 1
        assert len(eval_g_derivs) > len(self.eta_orders)
        for eta_deriv_list in eval_g_derivs:
            assert len(eta_deriv_list) > self.eps_order

    def __str__(self):
        return 'Order: {}\t{} * eta{} * eps[{}]'.format(
            self._order, self.prefactor, self.eta_orders, self.eps_order)

    def evaluate(self, eta0, eps0, deps, eta_derivs):
        """Evaluate the DerivativeTerm.

        Parameters
        ----------------------
        eta0, eps0 : `numpy.ndarray`
            Where to evaluate the derivative.
        deps : `numpy.ndarray`
            The direction in which to evaluate the derivative.
        eta_derivs : `list` of `numpy.ndarray`
            A list where ``eta_derivs[i]`` contains
            :math:`d\\eta^i / d\\epsilon^i \\Delta \\epsilon^i`.
        """

        # TODO: this check is best done elsewhere
        if len(eta_derivs) < self._order - 1:
            raise ValueError('Not enough derivatives in ``eta_derivs``.')

        # First eta arguments, then epsilons.
        vec_args = []

        for i in range(len(self.eta_orders)):
            eta_order = self.eta_orders[i]
            if eta_order > 0:
                # vec = self._eval_eta_derivs[i](eta0, eps0, deps)
                vec = eta_derivs[i]
                for j in range(eta_order):
                    vec_args.append(vec)

        for i in range(self.eps_order):
            vec_args.append(deps)

        return self.prefactor * self.eval_g_deriv(eta0, eps0, *vec_args)

    def differentiate(self):
        derivative_terms = []
        # new_eval_eta_derivs = deepcopy(self._eval_eta_derivs)
        #new_eval_eta_derivs.append(eval_next_eta_deriv)

        old_eta_orders = deepcopy(self.eta_orders)
        old_eta_orders.append(0)

        # dG / deps.
        derivative_terms.append(
            DerivativeTerm(
                eps_order=self.eps_order + 1,
                eta_orders=deepcopy(old_eta_orders),
                prefactor=self.prefactor,
                # eval_eta_derivs=new_eval_eta_derivs,
                eval_g_derivs=self._eval_g_derivs))

        # dG / deta.
        new_eta_orders = deepcopy(old_eta_orders)
        new_eta_orders[0] = new_eta_orders[0] + 1
        derivative_terms.append(
            DerivativeTerm(
                eps_order=self.eps_order,
                eta_orders=new_eta_orders,
                prefactor=self.prefactor,
                # eval_eta_derivs=new_eval_eta_derivs,
                eval_g_derivs=self._eval_g_derivs))

        # Derivatives of each d^{i}eta / deps^i term.
        for i in range(len(self.eta_orders)):
            eta_order = self.eta_orders[i]
            if eta_order > 0:
                new_eta_orders = deepcopy(old_eta_orders)
                new_eta_orders[i] = new_eta_orders[i] - 1
                new_eta_orders[i + 1] = new_eta_orders[i + 1] + 1
                derivative_terms.append(
                    DerivativeTerm(
                        eps_order=self.eps_order,
                        eta_orders=new_eta_orders,
                        prefactor=self.prefactor * eta_order,
                        # eval_eta_derivs=new_eval_eta_derivs,
                        eval_g_derivs=self._eval_g_derivs))

        return derivative_terms

    # Return whether another term matches this one in the pattern of derivatives.
    def check_similarity(self, term):
        return \
            (self.eps_order == term.eps_order) & \
            (self.eta_orders == term.eta_orders)

    # Assert that another term has the same pattern of derivatives and
    # return a new term that combines the two.
    def combine_with(self, term):
        assert self.check_similarity(term)
        return DerivativeTerm(
            eps_order=self.eps_order,
            eta_orders=self.eta_orders,
            prefactor=self.prefactor + term.prefactor,
            # eval_eta_derivs=self._eval_eta_derivs,
            eval_g_derivs=self._eval_g_derivs)


def _generate_two_term_fwd_derivative_array(fun, order):
    """
    Generate an array of JVPs of the two arguments of the target function fun.

    Parameters
    -------------
    fun: callable function
        The function to be differentiated.  The first two arguments
        should be vectors for differentiation, i.e., fun should have signature
        fun(x1, x2, ...) and return a numeric value.
     order: integer
        The maximum order of the derivative to be generated.

    Returns
    ------------
    An array of functions where element eval_fun_derivs[i][j] is a function
    ``eval_fun_derivs[i][j](x1, x2, ..., v1, ... vi, w1, ..., wj)) =
    d^{i + j}fun / (dx1^i dx2^j) v1 ... vi w1 ... wj``.
    """
    eval_fun_derivs = [[ fun ]]
    for x1_ind in range(order):
        if x1_ind > 0:
            # Append one x1 derivative.
            next_deriv = _append_jvp(
                eval_fun_derivs[x1_ind - 1][0], num_base_args=2, argnum=0)
            eval_fun_derivs.append([ next_deriv ])
        for x2_ind in range(order):
            # Append one x2 derivative.
            next_deriv = _append_jvp(
                eval_fun_derivs[x1_ind][x2_ind], num_base_args=2, argnum=1)
            eval_fun_derivs[x1_ind].append(next_deriv)
    return eval_fun_derivs



def _consolidate_terms(dterms):
    """
    Combine like derivative terms.

    Arguments
    -----------
    dterms:
        A list of DerivativeTerms.

    Returns
    ------------
    A new list of derivative terms that evaluate equivalently where
    terms with the same derivative signature have been combined.
    """
    unmatched_indices = [ ind for ind in range(len(dterms)) ]
    consolidated_dterms = []
    while len(unmatched_indices) > 0:
        match_term = dterms[unmatched_indices.pop(0)]
        for ind in unmatched_indices:
            if (match_term.eta_orders == dterms[ind].eta_orders):
                match_term = match_term.combine_with(dterms[ind])
                unmatched_indices.remove(ind)
        consolidated_dterms.append(match_term)

    return consolidated_dterms


# Get the terms to start a Taylor expansion.
def _get_taylor_base_terms(eval_g_derivs):
    dterms1 = [ \
        DerivativeTerm(
            eps_order=1,
            eta_orders=[0],
            prefactor=1.0,
            eval_g_derivs=eval_g_derivs),
        DerivativeTerm(
            eps_order=0,
            eta_orders=[1],
            prefactor=1.0,
            eval_g_derivs=eval_g_derivs) ]
    return dterms1


class ParametricSensitivityTaylorExpansion(object):
    """
    Evaluate the Taylor series of an optimum on a hyperparameter.

    This is a class for computing the Taylor series of
    eta(eps) = argmax_eta objective(eta, eps) using forward-mode automatic
    differentation.

    .. note:: This class is experimental and should be used with caution.
    """
    def __init__(self, objective_function,
                 input_val0, hyper_val0, order,
                 hess0=None,
                 hyper_par_objective_function=None):
        """
        Parameters
        ------------------
        objective_function : `callable`
            The optimization objective as a function of two arguments
            (eta, eps), where eta is the parameter that is optimized and
            eps is a hyperparameter.
        input_val0 : `numpy.ndarray` (N,)
            The value of ``input_par`` at the optimum.
        hyper_val0 : `numpy.ndarray` (M,)
            The value of ``hyper_par`` at which ``input_val0`` was found.
        order : `int`
            The maximum order of the Taylor series to be calculated.
        hess0 : `numpy.ndarray` (N, N)
            Optional.  The Hessian of the objective at
            (``input_val0``, ``hyper_val0``).
            If not specified it is calculated at initialization.
        hyper_par_objective_function : `callable`
            Optional.  A function containing the dependence
            of ``objective_function`` on the hyperparameter.  Sometimes
            only a small, easily calculated part of the objective depends
            on the hyperparameter, and by specifying
            ``hyper_par_objective_function`` the
            necessary calculations can be more efficient.  If
            unset, ``objective_function`` is used.
        """
        warnings.warn(
            'The ParametricSensitivityTaylorExpansion is experimental.')
        self._objective_function = objective_function
        self._objective_function_hessian = \
            autograd.hessian(self._objective_function, argnum=0)

        # In order to calculate derivatives d^kinput_dhyper^k, we will be
        # Taylor expanding the gradient of the objective with respect to eta.
        self._objective_function_eta_grad = \
            autograd.grad(self._objective_function, argnum=0)

        if hyper_par_objective_function is None:
            self._hyper_par_objective_function = self._objective_function
        else:
            self._hyper_par_objective_function = hyper_par_objective_function

        self.set_base_values(input_val0, hyper_val0, hess0=hess0)
        self._set_order(order)

    def set_base_values(self, input_val0, hyper_val0, hess0=None):
        """
        Set the values at which the Taylor series is to be evaluated.

        Parameters
        ---------------
        input_val0: `numpy.ndarray` (N,)
            The value of input_par at the optimum.
        hyper_val0: `numpy.ndarray` (M,)
            The value of hyper_par at which input_val0 was found.
        hess0: `numpy.ndarray` (N, N)
            Optional.  The Hessian of the objective at (input_val0, hyper_val0).
            If not specified it is calculated at initialization.
        """
        self._input_val0 = deepcopy(input_val0)
        self._hyper_val0 = deepcopy(hyper_val0)

        if hess0 is None:
            self._hess0 = \
                self._objective_function_hessian(
                    self._input_val0, self._hyper_val0)
        else:
            self._hess0 = hess0

        # TODO: if the objective function returns a 1-d array and not a
        # float then the Cholesky decomposition will fail because
        # the Hessian will have an extra dimension.  This is a confusing
        # error that we could catch explicitly at the cost of an extra
        # function evaluation.  Is it worth it?
        self.hess_solver = SystemSolver(self._hess0, 'factorization')

    def _differentiate_terms(self, dterms):
        dterms_derivs = []
        for term in dterms:
            dterms_derivs += term.differentiate()
        return _consolidate_terms(dterms_derivs)

    def _set_order(self, order):
        """Generate the matrix of g partial derivatives and differentiate
        the Taylor series up to the required order.
        """
        self._order = order

        # You need one more gradient derivative than the order of the Taylor
        # approximation.
        self._eval_g_derivs = _generate_two_term_fwd_derivative_array(
            self._objective_function_eta_grad, order=self._order + 1)

        self._taylor_terms_list = \
            [ _get_taylor_base_terms(self._eval_g_derivs) ]
        for k in range(1, self._order):
            next_taylor_terms = \
                self._differentiate_terms(self._taylor_terms_list[k - 1])
            self._taylor_terms_list.append(next_taylor_terms)

    def get_max_order(self):
        return self._order

    def _evaluate_dkinput_dhyperk(self, dhyper, input_derivs, k):
        """
        Evaluate the derivative d^k input / d hyper^k in the direction dhyper.

        Parameters
        --------------
        dhyper : `numpy.ndarray` (N, )
            The direction ``new_hyper_val - hyper_val0`` in which to evaluate
            the directional derivative.
        input_derivs : `list` of `numpy.ndarray`
            A list of previous dkinput_dhyperk up to order k - 1.
        k : `int`
            The order of the derivative.

        Returns
        ------------
            The value of the k^th derivative in the directoin dhyper.
        """
        if k <= 0:
            raise ValueError('k must be at least one.')
        if k > self._order:
            raise ValueError(
                'k must be no greater than the declared order={}'.format(
                    self._order))
        if len(input_derivs) < k - 1:
            raise ValueError('Not enough eta_derivs provided.')
        vec = np.zeros_like(self._input_val0)
        for term in self._taylor_terms_list[k - 1]:
            # Exclude the highest order eta derivative -- this what
            # we are trying to calculate.
            if (term.eta_orders[-1] == 0):
                vec += \
                    term.evaluate(
                        eta0=self._input_val0,
                        eps0=self._hyper_val0,
                        deps=dhyper,
                        eta_derivs=input_derivs)
        return -1 * self.hess_solver.solve(vec)


    def evaluate_input_derivs(self, dhyper, max_order=None):
        """Return a list of the derivatives dkinput / dhyperk dhyper^k
        """
        if max_order is None:
            max_order = self._order
        input_derivs = []
        for k in range(1, max_order + 1):
            dinputk_dhyperk = \
                self._evaluate_dkinput_dhyperk(
                    dhyper=dhyper,
                    input_derivs=input_derivs,
                    k=k)
            input_derivs.append(dinputk_dhyperk)
        return input_derivs


    def evaluate_taylor_series_terms(self, new_hyper_val, add_offset=True,
                                     max_order=None):
        """Return the terms in a Taylor series approximation.
        """
        if max_order is None:
            max_order = self._order
        if max_order <= 0:
            raise ValueError('max_order must be greater than zero.')
        if max_order > self._order:
            raise ValueError(
                'max_order must be no greater than the order={}'.format(
                    self._order))

        if add_offset:
            dinput_terms = [self._input_val0]
        else:
            dinput_terms = [np.zeros_like(self._input_val0)]
        dhyper = new_hyper_val - self._hyper_val0
        input_derivs = \
            self.evaluate_input_derivs(dhyper, max_order=max_order)

        for k in range(1, max_order + 1):
            dinput_terms.append(input_derivs[k - 1] / float(factorial(k)))

        return dinput_terms


    def evaluate_taylor_series(self, new_hyper_val,
                               add_offset=True, max_order=None,
                               sum_terms=True):
        """
        Evaluate the derivative ``d^k input / d hyper^k`` in the direction dhyper.

        Parameters
        --------------
        new_hyper_val: `numpy.ndarray` (N, )
            The new hyperparameter value at which to evaluate the
            Taylor series.
        add_offset: `bool`
            Optional.  Whether to add the initial constant input_val0 to the
            Taylor series.
        max_order: `int`
            Optional.  The order of the Taylor series.  Defaults to the
            ``order`` argument to ``__init__``.
        sum_terms: `bool`
            If ``True``, add the terms in the Taylor series.  If ``False``,
            return the terms as a list.

        Returns
        ------------
            The Taylor series approximation to ``input_val(new_hyper_val)`` if
            ``add_offset`` is ``True``, or to
            ``input_val(new_hyper_val) - input_val0`` if ``False``.  If
            ``sum_terms`` is ``True``, then a vector of the same length as
            ``input_val`` is returned.  Otherwise, an array of
            shape ``max_order + 1, len(input_val)`` is returned containing
            the terms of the Taylor series approximation.
        """

        dinput_terms = self.evaluate_taylor_series_terms(
            new_hyper_val=new_hyper_val,
            add_offset=add_offset,
            max_order=max_order)
        return np.sum(dinput_terms, axis=0)


    def print_terms(self, k=None):
        """
        Print the derivative terms in the Taylor series.

        Parameters
        ---------------
        k: integer
            Optional.  Which term to print.  If unspecified, all terms are
            printed.
        """
        if k is not None and k > self._order:
            raise ValueError(
                'k must be no greater than order={}'.format(self._order))
        for order in range(self._order):
            if k is None or order == (k - 1):
                print('\nTerms for order {}:'.format(order + 1))
                for term in self._taylor_terms_list[order]:
                    print(term)


class SparseBlockHessian():
    """Efficiently calculate block-sparse Hessians.

        The objective function is expected to be of the form

        .. math ::
            x = (x_1 , ... , x_G) \\textrm{ (or some permutation thereof)}

            f(x) = \sum_{g=1}^{G} f_g(x_g)

        Each :math:`x_g` is
        expected to have the same dimension.  Consequently, the Hessian
        matrix of ``f`` with respect to ``x``, is block diagonal with
        ``G`` blocks, up to a permutation of the order of ``x``.
        The purpose of this class is to efficiently calculate
        this Hessian when the block structure (i.e., the partition of ``x``)
        is known.

    """
    def __init__(self, objective_function, sparsity_array):
        """In terms of the class description, ``objective_function = f``,
        ``opt_par = x``, and ``sparsity_array`` describes
        the partition of ``x`` into :math:`(x_1, ..., x_G)`.

        Parameters
        ------------
        objective_function : `callable`
            An objective function of which to calculate a Hessian.   The
            argument should be

            - ``opt_par``: `numpy.ndarray` (N,) The optimization parameter.

        sparsity_array : `numpy.ndarray` (G, M)
            An array containing the indices of rows and columns of each block.
            The Hessian should contain ``G`` dense blocks, each of which
            is ``M`` by ``M``.  Each row of ``sparsity_array`` should contain
            the indices of the corresponding block.  There must be no repeated
            indices, and each block must be the same size.
        """
        self._fun = objective_function
        self._sparsity_array = sparsity_array
        self._num_blocks = self._sparsity_array.shape[0]
        self._block_size = self._sparsity_array.shape[1]

        if len(np.unique(sparsity_array)) != len(sparsity_array.flatten()):
            raise ValueError(
                'The indices in ``sparsity array`` must be unique.')

        self._f_grad = autograd.grad(self._fun, argnum=0)
        self._f_fwd_hess = _append_jvp(self._f_grad, num_base_args=1)

    def _hess_summed_term(self, opt_par, ib):
        """``ib`` is the index within the block.
        """
        v = np.zeros_like(opt_par)
        v[self._sparsity_array[:, ib]] = 1
        return self._f_fwd_hess(opt_par, v)

    def get_block_hessian(self, opt_par, print_every=0):
        """Get the block Hessian at ``opt_par`` and ``weights``.

        Parmeters
        ----------
        opt_par : `numpy.ndarray`
            The argument to ``objective_function`` at which to evaluate
            the Hessian matrix.
        print_every : `int`, optional.
            How often to display progress.  If ``0``, nothing is printed.

        Returns
        --------
        hessian : `scipy.sparse.coo_matrix` (N, N)
            The block-sparse Hessian given by and ``sparsity_array``.
        """
        opt_par = np.atleast_1d(opt_par)
        if opt_par.ndim != 1:
            raise ValueError('``opt_par`` must be a vector.')

        mat_vals = [] # These will be the entries of the Hessian
        mat_rows = [] # These will be the row indices
        mat_cols = [] # These will be the column indices

        for ib in range(self._block_size):
            if print_every > 0:
                if ib % print_every == 0:
                    print('Block index {} of {}.'.format(ib, self._block_size))
            hess_prod = self._hess_summed_term(opt_par, ib)
            for b in range(self._num_blocks):
                hess_inds = self._sparsity_array[b, :]
                mat_vals.extend(hess_prod[hess_inds])
                mat_rows.extend(hess_inds)
                mat_cols.extend(np.full(self._block_size, hess_inds[ib]))
        if print_every > 0:
            print('Done differentiating.')

        d = len(opt_par)
        h_sparse = coo_matrix((mat_vals, (mat_rows, mat_cols)), (d, d))
        return h_sparse

    def get_global_hessian(self, opt_par, global_inds=None, print_every=0):
        """Get the dense Hessian terms for the global parameters, which
        are, by default, indexed by any indices not in ``_sparsity_array``.
        """
        local_inds = np.hstack(self._sparsity_array)
        if global_inds is None:
            global_inds = np.setdiff1d(np.arange(len(opt_par)), local_inds)

        global_local_intersection = np.intersect1d(global_inds, local_inds)
        if len(global_local_intersection) > 0:
            raise ValueError(
                'The global and local indices must be disjoint.  {}'.format(
                    global_local_intersection))

        mat_vals = [] # These will be the entries of the Hessian
        mat_rows = [] # These will be the row indices
        mat_cols = [] # These will be the column indices

        v = np.zeros_like(opt_par)
        count = 0
        for ig in global_inds:
            if print_every > 0:
                if count % print_every == 0:
                    print('Global index {} of {}.'.format(
                        count, len(global_inds)))
            v[ig] = 1
            hess_row = self._f_fwd_hess(opt_par, v)
            for il in local_inds:
                mat_vals.append(hess_row[il])
                mat_cols.append(ig)
                mat_rows.append(il)

                mat_vals.append(hess_row[il])
                mat_cols.append(il)
                mat_rows.append(ig)

            for ig2 in global_inds:
                mat_vals.append(0.5 * hess_row[ig2])
                mat_cols.append(ig)
                mat_rows.append(ig2)

                mat_vals.append(0.5 * hess_row[ig2])
                mat_cols.append(ig2)
                mat_rows.append(ig)

            v[ig] = 0
            count += 1

        if print_every > 0:
            print('Done differentiating.')

        d = len(opt_par)
        h_sparse = coo_matrix((mat_vals, (mat_rows, mat_cols)), (d, d))
        return h_sparse

    def get_hessian(self, opt_par, print_every=0):
        local_hessian = self.get_block_hessian(opt_par, print_every=print_every)
        global_hessian = self.get_global_hessian(opt_par, print_every=print_every)
        return local_hessian + global_hessian
