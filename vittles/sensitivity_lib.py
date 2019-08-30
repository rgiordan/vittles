##########################################################################
# Functions for evaluating the sensitivity of optima to hyperparameters. #
##########################################################################

import autograd
import autograd.numpy as np
from autograd.core import primitive, defvjp, defjvp

from copy import deepcopy
from math import factorial
from string import ascii_lowercase
import warnings

from paragami import FlattenFunctionInput
from . import solver_lib


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
            validate_optimum=validate_optimum,
            grad_tol=self._grad_tol)

    def set_base_values(self,
                        opt_par_value, hyper_par_value,
                        hessian_at_opt, cross_hess_at_opt,
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

        self.hess_solver = solver_lib.get_cholesky_solver(self._hess0)

        if validate_optimum:
            if grad_tol is None:
                grad_tol = self._grad_tol

            # Check that the gradient of the objective is zero at the optimum.
            grad0 = self._obj_fun_grad(self._opt0, self._hyper0)
            newton_step = -1 * self.hess_solver(grad0)

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

        self._sens_mat = -1 * self.hess_solver(self._cross_hess)


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

    def _check_hyper_par_value(self, hyper_par, tolerance=1e-8):
        if np.max(np.abs(hyper_par - self._hyper0)) > tolerance:
            raise ValueError(
                'get_opt_par must be evaluated at ',
                self._hyper0, ' != ', hyper_par)

    def get_opt_par_function(self):
        """Return a differentiable function returning the optimal value.
        """

        @primitive
        def get_opt_par(hyper_par):
            """Evaluate the optimum as a differentiable function.

            Parameters
            -------------
            hyper_par: numeric
                The value must match `hyper_par_value`, but it can be an
                array box, allowing functions of the optimal parameter to be
                differentiated with `autograd`.
            """

            self._check_hyper_par_value(hyper_par)
            return self._opt0

        # Reverse mode.
        def _get_parhat_vjp(ans, hyperpar):
            self._check_hyper_par_value(hyperpar)

            # Make this primitive to prevent attempting higher-order derivatives.
            # To do so efficiently will require a higher order approximation.
            @primitive
            def vjp(g):
                return g.T @ self._sens_mat
            return vjp

        # Forward mode.
        # Make this primitive to prevent attempting higher-order derivatives.
        # To do so efficiently will require a higher order approximation.
        @primitive
        def _get_parhat_jvp(g, ans, hyperpar):
            self._check_hyper_par_value(hyperpar)
            return self._sens_mat @ g

        # Define the derivatives of `get_opt_par`
        defvjp(get_opt_par, _get_parhat_vjp)
        defjvp(get_opt_par, _get_parhat_jvp)

        return get_opt_par

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
    differentiate:
        Get a list of derivatives terms resulting from differentiating this
        term.
    check_similarity:
        Return a boolean indicating whether this term is equivalent to another
        term in the order of its derivative.
    combine_with:
        Return the sum of this term and another term.
    """
    def __init__(self, eps_order, eta_orders, prefactor):
        """
        Parameters
        -------------
        eps_order:
            The total number of epsilon partial derivatives of g.
        eta_orders:
            A vector of length order - 1.  Entry i contains the number
            of terms :math:`d\\eta^{i + 1} / d\\epsilon^{i + 1}`.
        prefactor:
            The constant multiple in front of this term.
        """
        # Base properties.
        self.eps_order = eps_order
        self.eta_orders = eta_orders
        self.prefactor = prefactor

        # Derived quantities.

        # The total number of eta partials.
        self.total_eta_order = np.sum(self.eta_orders)

        # The order is the total number of epsilon derivatives.
        self._order = int(
            self.eps_order + \
            np.sum(self.eta_orders * np.arange(1, len(self.eta_orders) + 1)))

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

    def __str__(self):
        return 'Order: {}\t{} * eta{} * eps[{}]'.format(
            self._order, self.prefactor, self.eta_orders, self.eps_order)

    def order(self):
        return self._order

    def differentiate(self):
        derivative_terms = []

        old_eta_orders = deepcopy(self.eta_orders)
        old_eta_orders.append(0)

        # dG / deps.
        derivative_terms.append(
            DerivativeTerm(
                eps_order=self.eps_order + 1,
                eta_orders=deepcopy(old_eta_orders),
                prefactor=self.prefactor))

        # dG / deta.
        new_eta_orders = deepcopy(old_eta_orders)
        new_eta_orders[0] = new_eta_orders[0] + 1
        derivative_terms.append(
            DerivativeTerm(
                eps_order=self.eps_order,
                eta_orders=new_eta_orders,
                prefactor=self.prefactor))

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
                        prefactor=self.prefactor * eta_order))

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
            prefactor=self.prefactor + term.prefactor)


def _evaluate_term_fwd(term, eta0, eps0, deps, eta_derivs,
                       eval_directional_derivative, validate=False):
    """Evaluate the DerivativeTerm in forward mode.

    Parameters
    ----------------------
    term: `DerivativeTerm`
        A `DerivativeTerm` object to evaluate.
    eta0, eps0 : `numpy.ndarray`
        Where to evaluate the derivative.
    deps : `numpy.ndarray`
        The direction in which to evaluate the derivative.
    eta_derivs : `list` of `numpy.ndarray`
        A list where ``eta_derivs[i]`` contains
        :math:`d\\eta^i / d\\epsilon^i \\Delta \\epsilon^i`.
    eval_directional_derivative:
        A function matching `eval_directional_derivative` in
        ForwardModeDerivativeArray.
    validate: optional `bool`
        If `True`, run checks for the appropriate sizes of arguments and
        produce helpful error messages if necessary.
        Default is `False`.
    """

    if validate:
        if len(eta_derivs) < term.order() - 1:
            raise ValueError('Not enough derivatives in ``eta_derivs``.')

    # First eta arguments, then epsilons.
    vec_args = []

    eta_directions = []
    for i in range(len(term.eta_orders)):
        eta_order = term.eta_orders[i]
        if eta_order > 0:
            for j in range(eta_order):
                eta_directions.append(eta_derivs[i])

    eps_directions = []
    for i in range(term.eps_order):
        eps_directions.append(deps)

    return term.prefactor * eval_directional_derivative(
        eta0, eps0, eta_directions, eps_directions, validate=validate)


def _contract_tensor(deriv_array, dx1s, dx2s):
    """Apply deriv_array to the list of vectors in dx1s and dx2s, keeping the
    first dimension of deriv_array.

    This is best understood with an example.  Consider:

    >>> deriv_array = np.random.random((3, 4, 4, 2))
    >>> dx1s = [ np.random.random(4) for _ in range(2) ]
    >>> dx2s = [ np.random.random(2) for _ in range(1) ]

    We want
    >>> einsum('zabc,a,b,c->z', deriv_array, dx1s[0], dx1s[1], dx2s[0])

    We construct this automatically for any length vectors.
    """
    if len(dx1s) + len(dx2s) >= len(ascii_lowercase):
        raise ValueError(
            'You cannot use _contract_tensor with so many vectors ' +
            'because of the crazy way rgiordan decided to do this with einsum.')

    if (len(dx1s) + len(dx2s)) == 0:
        return deriv_array

    ind_letters =  ascii_lowercase[0:(len(dx1s) + len(dx2s))]
    einsum_str = \
        'z' + ''.join(ind_letters) + ',' + ','.join(ind_letters) + '->z'
    return np.einsum(einsum_str, deriv_array, *(dx1s + dx2s))


class ForwardModeDerivativeArray():
    def __init__(self, fun, order1, order2):
        self._order1 = order1
        self._order2 = order2

        self._eval_fun_derivs = [[ fun ]]
        for x1_ind in range(self._order1 + 1):
            if x1_ind > 0: # The first row should be 0-th order x1 partials.
                # Append one x1 derivative.
                next_deriv = _append_jvp(
                    self._eval_fun_derivs[x1_ind - 1][0],
                    num_base_args=2, argnum=0)
                self._eval_fun_derivs.append([ next_deriv ])
            for x2_ind in range(self._order2):
                # Append one x2 derivative.  Note that the array already contains
                # a 0-th order x2 partial, and we are appending order2 new
                # derivatives, for a max order of order2 in order2 + 1 columns.
                next_deriv = _append_jvp(
                    self._eval_fun_derivs[x1_ind][x2_ind],
                    num_base_args=2, argnum=1)
                self._eval_fun_derivs[x1_ind].append(next_deriv)

    def eval_directional_derivative(self, x1, x2, dx1s, dx2s, validate=True):
        """Evaluate the directional derivative in the directions
        dx1s[0] ..., dx1s[N_1], dx2s[0], ..., dx2s[N_2].
        """

        order1 = len(dx1s)
        order2 = len(dx2s)
        if validate:
            if order1 > self._order1:
                raise ValueError(
                    'The number of `dx1s` () must be <= order1 = {}'.format(
                        order1, self._order1))
            if order2 > self._order2:
                raise ValueError(
                    'The number of `dx2s` () must be <= order2 = {}'.format(
                        order1, self._order1))
            if (not isinstance(dx1s, list)) or (not isinstance(dx2s, list)):
                raise ValueError('`dx1s` and `dx2s` must be lists of vectors.')

        return self._eval_fun_derivs[order1][order2](x1, x2, *[*dx1s, *dx2s])


class ReverseModeDerivativeArray():
    def __init__(self, fun, order1, order2):
        self._order1 = order1
        self._order2 = order2

        # Build an array of higher order reverse mode partial derivatives.
        # Traverse the rows (x1 partials) first, then traverse the
        # columns (x2 partials).
        self._eval_deriv_arrays = [[ fun ]]
        for x1_ind in range(self._order1 + 1):
            if x1_ind > 0:
                # Append one x1 derivative.
                next_deriv = autograd.jacobian(
                    self._eval_deriv_arrays[x1_ind - 1][0], argnum=0)
                self._eval_deriv_arrays.append([ next_deriv ])
            for x2_ind in range(self._order2):
                # Append one x2 derivative.  Note that the array already has
                # a 0-th order x2 partial, and we are appending order2 new
                # derivatives, for a max order of order2 in order2 + 1 columns.
                next_deriv = autograd.jacobian(
                    self._eval_deriv_arrays[x1_ind][x2_ind], argnum=1)
                self._eval_deriv_arrays[x1_ind].append(next_deriv)

    def set_evaluation_location(self, x1, x2, force=False, verbose=False):
        """Set the location at which we evaluate the partial derivatives.
        """

        x1 = np.atleast_1d(x1)
        x2 = np.atleast_1d(x2)
        if x1.ndim != 1 or x2.ndim != 1:
            raise ValueError('x1 and x2 must be 1d arrays.')

        self._deriv_arrays = \
            [[ np.atleast_1d(self._eval_deriv_arrays[0][0](x1, x2)) ]]
        dim0 = len(self._deriv_arrays[0][0])
        dim1 = len(x1)
        dim2 = len(x2)
        total_size = 0
        for i1 in range(self._order1 + 1):
            for i2 in range(self._order2 + 1):
                total_size += dim0 * (dim1 ** i1) * (dim2 ** i2)

        max_allowed_size = 100000
        if total_size > max_allowed_size and not force:
            err_msg = ('With len(x1) = {}, len(x2) = {}, ' +
                       'order1 = {}, and order2 = {}, this will create a ' +
                       'partial derivative array of size {} > {}.  ' +
                       'To force the creation of these arrays, set' +
                       '`force=True`.').format(
                dim1, dim2, self._order1, self._order2,
                total_size, max_allowed_size)
            raise ValueError(err_msg)

        if self._order1 > 2 and self._order2 > 2 and not force:
            err_msg = ('With both orders greater than two, reverse ' +
                       'mode can be slow even in low-dimensional problems.  ' +
                       'To force the creation of the arrays, set force=True.')
            raise ValueError(err_msg)

        # Save the location at which the partial derivatives are evaluated.
        self._x1 = deepcopy(x1)
        self._x2 = deepcopy(x2)
        if self._deriv_arrays[0][0].ndim != 1:
            raise ValueError(
                'The base function is expected to evaluate to a 1d vector.')

        self._deriv_arrays = \
            [[ None for _ in range(self._order2 + 1)] \
                for _ in range(self._order1 + 1)]
        for x1_ind in range(self._order1 + 1):
            for x2_ind in range(self._order2 + 1):
                if verbose:
                    print('Evaluating the derivative {}, {}'.format(
                        x1_ind, x2_ind))
                self._deriv_arrays[x1_ind][x2_ind] = \
                    self._eval_deriv_arrays[x1_ind][x2_ind](x1, x2)

    def deriv_arrays(self, order1, order2):
        return self._deriv_arrays[order1][order2]

    def _check_location(self, x1, x2, tol=1e-8):
        if (np.max(np.abs(x1 - self._x1)) > tol) or \
            (np.max(np.abs(x2 - self._x2)) > tol):
            raise ValueError(
                'You must use the x1 and x2 set in `set_evaluation_location`.')

    def eval_directional_derivative(self, x1, x2, dx1s, dx2s, validate=True):
        """Evaluate the directional derivative in the directions
        dx1s[0] ..., dx1s[N_1], dx2s[0], ..., dx2s[N_2].
        """

        order1 = len(dx1s)
        order2 = len(dx2s)
        if validate:
            self._check_location(x1, x2)
            if order1 > self._order1:
                raise ValueError(
                    'The number of `dx1s` () must be <= order1 = {}'.format(
                        order1, self._order1))
            if order2 > self._order2:
                raise ValueError(
                    'The number of `dx2s` () must be <= order2 = {}'.format(
                        order1, self._order1))

        # This keeps the first dimension of the partial derivative array,
        # and sums over the x1 first, then the x2.  This needs to match the
        # order in which the partial derivatives are taken in `__init__`.
        return _contract_tensor(
            self._deriv_arrays[order1][order2], dx1s, dx2s)


class ReorderedReverseModeDerivativeArray():
    """
    Like a ReverseModeDerivativeArray, but takes derivatives with respect
    to the larger dimension last for efficiency.

    ReverseModeDerivativeArray takes derivatives with respect to the
    second variable last.  The second variable should have the largest
    dimension for efficient automatic differentiation.  This class wraps
    the methods of ReverseModeDerivativeArray, swapping the order of the
    arguments if necessary to evaluate derivatives in the more efficient
    direction.
    """
    def __init__(self, fun, order1, order2, swapped=None):
        self._swapped = swapped

        # The underlying ReverseModeDerivativeArray will be with respect to
        # z1, z2 for the purposes of naming variables.
        if self._swapped:
            self._orderz1 = order2
            self._orderz2 = order1
        else:
            self._orderz1 = order1
            self._orderz2 = order2

        def _fun(z1, z2):
            return fun(*(self._swapped_args(z1, z2)))
        self._fun = _fun

        self._rmda = ReverseModeDerivativeArray(
            fun=self._fun, order1=self._orderz1, order2=self._orderz2)

    def _swapped_args(self, x1, x2):
        if self._swapped:
            return x2, x1
        else:
            return x1, x2

    def set_evaluation_location(self, x1, x2, force=False, verbose=False):
        z1, z2 = self._swapped_args(x1, x2)
        return self._rmda.set_evaluation_location(
            x1=z1, x2=z2, force=force, verbose=verbose)

    def eval_directional_derivative(self, x1, x2, dx1s, dx2s, validate=True):
        z1, z2 = self._swapped_args(x1, x2)
        dz1s, dz2s = self._swapped_args(dx1s, dx2s)
        return self._rmda.eval_directional_derivative(
            z1, z2, dz1s, dz2s, validate=validate)

    def deriv_arrays(self, order1, order2):
        if self._swapped:
            orig_array = self._rmda.deriv_arrays(order2, order1)
            # Derivatives with respect to the second variable come first
            # (after the axis of the base function), and need to be at the end.
            axes2 = np.arange(order2) + 1
            return np.moveaxis(orig_array, axes2, -1 * axes2)
        else:
            return self._rmda.deriv_arrays(order1, order2)


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
def _get_taylor_base_terms():
    dterms1 = [ \
        DerivativeTerm(
            eps_order=1,
            eta_orders=[0],
            prefactor=1.0),
        DerivativeTerm(
            eps_order=0,
            eta_orders=[1],
            prefactor=1.0) ]
    return dterms1


class ParametricSensitivityTaylorExpansion(object):
    """
    Evaluate the Taylor series of an optimum on a hyperparameter.

    This is a class for computing the Taylor series of
    eta(eps) = argmax_eta objective(eta, eps) using forward-mode automatic
    differentation.

    .. note:: This class is experimental and should be used with caution.
    """
    @classmethod
    def optimization_objective(
        cls,
        objective_function,
        input_val0,
        hyper_val0,
        order,
        hess0=None,
        forward_mode=True,
        max_input_order=None,
        max_hyper_order=None,
        force=False):
        """
        Parameters
        ------------------
        objective_function : `callable`
            The optimization objective as a function of two arguments
            (eta, eps), where eta is the parameter that is optimized and
            eps is a hyperparameter.
        hess0 : `numpy.ndarray` (N, N)
            Optional.  The Hessian of the objective at
            (``input_val0``, ``hyper_val0``).
            If not specified it is calculated at initialization.
        The remaining arguments are the same as for the `__init__` method.
        """
        # In order to calculate derivatives d^kinput_dhyper^k, we will be
        # Taylor expanding the gradient of the objective with respect to eta.
        estimating_equation = autograd.grad(objective_function, argnum=0)

        if hess0 is None:
            objective_function_hessian = \
                autograd.hessian(objective_function, argnum=0)
            hess0 = objective_function_hessian(input_val0, hyper_val0)

        # TODO: if the objective function returns a 1-d array and not a
        # float then the Cholesky decomposition will fail because
        # the Hessian will have an extra dimension.  This is a confusing
        # error that we could catch explicitly at the cost of an extra
        # function evaluation.  Is it worth it?
        hess_solver = solver_lib.get_cholesky_solver(hess0)

        return cls(
            estimating_equation=estimating_equation,
            input_val0=input_val0,
            hyper_val0=hyper_val0,
            order=order,
            hess_solver=hess_solver,
            forward_mode=forward_mode,
            max_input_order=max_input_order,
            max_hyper_order=max_hyper_order,
            force=force)

    def __init__(self,
                 estimating_equation,
                 input_val0, hyper_val0,
                 order,
                 hess_solver,
                 forward_mode=True,
                 max_input_order=None,
                 max_hyper_order=None,
                 force=False):
        """
        Parameters
        ------------------
        estimating_equation : `callable`
            A vector-valued function function of two arguments,
            (`input`, `output`), where the length of the vector is the same
            as the length of `input`, and which is (approximately) the zero
            vector when evaluated at (`input_val0`, `hyper_val0`).
        input_val0 : `numpy.ndarray` (N,)
            The value of ``input_par`` at the optimum.
        hyper_val0 : `numpy.ndarray` (M,)
            The value of ``hyper_par`` at which ``input_val0`` was found.
        order : `int`
            The maximum order of the Taylor series to be calculated.
        hess_solver : `function`
            A function that takes a single argument, `v`, and returns

            .. math::

                \\frac{\\partial G}{\\partial \\eta}^{-1} v,

            where :math:`G(\\eta, \\epsilon)` is the estimating equation,
            and the partial derivative is evaluated at
            :math:`(\\eta, \\epsilon) =`  (`input_val0`, `hyper_val0`).
        forward_mode : `bool`
            Optional.  If `True` (the default), use forward-mode automatic
            differentiation.  Otherwise, use reverse-mode.
        max_input_order : `int`
            Optional.  The maximum number of nonzero partial derivatives of
            the objective function gradient with respect to the input parameter.
            If `None`, calculate partial derivatives of all orders.
        max_hyper_order : `int`
            Optional.  The maximum number of nonzero partial derivatives of
            the objective function gradient with respect to the hyperparameter.
            If `None`, calculate partial derivatives of all orders.
        force: `bool`
            Optional.  If `True`, force the instantiation of potentially
            expensive reverse mode derivative arrays.  Default is `False`.
        """
        self._input_val0 = deepcopy(input_val0)
        self._hyper_val0 = deepcopy(hyper_val0)
        self._objective_function_eta_grad = estimating_equation
        self._set_order(order, max_input_order, max_hyper_order, forward_mode)
        self.hess_solver = hess_solver

        if not self._forward_mode:
            # Set the derivative arrays.
            # TODO: don't duplicate the Hessian calculation?
            self._deriv_array.set_evaluation_location(
                self._input_val0, self._hyper_val0, force=force)

    def _set_order(self, order, max_input_order, max_hyper_order, forward_mode):
        """Generate the matrix of g partial derivatives and differentiate
        the Taylor series up to the required order.
        """
        self._max_input_order = max_input_order
        self._max_hyper_order = max_hyper_order
        self._forward_mode = forward_mode
        if not self._forward_mode:
            warnings.warn(
                'Reverse mode Taylor expansions are experimental.')

        if self._max_input_order is not None or \
           self._max_hyper_order is not None:
            warnings.warn(
                'Setting _max_hyper_order or _max_input_order is experimental.')

        if self._max_input_order is not None and self._max_input_order < 1:
            raise ValueError('max_input_order must be >= 1.')
        if self._max_hyper_order is not None and self._max_hyper_order < 1:
            raise ValueError('max_hyper_order must be >= 1.')

        self._order = order

        # In fact can't this be self._order - 1?
        if self._max_input_order is None:
            order1 = self._order
        else:
            order1 = min(self._order, self._max_input_order)

        if self._max_hyper_order is None:
            order2 = self._order
        else:
            order2 = min(self._order, self._max_hyper_order)

        if self._forward_mode:
            self._deriv_array = ForwardModeDerivativeArray(
                self._objective_function_eta_grad,
                order1=order1,
                order2=order2)
        else:
            # If the input has higher dimension than the hyperparameter,
            # we want to take reverse mode derivatives with respect to
            # the hyperparameter first, so swap the order of the arguments.
            swapped = len(self._input_val0) > len(self._hyper_val0)
            self._deriv_array = ReorderedReverseModeDerivativeArray(
                self._objective_function_eta_grad,
                order1=order1,
                order2=order2,
                swapped=swapped)

        def differentiate_terms(dterms):
            dterms_derivs = []
            for term in dterms:
                dterms_derivs += term.differentiate()
            return _consolidate_terms(dterms_derivs)

        self._taylor_terms_list = [ _get_taylor_base_terms() ]
        for k in range(1, self._order):
            next_taylor_terms = \
                differentiate_terms(self._taylor_terms_list[k - 1])
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
            The value of the k^th derivative in the direction dhyper.
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
            if (term.eta_orders[-1] > 0):
                continue

            # Assume that epsilon partials higher than this are zero.
            if self._max_hyper_order is not None and \
               term.eps_order > self._max_hyper_order:
                continue

            # Assume that eta partials higher than this are zero.
            if self._max_input_order is not None and \
               term.total_eta_order > self._max_input_order:
                continue

            vec += \
                _evaluate_term_fwd(
                    term=term,
                    eta0=self._input_val0,
                    eps0=self._hyper_val0,
                    deps=dhyper,
                    eta_derivs=input_derivs,
                    eval_directional_derivative= \
                        self._deriv_array.eval_directional_derivative)
        return -1 * self.hess_solver(vec)


    def _get_default_max_order(self, max_order):
        if max_order is None:
            return self._order
        if max_order <= 0:
            raise ValueError('max_order must be greater than zero.')
        if max_order > self._order:
            raise ValueError(
                'max_order must be no greater than the order={}'.format(
                    self._order))
        return max_order

    def evaluate_input_derivs(self, dhyper, max_order=None):
        """Return a list of the derivatives dkinput / dhyperk dhyper^k
        """
        max_order = self._get_default_max_order(max_order)
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
        max_order = self._get_default_max_order(max_order)
        if add_offset:
            dinput_terms = [self._input_val0]
        else:
            dinput_terms = [np.zeros_like(self._input_val0)]
        dhyper = new_hyper_val - self._hyper_val0
        input_derivs = self.evaluate_input_derivs(dhyper, max_order=max_order)

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
