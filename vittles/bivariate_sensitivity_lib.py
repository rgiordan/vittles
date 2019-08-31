import autograd
from autograd import numpy as np
from .sensitivity_lib import _append_jvp
from copy import deepcopy
import warnings


class CrossSensitivity():
    """Calculate a second-order derivative of an optimum with resepct to
    two hyperparameters.

    Given an estimating equation :math:`G(\\theta, \\epsilon_1, \\epsilon_2)`,
    with :math:`G(\\hat\\theta(\\epsilon_1, \\epsilon_2),
    \\epsilon_1, \\epsilon_2) = 0`, this class evaluates a directional
    derivatives

    .. math::
        \\frac{d^2\hat{\\theta}}{d\\epsilon_1 d\\epsilon_2}
        \\Delta \\epsilon_1 \\Delta \\epsilon_2.
    """
    def __init__(self,
                 estimating_equation,
                 solver,
                 input_base,
                 hyper1_base,
                 hyper2_base,
                 term_ii=True,
                 term_i1=True,
                 term_i2=True,
                 term_12=True):

        warnings.warn(
            'The CrossSensitivity class is very experimental and untested.')

        self._g = estimating_equation
        self._solver = solver

        # Copy these because the solver is only valid at these values.
        self._input_base = deepcopy(input_base)
        self._hyper1_base = deepcopy(hyper1_base)
        self._hyper2_base = deepcopy(hyper2_base)

        # TODO: for readability, wrap these so they are always
        # evaluated at the same first three arguments.
        self._g_i = _append_jvp(self._g, num_base_args=3, argnum=0)
        self._g_ii = _append_jvp(self._g_i, num_base_args=3, argnum=0)
        self._g_i1 = _append_jvp(self._g_i, num_base_args=3, argnum=1)
        self._g_i2 = _append_jvp(self._g_i, num_base_args=3, argnum=2)
        self._g_1 = _append_jvp(self._g, num_base_args=3, argnum=1)
        self._g_2 = _append_jvp(self._g, num_base_args=3, argnum=2)
        self._g_12 = _append_jvp(self._g_1, num_base_args=3, argnum=2)

        self._term_ii = term_ii
        self._term_i1 = term_i1
        self._term_i2 = term_i2
        self._term_12 = term_12

    def get_di1(self, dh1):
        g_1 = self._g_1(
            self._input_base, self._hyper1_base, self._hyper2_base,
            dh1)
        di1 = -1 * self._solver(g_1)
        return di1

    def get_di2(self, dh2):
        g_2 = self._g_2(
            self._input_base, self._hyper1_base, self._hyper2_base,
            dh2)
        di2 = -1 * self._solver(g_2)
        return di2

    def evaluate(self, dh1, dh2, di1=None, di2=None, debug=False):
        if self._term_ii or self._term_i2 or self._term_i12:
            if di1 is None:
                di1 = self.get_di1(dh1)

        if self._term_ii or self._term_i1 or self._term_i12:
            if di2 is None:
                di2 = self.get_di2(dh2)

        g_ii = 0
        g_i1 = 0
        g_i2 = 0
        g_12 = 0
        if self._term_ii:
            g_ii = self._g_ii(
                self._input_base, self._hyper1_base, self._hyper2_base,
                di1, di2)

        if self._term_i1:
            g_i1 = self._g_i1(
                self._input_base, self._hyper1_base, self._hyper2_base,
                di2, dh1)

        if self._term_i2:
            g_i2 = self._g_i2(
                self._input_base, self._hyper1_base, self._hyper2_base,
                di1, dh2)

        if self._term_12:
            g_12 = self._g_12(
                self._input_base, self._hyper1_base, self._hyper2_base,
                dh1, dh2)

        if debug:
            print('g_ii: ', g_ii)
            print('g_i1: ', g_i1)
            print('g_i2: ', g_i2)
            print('g_12: ', g_12)
            print('di1: ', di1)
            print('di2: ', di2)

        return -1 * self._solver(g_ii + g_i1 + g_i2 + g_12)


class OptimumChecker():
    def __init__(self,
                 estimating_equation,
                 solver,
                 input_base,
                 hyper_base):
        """Estimate the error in sensitivity due to incomplete optimization.

        Parameters
        ---------------
        estimating_equation : `callable`
            A function taking arguments `(input, hyper)` and returning
            a vector, typically the same length as the input.  The idea is
            that `estimating_equation(input_base, hyper_base) = [0, ..., 0]`.
        solver : `callable`
            A function of a single vector variable `v` solving :math:`H^{-1} v`,
            where `H` is the Hessian of the estimating equation with respect
            to the input variable at `input_base`, `hyper_base`.
        input_base : `numpy.ndarray`
            The base value of the parameter to be optimized
        hyper_base : `numpy.ndarray`
            The base value of the hyperparameter.
        """

        self._input_base = deepcopy(input_base)
        self._hyper_base = deepcopy(hyper_base)
        self._solver = solver

        def estimating_equation_lagrange(ipar, hpar, lam):
            return estimating_equation(ipar, hpar) + lam
        self.estimating_equation_lagrange = estimating_equation_lagrange

        # self._obj = lambda ipar: estimating_equation(ipar, self._hyper_base)
        # self._obj_grad = autograd.jacobian(self._obj)

        #self._lam_base = -1 * self._obj_grad(self._input_base)
        self._lam_base = \
            -1 * estimating_equation(self._input_base, self._hyper_base)
        self._dlam = -1 * self._lam_base

        self._cross_sens = CrossSensitivity(
            estimating_equation = self.estimating_equation_lagrange,
            solver =        self._solver,
            input_base =    self._input_base,
            hyper1_base =   self._hyper_base,
            hyper2_base =   self._lam_base,
            term_i2=False,
            term_12=False)

    def get_newton_step(self):
        """Return a Netwon step towards the optimum.
        """
        return self._cross_sens.get_di2(self._dlam)

    def get_dinput_dhyper(self, dhyper):
        """Return the first directional derivative of the optimum with respect
        to the hyperparameter in the direction `dhyper`.
        """
        return self._cross_sens.get_di1(dhyper)

    def correction(self, hyper_new, dinput_dhyper=None, newton_step=None):
        """Return the first-order correction to the change in
        dinput_dhyper as you take a Newton step.
        """
        dhyper = hyper_new - self._hyper_base
        if dinput_dhyper is None:
            dinput_dhyper = self.get_dinput_dhyper(dhyper)
        if newton_step is None:
            newton_step = self.get_newton_step()
        dinput_dhyper_correction = \
            self._cross_sens.evaluate(
                dhyper, self._dlam,
                di1=dinput_dhyper,
                di2=newton_step)
        return dinput_dhyper_correction
        return self._input_base + dinput_dhyper + dinput_dhyper_correction

    def evaluate(self, hyper_new, dinput_dhyper=None, newton_step=None):
        """Return the first-order approximation to the change in
        dinput_dhyper as you take a Newton step.
        """
        dhyper = hyper_new - self._hyper_base
        if dinput_dhyper is None:
            dinput_dhyper = self.get_dinput_dhyper(dhyper)
        dinput_dhyper_correction = self.correction(
            hyper_new, dinput_dhyper=dinput_dhyper, newton_step=newton_step)
        return self._input_base + dinput_dhyper + dinput_dhyper_correction
