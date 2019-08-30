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

    def evaluate(self, dh1, dh2, debug=False):
        if self._term_ii or self._term_i2 or self._term_i12:
            g_1 = self._g_1(
                self._input_base, self._hyper1_base, self._hyper2_base,
                dh1)
            di1 = -1 * self._solver(g_1)

        if self._term_ii or self._term_i1 or self._term_i12:
            g_2 = self._g_2(
                self._input_base, self._hyper1_base, self._hyper2_base,
                dh2)
            di2 = -1 * self._solver(g_2)

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
