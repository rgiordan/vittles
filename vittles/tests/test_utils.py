from autograd import numpy as np
from contextlib import contextmanager # For testing stdout
import paragami
from io import StringIO
import sys # For testing stdout

import paragami

# For testing stdout
# https://gist.github.com/mogproject/fc7c4e94ba505e95fa03
@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


# This class will be used for testing.
class QuadraticModel(object):
    def __init__(self, dim):
        # Put lower bounds so we're testing the contraining functions
        # and so that derivatives of all orders are nonzero.
        self.dim = dim
        self.theta_pattern = \
            paragami.NumericArrayPattern(shape=(dim, ), lb=-10.)
        self.lambda_pattern = \
            paragami.NumericArrayPattern(shape=(dim, ), lb=-10.0)

        vec = np.linspace(0.1, 0.3, num=dim)
        self.matrix = np.outer(vec, vec) + np.eye(dim)

    def get_flat_objective(self, theta_free, lambda_free):
        return paragami.FlattenFunctionInput(
            self.get_objective,
            free=[theta_free, lambda_free],
            argnums=[0, 1],
            patterns=[self.theta_pattern, self.lambda_pattern])

    def get_flat_true_optimal_theta(self, theta_free, lambda_free):
        return paragami.FlattenFunctionInputAndOutput(
            self.get_true_optimal_theta,
            input_free=lambda_free,
            output_free=theta_free,
            input_patterns=self.lambda_pattern,
            output_patterns=self.theta_pattern,
            input_argnums=[0],
            output_retnums=[0])

    def get_default_lambda(self):
        return np.linspace(0.5, 10.0, num=self.dim)

    def get_hyper_par_objective(self, theta, lam):
        # Only the part of the objective that dependson the hyperparameters.
        return lam @ theta

    def get_objective(self, theta, lam):
        objective = 0.5 * theta.T @ self.matrix @ theta
        shift = self.get_hyper_par_objective(theta, lam)
        return objective + shift

    # Testing functions that use the fact that the optimum has a closed form.
    def get_true_optimal_theta(self, lam):
        return -1 * np.linalg.solve(self.matrix, lam)

    def get_default_flat_values(self, theta_free, lambda_free):
        lam0 = self.get_default_lambda()
        theta0 = self.get_true_optimal_theta(lam0)

        return \
            self.theta_pattern.flatten(theta0, theta_free), \
            self.lambda_pattern.flatten(lam0, lambda_free)
