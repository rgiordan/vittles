import autograd
import autograd.numpy as np
import copy
import scipy as osp

###############################
# Preconditioned objectives.  #
###############################


def _get_sym_matrix_inv_sqrt(mat, ev_min=None, ev_max=None):
    """
    Get the inverse square root of a symmetric matrix with thresholds for the
    eigenvalues.

    This is particularly useful for calculating preconditioners.
    """
    mat = np.atleast_2d(mat)

    # Symmetrize for numerical stability.
    mat_sym = 0.5 * (mat + mat.T)
    eig_val, eig_vec = np.linalg.eigh(mat_sym)

    if not ev_min is None:
        if not np.isreal(ev_min):
            raise ValueError('ev_min must be real-valued.')
        eig_val[np.real(eig_val) <= ev_min] = ev_min
    if not ev_max is None:
        if not np.isreal(ev_max):
            raise ValueError('ev_max must be real-valued.')
        eig_val[np.real(eig_val) >= ev_max] = ev_max

    mat_corrected = np.matmul(eig_vec,
                               np.matmul(np.diag(eig_val), eig_vec.T))
    mat_sqrt = \
        np.matmul(eig_vec,
                  np.matmul(np.diag(np.sqrt(eig_val)), eig_vec.T))

    mat_inv_sqrt = \
        np.matmul(eig_vec,
                  np.matmul(np.diag(1 / np.sqrt(eig_val)), eig_vec.T))

    return np.array(mat_inv_sqrt), \
           np.array(mat_sqrt), \
           np.array(mat_corrected)


class PreconditionedFunction():
    """
    Get a function whose input has been preconditioned.

    Throughout, the subscript ``_c`` will denote quantiites or
    funcitons in the preconditioned space.  For example, ``x`` will
    refer to a variable in the original space and ``x_c`` to the same
    variable after preconditioning.

    Preconditioning means transforming :math:`x \\rightarrow x_c = A^{-1} x`,
    where the matrix :math:`A` is the "preconditioner".  If :math:`f` operates
    on :math:`x`, then the preconditioned function operates on :math:`x_c` and
    is defined by :math:`f_c(x_c) := f(A x_c) = f(x)`. Gradients of the
    preconditioned function are defined with respect to its argument in the
    preconditioned space, e.g., :math:`f'_c = \\frac{df_c}{dx_c}`.

    A typical value of the preconditioner is an inverse square root of the
    Hessian of :math:`f`, because then the Hessian of :math:`f_c` is
    the identity when the gradient is zero.  This can help speed up the
    convergence of optimization algorithms.

    Methods
    ----------
    set_preconditioner:
        Set the preconditioner to a specified value.
    set_preconditioner_with_hessian:
        Set the preconditioner based on the Hessian of the objective
        at a point in the orginal domain.
    get_preconditioner:
        Return a copy of the current preconditioner.
    get_preconditioner_inv:
        Return a copy of the current inverse preconditioner.
    precondition:
        Convert from the original domain to the preconditioned domain.
    unprecondition:
        Convert from the preconditioned domain to the original domain.
    """
    def __init__(self, original_fun,
                 preconditioner=None,
                 preconditioner_inv=None):
        """
        Parameters
        -------------
        original_fun:
            callable function of a single argument
        preconditioner:
            The initial preconditioner.
        preconditioner_inv:
            The inverse of the initial preconditioner.
        """
        self._original_fun = original_fun
        self._original_fun_hessian = autograd.hessian(self._original_fun)
        if (preconditioner is None) and (preconditioner_inv is not None):
            raise ValueError(
                'If you specify preconditioner_inv, you must' +
                'also specify preconditioner. ')
        if preconditioner is not None:
            self.set_preconditioner(preconditioner, preconditioner_inv)
        else:
            self._preconditioner = None
            self._preconditioner_inv = None

    def get_preconditioner(self):
        return copy.copy(self._preconditioner)

    def get_preconditioner_inv(self):
        return copy.copy(self._preconditioner_inv)

    def set_preconditioner(self, preconditioner, preconditioner_inv=None):
        self._preconditioner = preconditioner
        if preconditioner_inv is None:
            self._preconditioner_inv = np.linalg.inv(self._preconditioner)
        else:
            self._preconditioner_inv = preconditioner_inv

    def set_preconditioner_with_hessian(self, x=None, hessian=None,
                                        ev_min=None, ev_max=None):
        """
        Set the precoditioner to the inverse square root of the Hessian of
        the original objective (or an approximation thereof).

        Parameters
        ---------------
        x: Numeric vector
            The point at which to evaluate the Hessian of ``original_fun``.
            If x is specified, the Hessian is evaluated with automatic
            differentiation.
            Specify either x or hessian but not both.
        hessian: Numeric matrix
            The hessian of ``original_fun`` or an approximation of it.
            Specify either x or hessian but not both.
        ev_min: float
            If not None, set eigenvaluse of ``hessian`` that are less than
            ``ev_min`` to ``ev_min`` before taking the square root.
        ev_maxs: float
            If not None, set eigenvaluse of ``hessian`` that are greater than
            ``ev_max`` to ``ev_max`` before taking the square root.

        Returns
        ------------
        Sets the precoditioner for the class and returns the Hessian with
        the eigenvalues thresholded by ``ev_min`` and ``ev_max``.
        """
        if x is not None and hessian is not None:
            raise ValueError('You must specify x or hessian but not both.')
        if x is None and hessian is None:
            raise ValueError('You must specify either x or hessian.')
        if hessian is None:
            # We now know x is not None.
            hessian = self._original_fun_hessian(x)

        hess_inv_sqrt, hess_sqrt, hess_corrected = \
            _get_sym_matrix_inv_sqrt(hessian, ev_min, ev_max)
        self.set_preconditioner(hess_inv_sqrt, hess_sqrt)

        return hess_corrected

    def precondition(self, x):
        """
        Multiply by the inverse of the preconditioner to convert
        :math:`x` in the original domain to :math:`x_c` in the preconditioned
        domain.

        This function is provided for convenience, but it is more numerically
        stable to use np.linalg.solve(preconditioner, x).
        """
        # On one hand, this is a numerically instable way to solve a linear
        # system.  On the other hand, the inverse is readily available from
        # the eigenvalue decomposition and the Cholesky factorization
        # is not AFAIK.
        if self._preconditioner_inv is None:
            raise ValueError('You must set the preconditioner.')
        return self._preconditioner_inv @ x

    def unprecondition(self, x_c):
        """
        Multiply by the preconditioner to convert
        :math:`x_c` in the preconditioned domain to :math:`x` in the
        original domain.
        """
        if self._preconditioner is None:
            raise ValueError('You must set the preconditioner.')
        return self._preconditioner @ x_c

    def __call__(self, x_c):
        """
        Evaluate the preconditioned function at a point in the preconditioned
        domain.
        """
        return self._original_fun(self.unprecondition(x_c))



class OptimizationObjective():
    """
    Derivatives and logging for an optimization objective function.

    Attributes
    -------------
    optimization_log: Dictionary
        A record of the optimization progress as recorded by ``log_value``.

    Methods
    ---------------
    f:
        The objective function with logging.
    grad:
        The gradient of the objective function.
    hessian:
        The Hessian of the objective function.
    hessian_vector_product:
        The Hessian vector product of the objective function.
    set_print_every:
        Set how often to display optimization progress.
    set_log_every:
        Set how often to log optimization progress.
    reset_iteration_count:
        Reset the number of iterations for the purpose of printing and logging.
    reset_log:
        Clear the log.
    reset:
        Run ``reset_iteration_count`` and ``reset_log``.
    print_value:
        Display a function evaluation.
    log_value:
        Log a function evaluation.
    """
    def __init__(self, objective_fun, print_every=1, log_every=0):
        """
        Parameters
        -------------
        obj_fun: Callable function of one argumnet
            The function to be minimized.
        print_every: integer
            Print the optimization value every ``print_every`` iterations.
        log_every: integer
            Log the optimization value every ``log_every`` iterations.
        """

        self._objective_fun = objective_fun
        self.grad = autograd.grad(self._objective_fun)
        self.hessian = autograd.hessian(self._objective_fun)
        self.hessian_vector_product = \
            autograd.hessian_vector_product(self._objective_fun)

        self.set_print_every(print_every)
        self.set_log_every(log_every)

        self.reset()

    def set_print_every(self, n):
        """
        Parameters
        -------------
        n: integer
            Print the objective function value every ``n`` iterations.
            If 0, do not print any output.
        """
        self._print_every = n

    def set_log_every(self, n):
        """
        Parameters
        -------------
        n: integer
            Log the objective function value every ``n`` iterations.
            If 0, do not log.
        """
        self._log_every = n

    def reset(self):
        """
        Reset the itreation count and clear the log.
        """
        self.reset_iteration_count()
        self.reset_log()

    def reset_iteration_count(self):
        self._num_f_evals = 0

    def num_iterations(self):
        """
        Return the number of times the optimization function has been called,
        not counting any derivative evaluations.
        """
        return self._num_f_evals

    def print_value(self, num_f_evals, x, f_val):
        """
        Display the optimization progress.  To display a custom
        update, overload this function.

        Parameters
        -------------
        num_f_vals: Integer
            The total number of function evaluations.
        x:
            The current argument to the objective function.
        f_val:
            The value of the objective at ``x``.
        """
        print('Iter {}: f = {:0.8f}'.format(num_f_evals, f_val))

    def reset_log(self):
        self.optimization_log = []

    def log_value(self, num_f_evals, x, f_val):
        """
        Log the optimization progress.  To create a custom log,
        overload this function.  By default, the log is a list of tuples
        ``(iteration, x, f(x))``.

        Parameters
        -------------
        num_f_vals: Integer
            The total number of function evaluations.
        x:
            The current argument to the objective function.
        f_val:
            The value of the objective at ``x``.
        """
        self.optimization_log.append((num_f_evals, x, f_val))

    def f(self, x):
        f_val = self._objective_fun(x)
        if self._print_every > 0 and self._num_f_evals % self._print_every == 0:
            self.print_value(self._num_f_evals, x, f_val)
        if self._log_every > 0 and self._num_f_evals % self._log_every == 0:
            self.log_value(self._num_f_evals, x, f_val)
        self._num_f_evals += 1
        return f_val
