from .base_patterns import Pattern
from .pattern_containers import register_pattern_json
import autograd.numpy as np
import copy
import json


def _unconstrain_array(array, lb, ub):
    if not (array <= ub).all():
        raise ValueError('Elements larger than the upper bound')
    if not (array >= lb).all():
        raise ValueError('Elements smaller than the lower bound')
    if ub <= lb:
        raise ValueError('Upper bound must be greater than lower bound')
    if ub == float("inf"):
        if lb == -float("inf"):
            # For consistent behavior, never return a reference.
            #return copy.deepcopy(array)
            return copy.copy(array)
        else:
            return np.log(array - lb)
    else:  # the upper bound is finite
        if lb == -float("inf"):
            return -1 * np.log(ub - array)
        else:
            return np.log(array - lb) - np.log(ub - array)


def _constrain_array(free_array, lb, ub):
    if ub <= lb:
        raise ValueError('Upper bound must be greater than lower bound')
    if ub == float("inf"):
        if lb == -float("inf"):
            # For consistency, never return a reference.
            #return copy.deepcopy(free_array)
            #return free_array
            return copy.copy(free_array)
        else:
            return np.exp(free_array) + lb
    else:  # the upper bound is finite
        if lb == -float("inf"):
            return ub - np.exp(-1 * free_array)
        else:
            exp_vec = np.exp(free_array)
            return (ub - lb) * exp_vec / (1 + exp_vec) + lb


def _get_inbounds_value(lb, ub):
    assert lb < ub
    if lb > -float('inf') and ub < float('inf'):
        return 0.5 * (ub - lb) + lb
    else:
        if lb > -float('inf'):
            # The upper bound is infinite.
            return lb + 1.0
        elif ub < float('inf'):
            # The lower bound is infinite.
            return ub - 1.0
        else:
            # Both are infinite.
            return 0.0


class NumericArrayPattern(Pattern):
    """
    A pattern for (optionally bounded) arrays of numbers.

    Attributes
    -------------
    default_validate: Bool
        Whether or not the array is checked by default to lie within the
        specified bounds.

    Methods
    ----------------
    validate_folded: Check whether the folded array lies within the bounds.
    """
    def __init__(self, shape,
                 lb=-float("inf"), ub=float("inf"), default_validate=True):

        """
        Parameters
        -------------
        shape: Tuple of int
            The shape of the array.
        lb: float
            The (inclusive) lower bound for the entries of the array.
        ub: float
            The (inclusive) upper bound for the entries of the array.
        default_validate: bool
            Whether or not the array is checked by default to lie within the
            specified bounds.
        """
        self.default_validate = default_validate
        self.__shape = tuple(shape)
        self.__lb = lb
        self.__ub = ub
        assert lb >= -float('inf')
        assert ub <= float('inf')
        if lb >= ub:
            raise ValueError(
                'Upper bound ub must strictly exceed lower bound lb')

        free_flat_length = flat_length = int(np.product(self.__shape))

        super().__init__(flat_length, free_flat_length)

    def __str__(self):
        return 'NumericArrayPattern {} (lb={}, ub={})'.format(
            self.__shape, self.__lb, self.__ub)

    def as_dict(self):
        return {
            'pattern': self.json_typename(),
            'lb': self.__lb,
            'ub': self.__ub,
            'shape': self.__shape,
            'default_validate': self.default_validate}

    def empty(self, valid):
        if valid:
            return np.full(
                self.__shape, _get_inbounds_value(self.__lb, self.__ub))
        else:
            return np.empty(self.__shape)

    def check_folded(self, folded_val, validate=None):
        if folded_val.shape != self.shape():
            raise ValueError('Wrong size for Array.' +
                             ' Expected shape: ' + str(self.shape()) +
                             ' Got shape: ' + str(folded_val.shape))
        if validate is None:
            validate = self.default_validate
        if validate:
            if (np.array(folded_val < self.__lb)).any():
                raise ValueError('Value beneath lower bound.')
            if (np.array(folded_val > self.__ub)).any():
                raise ValueError('Value above upper bound.')

    def _free_fold(self, free_flat_val):
        if free_flat_val.size != self._free_flat_length:
            error_string = \
                'Wrong size for Array.  Expected {}, got {}'.format(
                    str(self._free_flat_length),
                    str(free_flat_val.size))
            raise ValueError(error_string)
        constrained_array = \
            _constrain_array(free_flat_val, self.__lb, self.__ub)
        return constrained_array.reshape(self.__shape)

    def _free_flatten(self, folded_val, validate=None):
        self.check_folded(folded_val, validate)
        return _unconstrain_array(folded_val, self.__lb, self.__ub).flatten()

    def _notfree_fold(self, flat_val, validate=None):
        if flat_val.size != self._flat_length:
            error_string = \
                'Wrong size for Array.  Expected {}, got {}'.format(
                    str(self._flat_length), str(flat_val.size))
            raise ValueError(error_string)
        folded_val = flat_val.reshape(self.__shape)
        self.check_folded(folded_val, validate)
        return folded_val

    def _notfree_flatten(self, folded_val, validate=None):
        self.check_folded(folded_val, validate)
        return folded_val.flatten()

    def fold(self, flat_val, free, validate=None):
        flat_val = np.atleast_1d(flat_val)
        if len(flat_val.shape) != 1:
            raise ValueError('The argument to fold must be a 1d vector.')
        if free:
            return self._free_fold(flat_val)
        else:
            return self._notfree_fold(flat_val, validate)

    def flatten(self, folded_val, free, validate=None):
        folded_val = np.atleast_1d(folded_val)
        if free:
            return self._free_flatten(folded_val, validate)
        else:
            return self._notfree_flatten(folded_val, validate)

    def shape(self):
        return self.__shape

    def bounds(self):
        return self.__lb, self.__ub

    def flat_length(self, free):
        if free:
            return self._free_flat_length
        else:
            return self._flat_length


register_pattern_json(NumericArrayPattern)
