
from .base_patterns import Pattern
from .pattern_containers import register_pattern_json

import autograd.numpy as np
import autograd.scipy as sp

import itertools

import json
import warnings


def logsumexp(mat, axis):
    """Calculate logsumexp along the axis ``axis`` without dropping the axis.
    """
    if not type(axis) is int:
        raise ValueError(
            'This version of logsumexp is only designed for exactly one axis.')
    mat_max = np.max(mat, axis=axis, keepdims=True)
    exp_mat_norm = np.exp(mat - mat_max)
    return np.log(np.sum(exp_mat_norm, axis=axis, keepdims=True)) + mat_max


def _constrain_simplex_matrix(free_mat):
    # The first column is the reference value.  Append a column of zeros
    # to each simplex representing this reference value.
    reference_col = np.expand_dims(np.full(free_mat.shape[0:-1], 0), axis=-1)
    free_mat_aug = np.concatenate([reference_col, free_mat], axis=-1)

    # Note that autograd needs to update their logsumexp to be in special
    # not misc before this can be changed.  Furthermore, logsumexp is
    # not even available in the pypi version of autograd.
    log_norm = logsumexp(free_mat_aug, axis=-1)
    return np.exp(free_mat_aug - log_norm)


def _unconstrain_simplex_matrix(simplex_mat):
    return np.log(simplex_mat[..., 1:]) - \
           np.expand_dims(np.log(simplex_mat[..., 0]), axis=-1)


class SimplexArrayPattern(Pattern):
    """
    A pattern for an array of simplex parameters.

    The last index represents entries of the simplex.  For example,
    if `array_shape=(2, 3)` and `simplex_size=4`, then the pattern is
    for a 2x3 array of 4d simplexes.  If such value of the simplex
    array is given by `val`, then `val.shape = (2, 3, 4)` and
    `val[i, j, :]` is the `i,j`th of the six simplicial vectors, i.e,
    `np.sum(val[i, j, :])` equals 1 for each `i` and `j`.

    Attributes
    -------------
    default_validate: Bool
        Whether or not the simplex is checked by default to be
        non-negative and to sum to one.

    Methods
    ---------
    array_shape: tuple of ints
        The shape of the array of simplexes, not including the simplex
        dimension.

    simplex_size: int
        The length of each simplex.

    shape: tuple of ints
        The shape of the entire array including the simplex dimension.
    """
    def __init__(self, simplex_size, array_shape, default_validate=True):
        """
        Parameters
        ------------
        simplex_size: int
            The length of the simplexes.
        array_shape: tuple of integers
            The size of the array of simplexes (not including the simplexes
            themselves).
        default_validate: bool
            Whether or not to check for legal (i.e., positive and normalized)
            folded values by default.
        """
        self.__simplex_size = int(simplex_size)
        if self.__simplex_size <= 1:
            raise ValueError('simplex_size must be >= 2.')
        self.__array_shape = array_shape
        self.__shape = self.__array_shape + (self.__simplex_size, )
        self.__free_shape = self.__array_shape + (self.__simplex_size - 1, )
        self.default_validate = default_validate
        super().__init__(np.prod(self.__shape), np.prod(self.__free_shape))

    def __str__(self):
        return 'SimplexArrayPattern {} of {}-d simplices'.format(
            self.__array_shape, self.__simplex_size)

    def array_shape(self):
        return self.__array_shape

    def simplex_size(self):
        return self.__simplex_size

    def shape(self):
        return self.__shape

    def as_dict(self):
        return {
            'pattern': self.json_typename(),
            'simplex_size': self.__simplex_size,
            'array_shape': self.__array_shape,
            'default_validate': self.default_validate}

    def empty(self, valid):
        if valid:
            return np.full(self.__shape, 1.0 / self.__simplex_size)
        else:
            return np.empty(self.__shape)

    def _validate_folded_shape(self, folded_val):
        if folded_val.shape != self.__shape:
            return False, 'The folded value has the wrong shape.'
        else:
            return True, ''

    def validate_folded(self, folded_val, validate_value=None):
        shape_ok, err_msg = self._validate_folded_shape(folded_val)
        if not shape_ok:
            raise ValueError(err_msg)
        if validate_value is None:
            validate_value = self.default_validate
        if validate_value:
            if np.any(folded_val < 0):
                return False, 'Some values are negative.'
            simplex_sums = np.sum(folded_val, axis=-1)
            if np.any(np.abs(simplex_sums - 1) > 1e-12):
                return False, 'The simplexes do not sum to one.'
        return True, ''

    def fold(self, flat_val, free, validate_value=None):
        flat_size = self.flat_length(free)
        if len(flat_val) != flat_size:
            raise ValueError('flat_val is the wrong length.')
        if free:
            free_mat = np.reshape(flat_val, self.__free_shape)
            return _constrain_simplex_matrix(free_mat)
        else:
            folded_val = np.reshape(flat_val, self.__shape)
            valid, msg = self.validate_folded(folded_val, validate_value)
            if not valid:
                raise ValueError(msg)
            return folded_val

    def flatten(self, folded_val, free, validate_value=None):
        valid, msg = self.validate_folded(folded_val, validate_value)
        if not valid:
            raise ValueError(msg)
        if free:
            return _unconstrain_simplex_matrix(folded_val).flatten()
        else:
            return folded_val.flatten()

    @classmethod
    def from_json(cls, json_string):
        """
        Return a pattern instance from ``json_string`` created by ``to_json``.
        """
        json_dict = json.loads(json_string)
        cls._validate_json_dict_type(json_dict)
        return cls(
            simplex_size=json_dict['simplex_size'],
            array_shape=tuple(json_dict['array_shape']),
            default_validate=json_dict['default_validate'])

    def flat_indices(self, folded_bool, free):
        shape_ok, err_msg = self._validate_folded_shape(folded_bool)
        if not shape_ok:
            raise ValueError(err_msg)
        if not free:
            folded_indices = self.fold(
                np.arange(self.flat_length(False)),
                validate_value=False, free=False)
            return folded_indices[folded_bool]
        else:
            # Every element of a particular simplex depends on all
            # the free values for that simplex.

            # The simplex is the last index, which moves the fastest.
            indices = []
            offset = 0
            free_simplex_length = self.__simplex_size - 1
            array_ranges = (range(n) for n in self.__array_shape)
            for ind in itertools.product(*array_ranges):
                if np.any(folded_bool[ind]):
                    free_inds = np.arange(
                        offset * free_simplex_length,
                        (offset + 1) * free_simplex_length)
                    indices.append(free_inds)
                offset += 1
            if len(indices) > 0:
                return np.hstack(indices)
            else:
                return np.array([])


register_pattern_json(SimplexArrayPattern)
