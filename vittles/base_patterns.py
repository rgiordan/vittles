import autograd
import json
import numpy as np
from scipy.sparse import coo_matrix

class Pattern(object):
    """
    A pattern for folding and unfolding a parameter.

    Attributes
    ------------

    Methods
    ---------
    __str__(): A string description of the pattern.
    __eq__(): Check two patterns for equality.

    Examples
    ------------
    Todo.
    """
    def __init__(self, flat_length, free_flat_length):
        """
        Parameters
        -----------
        flat_length : int
            The length of a non-free flattened vector.
        free_flat_length : int
            The length of a free flattened vector.
        """
        self._flat_length = flat_length
        self._free_flat_length = free_flat_length

        self._freeing_jacobian = autograd.jacobian(self._freeing_transform)
        self._unfreeing_jacobian = autograd.jacobian(self._unfreeing_transform)

    def __str__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return self.as_dict() == other.as_dict()

    @classmethod
    def json_typename(cls):
        return '.'.join([ cls.__module__, cls.__name__])

    def as_dict(self):
        """
        Return a dictionary of attributes that determine equality.

        If the keys of the returned dictionary match the arguments to
        ``__init__``, then the default methods for ``to_json`` and
        ``from_json`` will work.
        """
        raise NotImplementedError()

    def _freeing_transform(self, flat_val):
        """
        From the flat to the free flat value.
        """
        return self.flatten(self.fold(flat_val, free=False), free=True)

    def _unfreeing_transform(self, free_flat_val):
        """
        From the free flat to the flat value.
        """
        return self.flatten(self.fold(free_flat_val, free=True), free=False)

    def fold(self, flat_val, free, validate=None):
        """
        Fold a flat value into a parameter.

        Parameters
        -----------
        flat_val: 1-d float array
            The flattened value.
        free: Boolean
            Whether or not the flattened value is a free parameterization.
        validate: Boolean
            Whether to validate that the folded value respects the constraints.
            If None, default either to the pattern's default or, if that is
            unspecified, to True.

        Returns
        ---------
        The parameter value in its original "folded" shape.
        """
        raise NotImplementedError()

    def flatten(self, folded_val, free, validate=None):
        """
        Flatten a folded value into a flat vector.

        Parameters
        -----------
        folded_val
            The parameter in its original "folded" shape.
        free: Boolean
            Whether or not the flattened value is to be in a free
            parameterization.
        validate: Boolean
            Whether to validate that the folded value respects the constraints.
            If None, default either to the pattern's default or, if that is
            unspecified, to True.

        Returns
        ---------
        1-d vector of floats
            The flattened value.
        """
        raise NotImplementedError()

    # Get the size of the flattened version.
    def flat_length(self, free):
        """
        Return the length of the pattern's flattened value.

        Parameters
        -----------
        free: Boolean
            Whether or not the flattened value is to be in a free
            parameterization.

        Returns
        ---------
        int
            The length of the pattern's flattened value.
        """
        if free:
            return self._free_flat_length
        else:
            return self._flat_length

    # Methods to generate valid values.
    def empty(self, valid):
        """
        Return an empty parameter in its "folded" shape.

        Parameters
        -----------
        valid: Boolean
            Whether or folded shape should be filled with valid values.

        Returns
        ---------
        A parameter value in its original "folded" shape.
        """
        raise NotImplementedError()

    def random(self):
        """
        Return an random, valid parameter in its "folded" shape.

        Returns
        ---------
        A random parameter value in its original "folded" shape.
        """
        return self.fold(np.random.random(self._free_flat_length), free=True)

    def freeing_jacobian(self, folded_val, sparse=True):
        """
        Return the Jacobian of the map from a flat free value to a flat value.

        If the folded value of the parameter is ``val``, ``val_flat =
        flatten(val, free=False)``, and ``val_freeflat = flatten(val,
        free=True)``, then this calculates the Jacobian matrix ``d val_free / d
        val_freeflat``.  For entries with no dependence between them, the
        Jacobian is taken to be zero.

        Parameters
        -------------
        folded_val:
            The folded value at which the Jacobian is to be evaluated.
        sparse: boolean
            Whether to return a sparse or a dense matrix.

        Returns
        -------------
        Numeric matrix:
            The Jacobian matrix ``d val_free / d val_freeflat``. Consistent with
            standard Jacobian notation, the elements of ``val_free`` correspond
            to the rows of the Jacobian matrix and the elements of
            ``val_freeflat`` correspond to the columns.
        """
        flat_val = self.flatten(folded_val, free=False)
        jac = self._freeing_jacobian(flat_val)
        if sparse:
            return coo_matrix(jac)
        else:
            return jac

    def unfreeing_jacobian(self, folded_val, sparse=True):
        """
        Return the Jacobian of the map from a flat value to a flat free value.

        If the folded value of the parameter is ``val``, ``val_flat =
        flatten(val, free=False)``, and ``val_freeflat = flatten(val,
        free=True)``, then this calculates the Jacobian matrix ``d val_freeflat /
        d val_free``.  For entries with no dependence between them, the Jacobian
        is taken to be zero.

        Parameters
        -------------
        folded_val:
            The folded value at which the Jacobian is to be evaluated.
        sparse: boolean
            Whether to return a sparse or a dense matrix.

        Returns
        -------------
        Numeric matrix:
            The Jacobian matrix ``d val_freeflat / d val_free``. Consistent with
            standard Jacobian notation, the elements of ``val_freeflat``
            correspond to the rows of the Jacobian matrix and the elements of
            ``val_free`` correspond to the columns.
        """
        freeflat_val = self.flatten(folded_val, free=True)
        jac = self._unfreeing_jacobian(freeflat_val)
        if sparse:
            return coo_matrix(jac)
        else:
            return jac

    def to_json(self):
        """
        Return a JSON representation of the pattern.
        """
        return json.dumps(self.as_dict())

    @classmethod
    def _validate_json_dict_type(cls, json_dict):
        if json_dict['pattern'] != cls.json_typename():
            error_string = \
                ('{}.from_json must be called on a json_string made ' +
                 'from a the same pattern type.  The json_string ' +
                 'pattern type was {}.').format(
                    cls.json_typename(), json_dict['pattern'])
            raise ValueError(error_string)

    @classmethod
    def from_json(cls, json_string):
        """
        Return a pattern instance from ``json_string`` created by ``to_json``.
        """
        json_dict = json.loads(json_string)
        cls._validate_json_dict_type(json_dict)
        del json_dict['pattern']
        return cls(**json_dict)
