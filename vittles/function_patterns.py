import copy
import numpy as np
import warnings


class TransformFunctionInput:
    """
    Convert a function of folded (or flattened) values into one that takes
    flattened (or folded) values.

    Examples
    ----------
    .. code-block:: python

        mat_pattern = paragami.PSDSymmetricMatrixPattern(3)

        def fun(offset, mat, kwoffset=3):
            return np.linalg.slogdet(mat + offset + kwoffset)[1]

        flattened_fun = paragami.TransformFunctionInput(
            original_fun=fun, patterns=mat_pattern,
            free=True, argnums=1, original_is_flat=False)

        # pd_mat is a matrix:
        pd_mat = np.eye(3) + np.full((3, 3), 0.1)

        # pd_mat_flat is an unconstrained vector:
        pd_mat_flat = mat_pattern.flatten(pd_mat, free=True)

        # These two functions return the same value:
        print('Original: {}'.format(
              fun(2, pd_mat, kwoffset=3)))
        print('Flat: {}'.format(
              flattened_fun(2, pd_mat_flat, kwoffset=3)))
    """
    def __init__(self, original_fun, patterns, free,
                 original_is_flat, argnums=None, ):
        """
        Parameters
        ------------
        original_fun: callable
            A function that takes one or more folded values as input.

        patterns: `paragami.Pattern` or list of `paragami.PatternPattern`
            A single pattern or array of patterns describing the input to
            `original_fun`.

        free: `bool` or list of `bool`
            Whether or not the corresponding elements of `patterns` should
            use free or non-free flattened values.

        original_is_flat: `bool`
            If `True`, convert `original_fun` from taking flat arguments to
            one taking folded arguments.  If `False`, convert `original_fun`
            from taking folded arguments to one taking flat arguments.

        argnums: `int` or list of `int`
            The 0-indexed locations of the corresponding pattern in `patterns`
            in the order of the arguments fo `original_fun`.
        """

        self._fun = original_fun
        self._patterns = np.atleast_1d(patterns)
        if argnums is None:
            argnums = np.arange(0, len(self._patterns))
        self._argnums = np.atleast_1d(argnums)
        self._argnum_sort = np.argsort(self._argnums)
        self.free = np.broadcast_to(free, self._patterns.shape)
        self._original_is_flat = original_is_flat

        self._validate_args()

    def _validate_args(self):
        if self._patterns.ndim != 1:
            raise ValueError('patterns must be a 1d vector.')
        if self._argnums.ndim != 1:
            raise ValueError('argnums must be a 1d vector.')
        if len(self._argnums) != len(np.unique(self._argnums)):
            raise ValueError('argnums must not contain duplicated values.')
        if len(self._argnums) != len(self._patterns):
            raise ValueError('argnums must be the same length as patterns.')
        # These two actually cannot be violated because the broadcast_to
        # would fail first.  In case something changes later, leave them in
        # as checks.
        if self.free.ndim != 1:
            raise ValueError(
                'free must be a single boolean or a 1d vector of booleans.')
        if len(self.free) != len(self._patterns):
            raise ValueError(
                'free must broadcast to the same shape as patterns.')

    def __str__(self):
        return(('Function: {}\nargnums: {}\n' +
                'free: {}\npatterns: {}, orignal_is_flat: {}').format(
                self._fun, self._argnums,
                self.free, self._patterns, self._original_is_flat))

    def __call__(self, *args, **kwargs):
        # Loop through the arguments from beginning to end, replacing
        # parameters with their flattened values.
        new_args = ()
        last_argnum = 0
        for i in self._argnum_sort:
            argnum = self._argnums[i]
            if self._original_is_flat:
                val_for_orig = \
                    self._patterns[i].flatten(args[argnum], free=self.free[i])
            else:
                val_for_orig = \
                    self._patterns[i].fold(args[argnum], free=self.free[i])
            new_args += args[last_argnum:argnum] + (val_for_orig, )
            last_argnum = argnum + 1
        new_args += args[last_argnum:len(args)]

        return self._fun(*new_args, **kwargs)


class FoldFunctionInput(TransformFunctionInput):
    """A convenience wrapper of `paragami.TransformFunctionInput`.

    See also
    -----------
    paragami.TransformFunctionInput
    """
    def __init__(self, original_fun, patterns, free, argnums=None):
        super().__init__(
            original_fun=original_fun,
            patterns=patterns,
            free=free,
            original_is_flat=True,
            argnums=argnums)


class FlattenFunctionInput(TransformFunctionInput):
    """A convenience wrapper of `paragami.TransformFunctionInput`.

    See also
    -----------
    paragami.TransformFunctionInput
    """
    def __init__(self, original_fun, patterns, free, argnums=None):
        super().__init__(
            original_fun=original_fun,
            patterns=patterns,
            free=free,
            original_is_flat=False,
            argnums=argnums)


class FoldFunctionOutput:
    """
    Convert a function returning a flat value to one returning a folded value.

    Examples
    ----------
    .. code-block:: python

        mat_pattern = paragami.PSDSymmetricMatrixPattern(3)

        def fun(scale, kwoffset=3):
            mat = np.eye(3) * scale + kwoffset
            return mat_pattern.fold(mat, free=True)

        folded_fun = paragami.FoldFunctionOutput(
            original_fun=fun, pattern=mat_pattern, free=True)

        flat_mat = fun(3, kwoffset=1)
        # These two are the same:
        mat_pattern.fold(flat_mat, free=True)
        folded_fun(3, kwoffset=1)
    """
    def __init__(self, original_fun, pattern, free):
        """
        Parameters
        ------------
        original_fun: callable
            A function that returns a flattened value.

        pattern: `paragami.Pattern`
            A pattern describing how to fold the output.

        free: `bool`
            Whether the returned value is free.
        """

        self._fun = original_fun
        self._pattern = pattern
        self._free = free

    def __str__(self):
        return('Function: {}\nfree: {}\npattern: {}'.format(
            self._fun, self._free, self._pattern))

    def __call__(self, *args, **kwargs):
        flat_val = self._fun(*args, **kwargs)
        return self._pattern.fold(flat_val, free=self._free)


class FoldFunctionInputAndOutput():
    """A convenience wrapper of `paragami.FoldFunctionInput` and
    `paragami.FoldFunctionOutput`.

    See also
    -----------
    paragami.FoldFunctionInput
    paragami.FoldFunctionOutput
    """
    def __init__(self, original_fun,
                 input_patterns, input_free, input_argnums,
                 output_pattern, output_free):
        self._folded_output = \
            FoldFunctionOutput(
                original_fun=original_fun,
                pattern=output_pattern,
                free=output_free)
        self._folded_fun = FoldFunctionInput(
            original_fun=self._folded_output,
            patterns=input_patterns,
            free=input_free,
            argnums=input_argnums)

    def __call__(self, *args, **kwargs):
        return self._folded_fun(*args, **kwargs)
