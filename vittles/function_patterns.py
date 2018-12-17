import numpy as np

class FlattenedFunction:
    """
    Convert a function of folded values into one that takes flat values.

    Methods
    ---------
    __str__(): Print the details of this flattened function.
    __call__(): Execute the original function at the flattened arguments.

    Examples
    ----------
    .. code-block:: python

        mat_pattern = vittles.PSDSymmetricMatrixPattern(3)

        def fun(offset, mat, kwoffset=3):
            return np.linalg.slogdet(mat + offset + kwoffset)[1]

        flattened_fun = vittles.FlattenedFunction(
            original_fun=fun, patterns=mat_pattern, free=True, argnums=1)

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
    def __init__(self, original_fun, patterns, free, argnums=None):
        """
        Parameters
        ------------
        original_fun: callable function
            A function that takes one or more folded values as input.

        patterns: `Pattern` or array of `Pattern`
            A single pattern or array of patterns describing the input to
            `original_fun`.

        free: bool or array of bool
            Whether or not the corresponding elements of `patterns` should
            use free or non-free flattened values.

        argnums: int or array of int
            The 0-indexed locations of the corresponding pattern in `patterns`
            in the order of the arguments fo `original_fun`.
        """
        self._fun = original_fun
        self._patterns = np.atleast_1d(patterns)
        if argnums is None:
            argnums = np.arange(0, len(self._patterns))
        if len(self._patterns.shape) != 1:
            raise ValueError('patterns must be a 1d vector.')
        self._argnums = np.atleast_1d(argnums)
        self._argnum_sort = np.argsort(self._argnums)
        self.free = np.broadcast_to(free, self._patterns.shape)

        self._validate_args()

        self._cached_args_set = False
        self._cached_args = None
        self._cached_kwargs = None

    def _validate_args(self):
        if len(self._argnums.shape) != 1:
            raise ValueError('argnums must be a 1d vector.')
        if len(self._argnums) != len(np.unique(self._argnums)):
            raise ValueError('argnums must not contain duplicated values.')
        if len(self._argnums) != len(self._patterns):
            raise ValueError('argnums must be the same length as patterns.')
        if len(self.free.shape) != 1:
            raise ValueError(
                'free must be a single boolean or a 1d vector of booleans.')
        if len(self.free) != len(self._patterns):
            raise ValueError(
                'free must broadcast to the same shape as patterns.')

    def __str__(self):
        return('Function: {}\nargnums: {}\nfree: {}\npatterns: {}'.format(
            self._fun, self._argnums, self.free, self._patterns))

    def __call__(self, *args, **kwargs):
        # Loop through the arguments from beginning to end, replacing
        # parameters with their flattened values.
        new_args = ()
        last_argnum = 0
        for i in self._argnum_sort:
            argnum = self._argnums[i]
            folded_val = \
                self._patterns[i].fold(args[argnum], free=self.free[i])
            new_args += args[last_argnum:argnum] + (folded_val, )
            last_argnum = argnum + 1
        new_args += args[last_argnum:len(args)]

        return self._fun(*new_args, **kwargs)


class Functor():
    """
    Cache the values of certain arguments to a function.

    This class converts a function of several arguments to a function of only
    a subset of the arguments, with cached values for the other arguments.
    The ``__call__`` method of a ``Functor`` is a function only of the
    arguments to ``original_fun`` specified by ``argnums``, using cached
    values for all other arguments.

    Methods
    ---------
    argnums(): Print which arguments of the original function
               are arguments of the functor.
    cached_args(): Return a tuple of the cached non-keyword arguments.
    cached_kwargs(): Return a dictionary of the cached keyword arguments.
    cache_args(): Cache the arguments passed to cache_args().
    cache_cached_args(): Clear the cached arguments.

    Examples
    ----------
    .. code-block:: python

        mat_pattern = vittles.PSDSymmetricMatrixPattern(3)

        def fun(offset, mat, kwoffset=3):
            return np.linalg.slogdet(mat + offset + kwoffset)[1]

        flattened_fun = vittles.FlattenedFunction(
            original_fun=fun, patterns=mat_pattern, free=True, argnums=1)

        pd_mat = np.eye(3) + np.full((3, 3), 0.1)
        pd_mat_flat = mat_pattern.flatten(pd_mat, free=True)

        # Define a functor where the first and third arguments are cached:
        flattened_functor = vittles.Functor(
            original_fun=flattened_fun, argnums=1)
        flattened_functor.cache_args(2, pd_mat_flat, kwoffset=3)

        # These three functions return the same value:
        print('Original: {}'.format(
            fun(2, pd_mat, kwoffset=3)))
        print('Flat: {}'.format(
            flattened_fun(2, pd_mat_flat, kwoffset=3)))
        print('Flat functor: {}'.format(
            flattened_functor(pd_mat_flat)))
    """
    def __init__(self, original_fun, argnums):
        """
        Parameters
        ------------
        original_fun: callable function
            A function that takes one or more arguments as input.

        argnums: int or array of int
            The 0-indexed locations of the functor arguments
            in the order of the arguments fo ``original_fun``.
        """
        self._fun = original_fun
        self._argnums = np.atleast_1d(argnums)

        if len(self._argnums) == 0:
            raise ValueError(
                'argnums must contain at least one argument location.')

        if len(self._argnums) != len(np.unique(self._argnums)):
            raise ValueError('argnums must not contain duplicated values.')

        self._argnum_sort = np.argsort(self._argnums)
        self._max_argnum = np.max(self._argnums)

        self._cached_args_set = False
        self._cached_args = None
        self._cached_kwargs = None

    def argnums(self):
        # We don't want _argnums to be modified, so copy before returning.
        return copy.copy(self._argnums)

    def cached_args(self):
        return self._cached_args

    def cached_kwargs(self):
        return self._cached_kwargs

    def cache_args(self, *args, **kwargs):
        """
        Parameters
        ------------
        *args, **kwargs:
            Arguments as they would be passed to ``original_fun``.

        ``cache_args`` must be called before using the ``__call__`` method
        of the ``Functor`` class.

        After calling ``cache_args``, the ``__call__`` method of the functor
        will evalute ``original_fun`` with ``*args`` and ``**kwargs`` except
        for the arguments specified in ``argnums``.
        """
        if len(args) < self._max_argnum:
            raise ValueError(
                'You must cache at least as many arguments as there are'
                'arguments in argnums.')
        self._cached_args_set = True
        self._cached_args = args
        self._cached_kwargs = kwargs

    def clear_cached_args(self):
        """
        Clear the cached values set by ``cache_args``.
        """
        self._cached_args_set = False
        self._cached_args = None
        self._cached_kwargs = None

    def __call__(self, *functor_args):
        if not self._cached_args_set:
            raise ValueError(
                'You must run cache_args to save default arguments' +
                'before calling the functor.')

        if len(functor_args) != len(self._argnums):
            raise ValueError(
                'The arguments to functor must be' +
                'the same length (and order) as argnums.')

        # Loop through the arguments from beginning to end, replacing
        # parameters with the passed values.
        new_args = ()
        last_argnum = 0

        # The argument flat_args is in the same order as _argnums.
        for i in self._argnum_sort:
            argnum = self._argnums[i]
            new_args += \
                self._cached_args[last_argnum:argnum] + (functor_args[i], )
            last_argnum = argnum + 1
        new_args += self._cached_args[last_argnum:len(self._cached_args)]

        return self._fun(*new_args, **self._cached_kwargs)
