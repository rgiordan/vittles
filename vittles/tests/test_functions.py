#!/usr/bin/env python3

import unittest
from numpy.testing import assert_array_almost_equal

import autograd.numpy as np
from autograd.test_util import check_grads

import itertools

import vittles


def get_test_pattern():
    pattern = vittles.PatternDict()
    pattern['array'] = vittles.NumericArrayPattern((2, 3, 4), lb=-1, ub=2)
    pattern['mat'] = vittles.PSDSymmetricMatrixPattern(3)
    pattern['simplex'] = vittles.SimplexArrayPattern(2, (3, ))
    subdict = vittles.PatternDict()
    subdict['array2'] = vittles.NumericArrayPattern((2, ), lb=-3, ub=5)
    pattern['dict'] = subdict
    return pattern


class TestPatterns(unittest.TestCase):
    def _test_functor(self, original_fun, argnums, args, kwargs):
        argnums_array = np.atleast_1d(argnums)
        functor = vittles.Functor(original_fun, argnums)
        functor_args = ()
        for i in argnums_array:
            functor_args += (args[i], )

        # Check that you have to set the cache.
        with self.assertRaises(ValueError):
            functor(*functor_args)

        functor.cache_args(*args, **kwargs)
        assert_array_almost_equal(
            original_fun(*args, **kwargs),
            functor(*functor_args))

        # Check you can clear the cache.
        functor.clear_cached_args()
        with self.assertRaises(ValueError):
            functor(*functor_args)

        # Check that the argument length must be correct.
        functor.cache_args(*args, **kwargs)
        bad_functor_args = functor_args + (2, )
        with self.assertRaises(ValueError):
            functor(*bad_functor_args)


    def test_functors(self):
        x = 1
        y = 2
        z = -3
        zz = -4

        def testfun(x):
            return x
        self._test_functor(testfun, 0, (x, ), {})

        def testfun(x, y):
            return x + y
        for argnums in [0, 1, [0, 1]]:
            self._test_functor(testfun, argnums, (x, y), {})

        def testfun(x, y, z=3):
            return x + y + z
        for argnums in [0, 1, [0, 1]]:
            self._test_functor(testfun, argnums, (x, y), {'z': 3})

        def testfun(x, y, z=3, zz=4):
            return x + y + z + zz
        for argnums in [0, 1, [0, 1]]:
            self._test_functor(testfun, argnums, (x, y), {'z': 3, 'zz': 4})


    def _test_flatten_function(self, original_fun, patterns, free, argnums,
                               args, flat_args, kwargs):

        fun_flat = vittles.FlattenedFunction(
            original_fun, patterns, free, argnums)

        # Sanity check that the flat_args were set correctly.
        argnums_array = np.atleast_1d(argnums)
        patterns_array = np.atleast_1d(patterns)
        free_array = np.atleast_1d(free)
        for i in range(len(argnums_array)):
            argnum = argnums_array[i]
            pattern = patterns_array[i]
            assert_array_almost_equal(
                flat_args[argnum],
                pattern.flatten(args[argnum], free=free_array[i]))

        # Check that the flattened and original function are the same.
        assert_array_almost_equal(
            original_fun(*args, **kwargs),
            fun_flat(*flat_args, **kwargs))


    def test_flatten_function(self):
        pattern = get_test_pattern()
        param_val = pattern.random()
        x = 3
        y = 4
        z = 5

        def testfun1(param_val):
            return \
                np.mean(param_val['array'] ** 2) + \
                np.mean(param_val['mat'] ** 2)

        def testfun2(x, param_val, y=5):
            return \
                np.mean(param_val['array'] ** 2) + \
                np.mean(param_val['mat'] ** 2) + x**2 + y**2

        def testfun3(param_val, x, y=5):
            return \
                np.mean(param_val['array'] ** 2) + \
                np.mean(param_val['mat'] ** 2) + x**2 + y**2

        for free in [False, True]:
            param_val_flat = pattern.flatten(param_val, free=free)
            self._test_flatten_function(
                testfun1, pattern, free, 0,
                (param_val, ), (param_val_flat, ), {})

            self._test_flatten_function(
                testfun2, pattern, free, 1,
                (x, param_val, ), (x, param_val_flat, ), {'y': 5})

            self._test_flatten_function(
                testfun3, pattern, free, 0,
                (param_val, x, ), (param_val_flat, x), {'y': 5})

            # Test once with arrays.
            self._test_flatten_function(
                testfun3, [pattern], [free], [0],
                (param_val, x, ), (param_val_flat, x), {'y': 5})

        # Test two-parameter flattening.
        def testfun1(a, b):
            return np.mean(a**2) + np.mean(b**2)

        def testfun2(x, a, z, b, y=5):
            return np.mean(a**2) + np.mean(b**2) + x**2 + y**2 + z**2

        def testfun3(a, z, b, x, y=5):
            return np.mean(a**2) + np.mean(b**2) + x**2 + y**2 + z**2

        a = param_val['array']
        b = param_val['mat']
        ft_list = [False, True]
        for (a_free, b_free) in itertools.product(ft_list, ft_list):
            a_flat = pattern['array'].flatten(param_val['array'], free=a_free)
            b_flat = pattern['mat'].flatten(param_val['mat'], free=b_free)

            self._test_flatten_function(
                testfun1, [pattern['array'], pattern['mat']],
                [a_free, b_free], [0, 1],
                (a, b, ), (a_flat, b_flat, ), {})

            self._test_flatten_function(
                testfun1, [pattern['mat'], pattern['array']],
                [b_free, a_free], [1, 0],
                (a, b, ), (a_flat, b_flat, ), {})

            self._test_flatten_function(
                testfun2, [pattern['array'], pattern['mat']],
                [a_free, b_free], [1, 3],
                (x, a, z, b, ), (x, a_flat, z, b_flat, ), {'y': 5})

            self._test_flatten_function(
                testfun2, [pattern['mat'], pattern['array']],
                [b_free, a_free], [3, 1],
                (x, a, z, b, ), (x, a_flat, z, b_flat, ), {'y': 5})

            self._test_flatten_function(
                testfun3, [pattern['array'], pattern['mat']],
                [a_free, b_free], [0, 2],
                (a, z, b, x, ), (a_flat, z, b_flat, x, ), {'y': 5})

            self._test_flatten_function(
                testfun3, [pattern['mat'], pattern['array']],
                [b_free, a_free], [2, 0],
                (a, z, b, x, ), (a_flat, z, b_flat, x, ), {'y': 5})


    def test_autograd(self):
        pattern = get_test_pattern()

        # The autodiff tests produces non-symmetric matrices.
        pattern['mat'].default_validate = False
        param_val = pattern.random()

        def tf1(param_val):
            return \
                np.mean(param_val['array'] ** 2) + \
                np.mean(param_val['mat'] ** 2)

        for free in [True, False]:
            tf1_flat = vittles.FlattenedFunction(tf1, pattern, free)
            param_val_flat = pattern.flatten(param_val, free=free)
            check_grads(
                tf1_flat, modes=['rev', 'fwd'], order=2)(param_val_flat)


if __name__ == '__main__':
    unittest.main()
