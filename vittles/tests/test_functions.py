#!/usr/bin/env python3

import unittest
from numpy.testing import assert_array_almost_equal

import autograd.numpy as np
from autograd.test_util import check_grads

import itertools

import paragami


def get_test_pattern():
    # autograd will pass invalid values, so turn off value checking.
    pattern = paragami.PatternDict()
    pattern['array'] = paragami.NumericArrayPattern(
        (2, 3, 4), lb=-1, ub=2, default_validate=False)
    pattern['mat'] = paragami.PSDSymmetricMatrixPattern(
        3, default_validate=False)
    pattern['simplex'] = paragami.SimplexArrayPattern(
        2, (3, ), default_validate=False)
    subdict = paragami.PatternDict()
    subdict['array2'] = paragami.NumericArrayPattern(
        (2, ), lb=-3, ub=5, default_validate=False)
    pattern['dict'] = subdict

    return pattern


def assert_test_dict_equal(d1, d2):
    """Assert that dictionaries corresponding to test pattern are equal.
    """
    for k in ['array', 'mat', 'simplex']:
        assert_array_almost_equal(d1[k], d2[k])
    assert_array_almost_equal(d1['dict']['array2'], d2['dict']['array2'])


class TestFlatteningAndFolding(unittest.TestCase):
    def _test_flatten_function(self, original_fun, patterns, free, argnums,
                               args, flat_args, kwargs):

        fun_flat = paragami.FlattenFunctionInput(
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

        # Check that the string method works.
        str(fun_flat)

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

        # Test bad inits
        with self.assertRaises(ValueError):
            fun_flat = paragami.FlattenFunctionInput(
                testfun1, [[ pattern['mat'] ]], True, 0)

        with self.assertRaises(ValueError):
            fun_flat = paragami.FlattenFunctionInput(
                testfun1, pattern['mat'], True, [[0]])

        with self.assertRaises(ValueError):
            fun_flat = paragami.FlattenFunctionInput(
                testfun1, pattern['mat'], True, [0, 0])

        with self.assertRaises(ValueError):
            fun_flat = paragami.FlattenFunctionInput(
                testfun1, pattern['mat'], True, [0, 1])


    def test_fold_function_output(self):
        pattern = get_test_pattern()
        param_val = pattern.random()
        param_flat = pattern.flatten(param_val, free=False)
        param_free = pattern.flatten(param_val, free=True)

        def get_param(a, b=0.1):
            param_val = pattern.empty(valid=False)
            param_val['array'][:] = a + b
            param_val['mat'] = \
                a * np.eye(param_val['mat'].shape[0]) + b
            param_val['simplex'] = np.full(param_val['simplex'].shape, 0.5)
            param_val['dict']['array2'][:] = a + b

            return param_val

        for free in [False, True]:
            def get_flat_param(a, b=0.1):
                return pattern.flatten(get_param(a, b=b), free=free)

            get_folded_param = paragami.FoldFunctionOutput(
                get_flat_param, pattern=pattern, free=free)
            a = 0.1
            b = 0.2
            assert_test_dict_equal(
                get_param(a, b=b), get_folded_param(a, b=b))

    def test_flatten_and_fold(self):
        pattern = get_test_pattern()
        pattern_val = pattern.random()
        free_val = pattern.flatten(pattern_val, free=True)

        def operate_on_free(free_val, a, b=2):
            return free_val * a + b

        a = 2
        b = 3

        folded_fun = paragami.FoldFunctionInputAndOutput(
            original_fun=operate_on_free,
            input_patterns=pattern,
            input_free=True,
            input_argnums=0,
            output_pattern=pattern,
            output_free=True)

        pattern_out = folded_fun(pattern_val, a, b=b)
        pattern_out_test = pattern.fold(
            operate_on_free(free_val, a, b=b), free=True)
        assert_test_dict_equal(pattern_out_test, pattern_out)

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
            tf1_flat = paragami.FlattenFunctionInput(tf1, pattern, free)
            param_val_flat = pattern.flatten(param_val, free=free)
            check_grads(
                tf1_flat, modes=['rev', 'fwd'], order=2)(param_val_flat)


if __name__ == '__main__':
    unittest.main()
