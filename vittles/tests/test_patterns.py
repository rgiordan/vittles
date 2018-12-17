#!/usr/bin/env python3
import autograd
import copy
import unittest
from numpy.testing import assert_array_almost_equal
import numpy as np
import scipy as sp

import json
import collections

import vittles

from autograd.test_util import check_grads

def _test_pattern(testcase, pattern, valid_value,
                  check_equal=assert_array_almost_equal,
                  jacobian_ad_test=True):

    print('Testing pattern {}'.format(pattern))

    ###############################
    # Execute required methods.
    empty_val = pattern.empty(valid=True)
    pattern.flatten(empty_val, free=False)
    empty_val = pattern.empty(valid=False)

    random_val = pattern.random()
    pattern.flatten(random_val, free=False)

    str(pattern)

    # Make sure to test != using a custom test.
    testcase.assertTrue(pattern == pattern)

    ###############################
    # Test folding and unfolding.
    for free in [True, False]:
        flat_val = pattern.flatten(valid_value, free=free)
        testcase.assertEqual(len(flat_val), pattern.flat_length(free))
        folded_val = pattern.fold(flat_val, free=free)
        check_equal(valid_value, folded_val)
        if hasattr(valid_value, 'shape'):
            testcase.assertEqual(valid_value.shape, folded_val.shape)

    ####################################
    # Test conversion to and from JSON.
    pattern_dict = pattern.as_dict()
    json_typename = pattern.json_typename()
    json_string = pattern.to_json()
    json_dict = json.loads(json_string)
    testcase.assertTrue('pattern' in json_dict.keys())
    testcase.assertTrue(json_dict['pattern'] == json_typename)
    new_pattern = vittles.get_pattern_from_json(json_string)
    testcase.assertTrue(new_pattern == pattern)

    ############################################
    # Test the freeing and unfreeing Jacobians.
    def freeing_transform(flat_val):
        return pattern.flatten(
            pattern.fold(flat_val, free=False), free=True)

    def unfreeing_transform(free_flat_val):
        return pattern.flatten(
            pattern.fold(free_flat_val, free=True), free=False)

    ad_freeing_jacobian = autograd.jacobian(freeing_transform)
    ad_unfreeing_jacobian = autograd.jacobian(unfreeing_transform)

    for sparse in [True, False]:
        flat_val = pattern.flatten(valid_value, free=False)
        freeflat_val = pattern.flatten(valid_value, free=True)
        freeing_jac = pattern.freeing_jacobian(valid_value, sparse)
        unfreeing_jac = pattern.unfreeing_jacobian(valid_value, sparse)
        free_len = pattern.flat_length(free=False)
        flatfree_len = pattern.flat_length(free=True)

        # Check the shapes.
        testcase.assertTrue(freeing_jac.shape == (flatfree_len, free_len))
        testcase.assertTrue(unfreeing_jac.shape == (free_len, flatfree_len))

        # Check the values of the Jacobians.
        if sparse:
            # The Jacobians should be inverses of one another and full rank
            # in the free flat space.
            assert_array_almost_equal(
                np.eye(flatfree_len),
                np.array((freeing_jac @ unfreeing_jac).todense()))
            if jacobian_ad_test:
                assert_array_almost_equal(
                    ad_freeing_jacobian(flat_val),
                    np.array(freeing_jac.todense()))
                assert_array_almost_equal(
                    ad_unfreeing_jacobian(freeflat_val),
                    np.array(unfreeing_jac.todense()))
        else:
            # The Jacobians should be inverses of one another and full rank
            # in the free flat space.
            assert_array_almost_equal(
                np.eye(flatfree_len), freeing_jac @ unfreeing_jac)
            if jacobian_ad_test:
                assert_array_almost_equal(
                    ad_freeing_jacobian(flat_val), freeing_jac)
                assert_array_almost_equal(
                    ad_unfreeing_jacobian(freeflat_val), unfreeing_jac)


class TestBasicPatterns(unittest.TestCase):
    def test_simplex_array_patterns(self):
        def test_shape_and_size(simplex_size, array_shape):
            shape = array_shape + (simplex_size, )
            valid_value = np.random.random(shape) + 0.1
            valid_value = \
                valid_value / np.sum(valid_value, axis=-1, keepdims=True)

            pattern = vittles.SimplexArrayPattern(
                simplex_size, array_shape)
            _test_pattern(self, pattern, valid_value)

        test_shape_and_size(4, (2, 3))
        test_shape_and_size(2, (2, 3))
        test_shape_and_size(2, (2, ))

        self.assertTrue(
            vittles.SimplexArrayPattern(3, (2, 3)) !=
            vittles.SimplexArrayPattern(3, (2, 4)))

        self.assertTrue(
            vittles.SimplexArrayPattern(4, (2, 3)) !=
            vittles.SimplexArrayPattern(3, (2, 3)))

    def test_numeric_array_patterns(self):
        for test_shape in [(1, ), (2, ), (2, 3), (2, 3, 4)]:
            valid_value = np.random.random(test_shape)
            pattern = vittles.NumericArrayPattern(test_shape)
            _test_pattern(self, pattern, valid_value)

            pattern = vittles.NumericArrayPattern(test_shape, lb=-1)
            _test_pattern(self, pattern, valid_value)

            pattern = vittles.NumericArrayPattern(test_shape, ub=2)
            _test_pattern(self, pattern, valid_value)

            pattern = vittles.NumericArrayPattern(test_shape, lb=-1, ub=2)
            _test_pattern(self, pattern, valid_value)

            # Test equality comparisons.
            self.assertTrue(
                vittles.NumericArrayPattern((1, 2)) !=
                vittles.NumericArrayPattern((1, )))

            self.assertTrue(
                vittles.NumericArrayPattern((1, 2)) !=
                vittles.NumericArrayPattern((1, 3)))

            self.assertTrue(
                vittles.NumericArrayPattern((1, 2), lb=2) !=
                vittles.NumericArrayPattern((1, 2)))

            self.assertTrue(
                vittles.NumericArrayPattern((1, 2), lb=2, ub=4) !=
                vittles.NumericArrayPattern((1, 2), lb=2))

            # Check that singletons work.
            pattern = vittles.NumericArrayPattern(shape=(1, ))
            _test_pattern(self, pattern, 1.0)

    def test_psdsymmetric_matrix_patterns(self):
        dim = 3
        valid_value = np.eye(dim) * 3 + np.full((dim, dim), 0.1)
        pattern = vittles.PSDSymmetricMatrixPattern(dim)
        _test_pattern(self, pattern, valid_value)

        pattern = vittles.PSDSymmetricMatrixPattern(dim, diag_lb=0.5)
        _test_pattern(self, pattern, valid_value)

        self.assertTrue(
            vittles.PSDSymmetricMatrixPattern(3) !=
            vittles.PSDSymmetricMatrixPattern(4))

        self.assertTrue(
            vittles.PSDSymmetricMatrixPattern(3, diag_lb=2) !=
            vittles.PSDSymmetricMatrixPattern(3))


class TestContainerPatterns(unittest.TestCase):
    def test_dictionary_patterns(self):
        def test_pattern(dict_pattern, dict_val):
            # autograd can't differnetiate the folding of a dictionary
            # because it involves assignment to elements of a dictionary.
            _test_pattern(self, dict_pattern, dict_val,
                          check_equal=check_dict_equal,
                          jacobian_ad_test=False)

        def check_dict_equal(dict1, dict2):
            self.assertEqual(dict1.keys(), dict2.keys())
            for key in dict1:
                if type(dict1[key]) is collections.OrderedDict:
                    check_dict_equal(dict1[key], dict2[key])
                else:
                    assert_array_almost_equal(dict1[key], dict2[key])

        print('dictionary pattern test: one element')
        dict_pattern = vittles.PatternDict()
        dict_pattern['a'] = \
            vittles.NumericArrayPattern((2, 3, 4), lb=-1, ub=2)
        test_pattern(dict_pattern, dict_pattern.random())

        print('dictionary pattern test: two elements')
        dict_pattern['b'] = \
            vittles.NumericArrayPattern((5, ), lb=-1, ub=10)
        test_pattern(dict_pattern, dict_pattern.random())

        print('dictionary pattern test: third matrix element')
        dict_pattern['c'] = \
            vittles.PSDSymmetricMatrixPattern(size=3)
        test_pattern(dict_pattern, dict_pattern.random())

        print('dictionary pattern test: sub-dictionary')
        subdict = vittles.PatternDict()
        subdict['suba'] = vittles.NumericArrayPattern((2, ))
        dict_pattern['d'] = subdict
        test_pattern(dict_pattern, dict_pattern.random())

        # Test keys.
        self.assertEqual(list(dict_pattern.keys()), ['a', 'b', 'c', 'd'])

        # Check that it works with ordinary dictionaries, not only OrderedDict.
        print('dictionary pattern test: non-ordered dictionary')
        test_pattern(dict_pattern, dict(dict_pattern.random()))

        # Check deletion and non-equality.
        print('dictionary pattern test: deletion')
        old_dict_pattern = copy.deepcopy(dict_pattern)
        del dict_pattern['b']
        self.assertTrue(dict_pattern != old_dict_pattern)
        test_pattern(dict_pattern, dict_pattern.random())

        # Check modifying an existing array element.
        print('dictionary pattern test: modifying array')
        dict_pattern['a'] = vittles.NumericArrayPattern((2, ), lb=-1, ub=2)
        test_pattern(dict_pattern, dict_pattern.random())

        # Check modifying an existing dictionary element.
        print('dictionary pattern test: modifying sub-dictionary')
        dict_pattern['d'] = \
            vittles.NumericArrayPattern((4, ), lb=-1, ub=10)
        test_pattern(dict_pattern, dict_pattern.random())

        # Check locking
        dict_pattern.lock()

        def delete():
            del dict_pattern['b']

        def add():
            dict_pattern['new'] = \
                vittles.NumericArrayPattern((4, ))

        def modify():
            dict_pattern['a'] = \
                vittles.NumericArrayPattern((4, ))

        self.assertRaises(ValueError, delete)
        self.assertRaises(ValueError, add)
        self.assertRaises(ValueError, modify)

    def test_pattern_array(self):
        array_pattern = vittles.NumericArrayPattern(
            shape=(2, ), lb=-1, ub=10.0)
        pattern_array = vittles.PatternArray((2, 3), array_pattern)
        valid_value = pattern_array.random()
        _test_pattern(self, pattern_array, valid_value)

        matrix_pattern = vittles.PSDSymmetricMatrixPattern(size=2)
        pattern_array = vittles.PatternArray((2, 3), matrix_pattern)
        valid_value = pattern_array.random()
        _test_pattern(self, pattern_array, valid_value)

        base_pattern_array = vittles.PatternArray((2, 1), matrix_pattern)
        pattern_array_array = vittles.PatternArray((1, 3), base_pattern_array)
        valid_value = pattern_array_array.random()
        _test_pattern(self, pattern_array_array, valid_value)

        self.assertTrue(
            vittles.PatternArray((3, 3), matrix_pattern) !=
            vittles.PatternArray((2, 3), matrix_pattern))

        self.assertTrue(
            vittles.PatternArray((2, 3), array_pattern) !=
            vittles.PatternArray((2, 3), matrix_pattern))


class TestJSONFiles(unittest.TestCase):
    def test_json_files(self):
        pattern = vittles.PatternDict()
        pattern['num'] = vittles.NumericArrayPattern((1, 2))
        pattern['mat'] = vittles.PSDSymmetricMatrixPattern(5)

        val_folded = pattern.random()
        extra = np.random.random(5)

        outfile_name = '/tmp/vittles_test_' + str(np.random.randint(1e6))

        vittles.save_folded(outfile_name, val_folded, pattern, extra=extra)

        val_folded_loaded, pattern_loaded, data = \
            vittles.load_folded(outfile_name + '.npz')

        self.assertTrue(pattern_loaded == pattern)
        self.assertTrue(val_folded.keys() == val_folded_loaded.keys())
        for keyname in val_folded.keys():
            assert_array_almost_equal(
                val_folded[keyname], val_folded_loaded[keyname])
        assert_array_almost_equal(extra, data['extra'])






class TestHelperFunctions(unittest.TestCase):
    def _test_logsumexp(self, mat, axis):
        # Test the more numerically stable version with this simple
        # version of logsumexp.
        def logsumexp_simple(mat, axis):
            return np.log(np.sum(np.exp(mat), axis=axis, keepdims=True))

        check_grads(
            vittles.simplex_patterns._logsumexp,
            modes=['fwd', 'rev'], order=3)(mat, axis)

        assert_array_almost_equal(
            logsumexp_simple(mat, axis),
            vittles.simplex_patterns._logsumexp(mat, axis))

    def test_logsumexp(self):
        mat = np.random.random((3, 3, 3))
        self._test_logsumexp(mat, 0)

    def test_pdmatrix_custom_autodiff(self):
        x_vec = np.random.random(6)
        x_mat = vittles.psdmatrix_patterns._unvectorize_ld_matrix(x_vec)

        check_grads(
            vittles.psdmatrix_patterns._vectorize_ld_matrix,
            modes=['fwd', 'rev'], order=3)(x_mat)
        check_grads(
            vittles.psdmatrix_patterns._unvectorize_ld_matrix,
            modes=['fwd', 'rev'], order=3)(x_vec)


if __name__ == '__main__':
    unittest.main()
