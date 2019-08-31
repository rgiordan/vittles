#!/usr/bin/env python3

import autograd
import autograd.numpy as np

from copy import deepcopy
from numpy.testing import assert_array_almost_equal
import paragami
import unittest

import vittles


class TestBlockHessian(unittest.TestCase):
    def test_block_hessian(self):
        np.random.seed(42)

        group_size = 3
        num_groups = 10
        d = group_size * num_groups

        pattern = paragami.PatternDict()
        pattern['array'] = \
            paragami.NumericArrayPattern((num_groups, group_size))
        mat_pattern = paragami.PSDSymmetricMatrixPattern(size=group_size)
        pattern['mats'] = paragami.PatternArray((num_groups,), mat_pattern)

        def f(x_dict):
            return 0.5 * np.einsum(
                'nij,ni,nj', x_dict['mats'], x_dict['array'], x_dict['array'])

        f_flat = paragami.FlattenFunctionInput(
            f, argnums=0, free=True, patterns=pattern)

        x = pattern.random()
        x_flat = pattern.flatten(x, free=True)
        f(x)

        f_hess = autograd.hessian(f_flat, argnum=0)
        h0 = f_hess(x_flat)

        inds = []
        for g in range(num_groups):
            x_bool = pattern.empty_bool(False)
            x_bool['array'][g, :] = True
            x_bool['mats'][g, :, :] = True
            inds.append(pattern.flat_indices(x_bool, free=True))
        inds = np.array(inds)

        sparse_hess = vittles.SparseBlockHessian(f_flat, inds)
        block_hess = sparse_hess.get_block_hessian(x_flat)

        assert_array_almost_equal(np.array(block_hess.todense()), h0)

    def test_full_hessian(self):
        np.random.seed(42)

        group_size = 3
        num_groups = 10
        d = group_size * num_groups

        pattern = paragami.PatternDict()
        pattern['array'] = \
            paragami.NumericArrayPattern((num_groups, group_size))
        mat_pattern = paragami.PSDSymmetricMatrixPattern(size=group_size)
        pattern['mats'] = paragami.PatternArray((num_groups,), mat_pattern)
        pattern['scales'] = paragami.NumericVectorPattern(length=2, lb=0.0)

        def f(x_dict):
            scale = np.prod(x_dict['scales'])
            scale_prior = np.exp(-1 * scale)
            return 0.5 * scale * np.einsum(
                'nij,ni,nj', x_dict['mats'], x_dict['array'], x_dict['array'])

        f_flat = paragami.FlattenFunctionInput(
            f, argnums=0, free=True, patterns=pattern)

        x = pattern.random()
        x_flat = pattern.flatten(x, free=True)
        f(x)

        group_inds = []
        x_bool = pattern.empty_bool(False)
        for g in range(num_groups):
            x_bool['array'][g, :] = True
            x_bool['mats'][g, :, :] = True
            group_inds.append(pattern.flat_indices(x_bool, free=True))
            x_bool['array'][g, :] = False
            x_bool['mats'][g, :, :] = False
        group_inds = np.array(group_inds)

        f_hess = autograd.hessian(f_flat, argnum=0)
        h0 = f_hess(x_flat)

        x_bool['scales'][:] = True
        global_inds_paragami = pattern.flat_indices(x_bool, free=True)
        x_bool['scales'][:] = False

        sparse_hess = vittles.SparseBlockHessian(f_flat, group_inds)

        block_hess = \
            sparse_hess.get_block_hessian(x_flat) + \
            sparse_hess.get_global_hessian(x_flat)
        assert_array_almost_equal(h0, block_hess.todense())

        block_hess = \
            sparse_hess.get_block_hessian(x_flat) + \
            sparse_hess.get_global_hessian(
                x_flat, global_inds=global_inds_paragami)
        assert_array_almost_equal(h0, block_hess.todense())

        block_hess = sparse_hess.get_hessian(x_flat, print_every=1)
        assert_array_almost_equal(h0, block_hess.todense())

if __name__ == '__main__':
    unittest.main()
