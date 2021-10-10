import unittest

import numpy as np
import torch
from torch.autograd import gradcheck, Variable

from model import utility
from model.crfae import CRFAE

EPS = 1e-12


class TestCrfaeLayer(unittest.TestCase):
    def test_length_3_case_1(self):
        # multiple root & no dep length constraint
        model = CRFAE(sentence_len=3, batch_size=1, is_multi_root=True)
        weights = torch.log(torch.DoubleTensor([[
            [0, 1, 2],
            [0, 0, 1],
            [0, 3, 0],
        ]]))
        model.forward(weights, weights)
        # Answers
        partition = 9.0
        expected_count = torch.DoubleTensor([[
            [0, 1 / 3, 8 / 9],
            [0, 0, 1 / 9],
            [0, 2 / 3, 0]
        ]])
        best_tree_score = 6.0

        self.assertAlmostEqual(utility.to_scalar(torch.exp(model.log_partition)), partition)
        self.assertTrue(torch.sum((model.diff() - expected_count).abs()) <= EPS)
        self.assertAlmostEqual(np.exp(model.best_score)[0], best_tree_score)

    def test_length_3_case_2(self):
        # single root & no dep length constraint
        model = CRFAE(sentence_len=3, batch_size=1, is_multi_root=False)
        weights = torch.log(torch.DoubleTensor([[
            [0, 1, 2],
            [0, 0, 1],
            [0, 3, 0],
        ]]))
        model.forward(weights, weights)
        # Answers
        partition = 7.0
        expected_count = torch.DoubleTensor([[
            [0, 1 / 7, 6 / 7],
            [0, 0, 1 / 7],
            [0, 6 / 7, 0]
        ]])
        best_tree_score = 6.0

        self.assertAlmostEqual(utility.to_scalar(torch.exp(model.log_partition)), partition)
        self.assertTrue(torch.sum((model.diff() - expected_count).abs()) <= EPS)
        self.assertAlmostEqual(np.exp(model.best_score)[0], best_tree_score)

    def test_length_3_case_3(self):
        # multiple root & dep length constraint = 2 & constraint on root
        model = CRFAE(sentence_len=3, batch_size=1, is_multi_root=True,
                      max_dependency_len=1,
                      length_constraint_on_root=True)
        weights = torch.log(torch.DoubleTensor([[
            [0, 1, 2],
            [0, 0, 1],
            [0, 3, 0],
        ]]))
        model.forward(weights, weights)
        # Answers
        partition = 1.0
        expected_count = torch.DoubleTensor([[
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ]])
        best_tree_score = 1.0

        self.assertAlmostEqual(utility.to_scalar(torch.exp(model.log_partition)), partition)
        self.assertTrue(torch.sum((model.diff() - expected_count).abs()) <= EPS)
        self.assertAlmostEqual(np.exp(model.best_score)[0], best_tree_score)

    def test_length_3_case_4(self):
        # multiple root & dep length constraint = 2 & no constraint on root
        model = CRFAE(sentence_len=3, batch_size=1, is_multi_root=True,
                      max_dependency_len=1,
                      length_constraint_on_root=False)
        weights = torch.log(torch.DoubleTensor([[
            [0, 1, 2],
            [0, 0, 1],
            [0, 3, 0],
        ]]))
        model.forward(weights, weights)
        # Answers
        partition = 9.0
        expected_count = torch.DoubleTensor([[
            [0, 1 / 3, 8 / 9],
            [0, 0, 1 / 9],
            [0, 2 / 3, 0]
        ]])
        best_tree_score = 6.0

        self.assertAlmostEqual(utility.to_scalar(torch.exp(model.log_partition)), partition)
        self.assertTrue(torch.sum((model.diff() - expected_count).abs()) <= EPS)
        self.assertAlmostEqual(np.exp(model.best_score)[0], best_tree_score)

    def test_length_4_case_1(self):
        # multiple root & no dep length constraint
        model = CRFAE(sentence_len=4, batch_size=1, is_multi_root=True,
                      max_dependency_len=-1,
                      length_constraint_on_root=False)
        weights = torch.log(torch.DoubleTensor([[
            [0, 1, 2, 3],
            [0, 0, 1, 4],
            [0, 3, 0, 1],
            [0, 4, 5, 0],
        ]]))
        model.forward(weights, weights)
        # Answers
        partition = 192.
        expected_count = torch.DoubleTensor([[
            [0, 51 / 192, 32 / 192, 159 / 192],
            [0, 0, 20 / 192, 24 / 192],
            [0, 69 / 192, 0, 9 / 192],
            [0, 72 / 192, 140 / 192, 0]
        ]])
        best_tree_score = 60

        self.assertAlmostEqual(utility.to_scalar(torch.exp(model.log_partition)), partition)
        self.assertTrue(torch.sum((model.diff() - expected_count).abs()) <= EPS)
        self.assertAlmostEqual(np.exp(model.best_score)[0], best_tree_score)

    def test_length_4_case_2(self):
        # single root & no dep length constraint
        model = CRFAE(sentence_len=4, batch_size=1, is_multi_root=False,
                      max_dependency_len=-1,
                      length_constraint_on_root=False)
        weights = torch.log(torch.DoubleTensor([[
            [0, 1, 2, 3],
            [0, 0, 1, 4],
            [0, 3, 0, 1],
            [0, 4, 5, 0],
        ]]))
        model.forward(weights, weights)
        # Answers
        partition = 148
        expected_count = torch.DoubleTensor([[
            [0, 25 / 148, 6 / 148, 117 / 148],
            [0, 0, 17 / 148, 24 / 148],
            [0, 51 / 148, 0, 7 / 148],
            [0, 72 / 148, 125 / 148, 0]
        ]])
        best_tree_score = 60

        self.assertAlmostEqual(utility.to_scalar(torch.exp(model.log_partition)), partition)
        self.assertTrue(torch.sum((model.diff() - expected_count).abs()) <= EPS)
        self.assertAlmostEqual(np.exp(model.best_score)[0], best_tree_score)

    def test_length_4_case_3(self):
        # single root & dep length constraint = 2 & no constraint on root
        model = CRFAE(sentence_len=4, batch_size=1, is_multi_root=False,
                      max_dependency_len=1,
                      length_constraint_on_root=False)
        weights = torch.log(torch.DoubleTensor([[
            [0, 1, 2, 3],
            [0, 0, 1, 4],
            [0, 3, 0, 1],
            [0, 4, 5, 0],
        ]]))
        model.forward(weights, weights)
        # Answers
        partition = 52
        expected_count = torch.DoubleTensor([[
            [0, 1 / 52, 6 / 52, 45 / 52],
            [0, 0, 1 / 52, 0 / 52],
            [0, 51 / 52, 0, 7 / 52],
            [0, 0 / 52, 45 / 52, 0]
        ]])
        best_tree_score = 45

        self.assertAlmostEqual(utility.to_scalar(torch.exp(model.log_partition)), partition)
        self.assertTrue(torch.sum((model.diff() - expected_count).abs()) <= EPS)
        self.assertAlmostEqual(np.exp(model.best_score)[0], best_tree_score)

    def test_length_4_case_4(self):
        # single root & dep length constraint = 2 &  constraint on root
        model = CRFAE(sentence_len=4, batch_size=1, is_multi_root=False,
                      max_dependency_len=1,
                      length_constraint_on_root=True)
        weights = torch.log(torch.DoubleTensor([[
            [0, 1, 2, 3],
            [0, 0, 1, 4],
            [0, 3, 0, 1],
            [0, 4, 5, 0],
        ]]))
        model.forward(weights, weights)
        # Answers
        partition = 1
        expected_count = torch.DoubleTensor([[
            [0, 1 / 1, 0 / 1, 0 / 1],
            [0, 0, 1 / 1, 0 / 1],
            [0, 0 / 1, 0, 1 / 1],
            [0, 0 / 1, 0 / 1, 0]
        ]])
        best_tree_score = 1

        self.assertAlmostEqual(utility.to_scalar(torch.exp(model.log_partition)), partition)
        self.assertTrue(torch.sum((model.diff() - expected_count).abs()) <= EPS)
        self.assertAlmostEqual(np.exp(model.best_score)[0], best_tree_score)

    def test_gradcheck(self):
        model = CRFAE(sentence_len=4, batch_size=1, is_multi_root=False,
                      max_dependency_len=1,
                      length_constraint_on_root=True)

        weights = torch.log(torch.DoubleTensor([[
            [0, 1, 2, 3],
            [0, 0, 1, 4],
            [0, 3, 0, 1],
            [0, 4, 5, 0],
        ]]))

        weights2 = weights.clone()

        test = gradcheck(model,
                         [Variable(weights, requires_grad=True), Variable(weights2, requires_grad=True)],
                         eps=1e-4, atol=1e-8)
        self.assertTrue(test)

        if __name__ == '__main__':
            unittest.main()
