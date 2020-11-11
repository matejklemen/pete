import unittest

import torch

from explain_nlp.methods.utils import estimate_feature_samples, estimate_max_samples, sample_permutations


class TestUtils(unittest.TestCase):
    def test_required_samples(self):
        alpha = 1 - 0.95  # 95% CI

        # Not rigorously tested due to high influence of floating point errors when `max_abs_error` is low
        estimate1 = estimate_feature_samples(torch.tensor([0.0, 0.0, 0.0]), alpha, max_abs_error=0.1)
        self.assertEqual(estimate1.tolist(), [0, 0, 0])

        estimate2 = estimate_feature_samples(torch.tensor([1.0, 0.1, 5.0]), alpha, max_abs_error=1)
        self.assertEqual(estimate2.tolist(), [3, 0, 19])

        estimate3 = estimate_max_samples(torch.tensor([1.0, 0.1, 5.0]), alpha, max_abs_error=1)
        self.assertEqual(int(estimate3), 22)

    def test_permutations(self):
        PERMUTED_INDICES = torch.tensor([0, 1, 5, 6, 7])

        permutations = sample_permutations(10, PERMUTED_INDICES, num_permutations=3)
        self.assertEqual(permutations.shape[0], 3)

        for i in range(3):
            self.assertSetEqual(set(permutations[i].tolist()), set(PERMUTED_INDICES.tolist()))
