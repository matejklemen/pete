import unittest

import torch
import numpy as np

from explain_nlp.methods.utils import estimate_feature_samples, estimate_max_samples, sample_permutations, \
    incremental_mean, incremental_var


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

    def test_incremental_mean(self):
        l = [1, 5, 2, 0.5, -3, -2]

        acc_mean = 0
        for idx, value in enumerate(l):
            # Classic way vs incremental
            mean_way1 = sum(l[: idx + 1]) / len(l[: idx + 1])
            mean_way2 = incremental_mean(acc_mean, value, len(l[: idx + 1]))

            self.assertAlmostEqual(mean_way1, mean_way2, places=7)
            acc_mean = mean_way2

    def test_incremental_var(self):
        l = [1, 5, 2, 0.5, -3, -2]

        acc_var = 0
        for idx, value in enumerate(l):
            if idx == 0:
                continue

            # Classic way vs incremental
            var_way1 = np.var(l[: idx + 1])
            var_way2 = incremental_var(curr_mean=np.mean(l[: idx]),
                                       curr_var=acc_var,
                                       new_mean=np.mean(l[: (idx + 1)]),
                                       new_value=l[idx],
                                       n=(1 + idx))

            self.assertAlmostEqual(var_way1, var_way2, places=7)
            acc_var = var_way2
