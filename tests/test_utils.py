import unittest

import torch
import numpy as np

from explain_nlp.methods.utils import estimate_feature_samples, sample_permutations, \
    incremental_mean, incremental_var, handle_custom_features
from explain_nlp.utils.metrics import iou_score, fidelity, snr_score, hpd_score


class TestUtils(unittest.TestCase):
    def test_custom_feature_handling(self):
        # Use primary units (sequence), generator's representation is same as model's
        (feats, mapped_feats), inds = handle_custom_features(
            custom_features=None,
            perturbable_mask=torch.tensor([[False, True, True, True, True, False]]),
            position_mapping=None
        )

        self.assertIsNone(feats)
        self.assertListEqual(mapped_feats, [[1], [2], [3], [4]])
        self.assertListEqual(inds, [1, 2, 3, 4])

        # Use primary units (sequence pair) with custom mapping
        (feats, mapped_feats), inds = handle_custom_features(
            custom_features=None,
            perturbable_mask=torch.tensor([[False, True, True, True, True, False, True, True, True, False]]),
            position_mapping={0: [1, 2], 1: [3], 2: [4], 3: [5, 6, 7, 8], 4: [9], 5: [10], 6: [11, 12]}
        )

        self.assertIsNone(feats)
        self.assertListEqual(mapped_feats, [[1, 2], [3], [4], [5, 6, 7, 8], [9], [10], [11, 12]])
        self.assertListEqual(inds, [1, 2, 3, 4, 6, 7, 8])

        # Position mapping must either be omitted (None) or provided for all perturbable positions, otherwise raise err
        self.assertRaises(KeyError,
                          lambda: handle_custom_features(
                              custom_features=None,
                              perturbable_mask=torch.tensor([[True, True, True]]),
                              position_mapping={0: [1, 2]}
                          ))

        # Use custom units, generator's representation is same as model's
        (feats, mapped_feats), inds = handle_custom_features(
            custom_features=[[1], [2], [3, 4]],
            perturbable_mask=torch.tensor([[False, True, True, True, True, False]]),
            position_mapping=None
        )
        self.assertListEqual(feats, [[1], [2], [3, 4]])
        self.assertListEqual(mapped_feats, [[1], [2], [3, 4]])
        self.assertListEqual(inds, [6, 7, 8])

        # Use custom units with custom mapping
        (feats, mapped_feats), inds = handle_custom_features(
            custom_features=[[1], [3, 4]],
            perturbable_mask=torch.tensor([[False, True, False, True, True, False]]),
            position_mapping={0: [1, 2, 3, 4], 1: [5, 6], 2: [7]}
        )
        self.assertListEqual(feats, [[1], [3, 4]])
        self.assertListEqual(mapped_feats, [[1, 2, 3, 4], [5, 6, 7]])
        self.assertListEqual(inds, [6, 7])

        # Use incomplete custom units with custom mapping (custom units don't cover all perturbable features):
        # In this case, we expect a warning to be given and the missing groups to be added
        with self.assertWarns(UserWarning):
            (feats, mapped_feats), inds = handle_custom_features(
                custom_features=[[1], [3, 4]],
                perturbable_mask=torch.tensor([[False, True, True, True, True, False]]),
                position_mapping={0: [1, 2, 3], 1: [4], 2: [5, 6], 3: [7]}
            )
            self.assertListEqual(feats, [[1], [3, 4], [2]])  # missing groups added to the end
            self.assertListEqual(mapped_feats, [[1, 2, 3], [5, 6, 7], [4]])
            self.assertListEqual(inds, [6, 7, 8])

        # Cover some perturbable features with two feature groups: not allowed - raise error
        # In this case, feature 2 is covered by groups [2] and [2, 3, 4]
        self.assertRaises(ValueError,
                          lambda: handle_custom_features(
                              custom_features=[[1], [2], [2, 3, 4]],
                              perturbable_mask=torch.tensor([[False, True, True, True, True, False]]),
                              position_mapping={0: [1, 2, 3, 4], 1: [5, 6], 2: [7]}
                          ))

        # Cover some unperturbable feature: not allowed - raise error
        self.assertRaises(ValueError,
                          lambda: handle_custom_features(
                              custom_features=[[1], [2], [3, 4], [5]],
                              perturbable_mask=torch.tensor([[False, True, True, True, True, False]]),
                              position_mapping=None
                          ))

    def test_required_samples(self):
        alpha = 1 - 0.95  # 95% CI

        # Not rigorously tested due to high influence of floating point errors when `max_abs_error` is low
        estimate1 = estimate_feature_samples(torch.tensor([0.0, 0.0, 0.0]), alpha, max_abs_error=0.1)
        self.assertEqual(estimate1.tolist(), [0, 0, 0])

        estimate2 = estimate_feature_samples(torch.tensor([1.0, 0.1, 5.0]), alpha, max_abs_error=1)
        self.assertEqual(estimate2.tolist(), [3, 0, 19])

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

    def test_fidelity(self):
        # completely different prediction
        self.assertAlmostEqual(fidelity(1.0, 0.0), 0.5)
        # correct prediction
        self.assertAlmostEqual(fidelity(0.5, 0.5), 1.0)
        # symmetry
        self.assertAlmostEqual(fidelity(0.2, 0.4), 1.0/1.2)
        self.assertAlmostEqual(fidelity(0.4, 0.2), 1.0/1.2)

        np.testing.assert_allclose(
            fidelity(np.array([1.0, 0.5, 0.2, 0.4]), np.array([0.0, 0.5, 0.4, 0.2])),
            np.array([0.5, 1.0, 1.0/1.2, 1.0/1.2])
        )

    def test_iou(self):
        gt_sents_general = np.array([1, 0, 1, 0, 0])
        gt_sents_all = np.array([1, 1, 1, 1, 1])
        gt_sents_one = np.array([0, 0, 1, 0, 0])
        gt_sents_none = np.array([0, 0, 0, 0, 0])

        pred_sents = np.array([0, 0, 1, 0, 0])
        self.assertAlmostEqual(iou_score(y_true=gt_sents_general, y_pred=pred_sents), 1/2)
        self.assertAlmostEqual(iou_score(y_true=gt_sents_all, y_pred=pred_sents), 1/5)
        self.assertAlmostEqual(iou_score(y_true=gt_sents_one, y_pred=pred_sents), 1.0)
        with self.assertRaises(AssertionError):
            iou_score(y_true=gt_sents_none, y_pred=pred_sents)

        pred_sents = np.array([1, 1, 1, 1, 1])
        self.assertAlmostEqual(iou_score(y_true=gt_sents_general, y_pred=pred_sents), 2/5)
        self.assertAlmostEqual(iou_score(y_true=gt_sents_all, y_pred=pred_sents), 1.0)
        self.assertAlmostEqual(iou_score(y_true=gt_sents_one, y_pred=pred_sents), 1/5)
        with self.assertRaises(AssertionError):
            iou_score(y_true=gt_sents_none, y_pred=pred_sents)

        pred_sents = np.array([0, 0, 0, 0, 0])
        self.assertAlmostEqual(iou_score(y_true=gt_sents_general, y_pred=pred_sents), 0.0)

    def test_hpd(self):
        sent_scores = np.array([0.1, 0.2, 0.05, 0.3, -0.05])
        sent_order = np.argsort(-sent_scores)  # descending

        # Best case: sentence 3 is correct and its score is highest, meaning no false positives
        np.testing.assert_allclose(hpd_score(sent_order, gt=np.eye(1, len(sent_scores), k=3, dtype=np.int32)[0]),
                                   1.0)
        # Worst case: sentence 4 is correct, but its score is lowest, meaning we predict 4 FP an 1 TP
        np.testing.assert_allclose(hpd_score(sent_order, gt=np.eye(1, len(sent_scores), k=4, dtype=np.int32)[0]),
                                   1/5)

        # General case (somewhere in between)
        np.testing.assert_allclose(hpd_score(sent_order, gt=np.eye(1, len(sent_scores), k=1, dtype=np.int32)[0]),
                                   1/2)

        # Invalid inputs:
        # 1) no ground truth sentence
        with self.assertRaises(ValueError):
            hpd_score(sent_order, gt=np.zeros(len(sent_scores)))

        # 2) more than one ground truth sentence
        with self.assertRaises(ValueError):
            hpd_score(sent_order, gt=np.array([1, 1, 0, 0, 0]))

    def test_snr(self):
        uniform_scores = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        uniform_noise_scores = np.array([0.2, 0.4, 0.2, 0.2, 0.2])
        general_scores = np.array([0.0, 1.0, -2.0, 0.1, 0.1])

        # sentence 1 is correct
        gt_vector = np.eye(1, 5, k=1, dtype=np.int32)[0]

        np.testing.assert_allclose(snr_score(general_scores, gt=gt_vector), 2.6199, atol=10e-4)

        # edge case: deviation of noise scores is zero, resulting in division by zero
        # convention: set denominator to one in that case
        with self.assertWarns(UserWarning):
            np.testing.assert_allclose(snr_score(uniform_scores, gt=gt_vector), 0.0)

        with self.assertWarns(UserWarning):
            np.testing.assert_allclose(snr_score(uniform_noise_scores, gt=gt_vector), 0.04)

        # edge case: no noise scores, convention: set mean to 0 and sd to 1
        single_sent_score = np.array([3.0])
        gt_vector = np.array([1], dtype=np.int32)
        np.testing.assert_allclose(snr_score(single_sent_score, gt=gt_vector), 9.0)

        # edge case: single noise score in addition to the correct score, deviation could be problematic
        # convention: sd of a single value = 0.0
        single_noise_score = np.array([3.0, 1.0])
        gt_vector = np.array([1, 0], dtype=np.int32)
        with self.assertWarns(UserWarning):
            np.testing.assert_allclose(snr_score(single_noise_score, gt=gt_vector), 4.0)
