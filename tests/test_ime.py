import unittest
import torch
from explain_nlp.methods.ime import IMEExplainer
from explain_nlp.methods.modeling import InterpretableDummy
from explain_nlp.methods.utils import estimate_feature_samples, estimate_max_samples


class TestIMEExplainer(unittest.TestCase):
    def setUp(self):
        self.model = InterpretableDummy()
        self.sample_data = torch.tensor([self.model.to_internal("doctor john disco")["input_ids"][0].tolist(),
                                         self.model.to_internal("banana broccoli banana")["input_ids"][0].tolist(),
                                         self.model.to_internal("broccoli coffee paper")["input_ids"][0].tolist(),
                                         self.model.to_internal("<UNK> coffee bin")["input_ids"][0].tolist()])
        self.sample_input = self.model.to_internal("broccoli bin coffee")["input_ids"]

    def test_return_options(self):
        """ Test that the correct things are returned based on keywords used at instantiation time """
        explainer1 = IMEExplainer(sample_data=self.sample_data, model=self.model)
        res1 = explainer1.explain(instance=self.sample_input,
                                  min_samples_per_feature=10,
                                  max_samples=100)

        # By default, return only feature importances and number of taken samples
        self.assertEqual(len(res1), 2)
        self.assertListEqual(list(res1.keys()), ["importance", "taken_samples"])
        self.assertIsInstance(res1["importance"], torch.Tensor)
        self.assertIsInstance(res1["taken_samples"], int)
        self.assertTrue(res1["importance"].shape[0] == 3)

        explainer2 = IMEExplainer(sample_data=self.sample_data, model=self.model,
                                  return_scores=True, return_num_samples=True, return_variance=True,
                                  return_samples=True)
        res2 = explainer2.explain(instance=self.sample_input,
                                  perturbable_mask=torch.tensor([[False, True, True]]),
                                  min_samples_per_feature=10,
                                  max_samples=100)

        EXPECTED_KEYS = ["importance", "taken_samples", "var", "num_samples", "samples", "scores"]
        EXPECTED_TYPES = [torch.Tensor, int, torch.Tensor, torch.Tensor, list, list]
        self.assertEqual(len(res2), len(EXPECTED_KEYS))
        self.assertListEqual(list(res2.keys()), EXPECTED_KEYS)
        for key, expected_key_type in zip(EXPECTED_KEYS, EXPECTED_TYPES):
            self.assertIsInstance(res2[key], expected_key_type)

        self.assertTrue(res2["importance"].shape[0] == 3)
        self.assertTrue(res2["var"].shape[0] == 3)
        self.assertTrue(res2["num_samples"].shape[0] == 3)
        self.assertTrue(len(res2["samples"]) == 3)
        self.assertTrue(len(res2["scores"]) == 3)

    def test_unperturbable_feature(self):
        """ Test that features marked as unperturbable do not take up samples """
        explainer = IMEExplainer(sample_data=self.sample_data, model=self.model,
                                 return_scores=True, return_num_samples=True, return_variance=True,
                                 return_samples=True)
        res = explainer.explain(instance=self.sample_input,
                                perturbable_mask=torch.tensor([[False, True, True]]),
                                min_samples_per_feature=10)

        self.assertEqual(res["num_samples"][0], 0)  # 0 samples taken for unperturbable feature
        self.assertIs(res["samples"][0], None)
        self.assertIs(res["scores"][0], None)
        self.assertEqual(res["num_samples"][1], 10)

    def test_required_samples(self):
        alpha = 1 - 0.95  # 95% CI

        # Not rigorously tested due to high influence of floating point errors when `max_abs_error` is low
        estimate1 = estimate_feature_samples(torch.tensor([0.0, 0.0, 0.0]), alpha, max_abs_error=0.1)
        self.assertEqual(estimate1.tolist(), [0, 0, 0])

        estimate2 = estimate_feature_samples(torch.tensor([1.0, 0.1, 5.0]), alpha, max_abs_error=1)
        self.assertEqual(estimate2.tolist(), [3, 0, 19])

        estimate3 = estimate_max_samples(torch.tensor([1.0, 0.1, 5.0]), alpha, max_abs_error=1)
        self.assertEqual(int(estimate3), 22)

    def test_exceptions_on_invalid_input(self):
        # Specified a minimum of 10 samples per features, but specified only 20 max samples in total
        # (10 samples * 3 features = 30 samples, which is lower than 20 max samples)
        explainer2 = IMEExplainer(sample_data=self.sample_data, model=self.model)
        self.assertRaises(AssertionError,
                          lambda: explainer2.explain(instance=self.sample_input,
                                                     min_samples_per_feature=10,
                                                     max_samples=20))
