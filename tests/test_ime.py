import unittest
import torch
from explain_nlp.methods.ime import IMEExplainer
from explain_nlp.methods.utils import estimate_feature_samples, estimate_max_samples


class TestIMEExplainer(unittest.TestCase):
    def test_return_options(self):
        """ Test that the correct things are returned based on keywords used at instantiation time """
        def _model_func(data: torch.Tensor):
            return torch.randn((data.shape[0], 2))

        explainer1 = IMEExplainer(sample_data=torch.tensor([[1, 2, 3], [4, 5, 6]]),
                                  model_func=_model_func)
        res1 = explainer1.explain(instance=torch.tensor([[0, 1, 2]]),
                                  min_samples_per_feature=10,
                                  max_samples=100)

        # By default, return only feature importances (1 per feature)
        self.assertEqual(len(res1), 1)
        self.assertListEqual(list(res1.keys()), ["importance"])
        self.assertIsInstance(res1["importance"], torch.Tensor)
        self.assertTrue(res1["importance"].shape[0] == 3)

        explainer2 = IMEExplainer(sample_data=torch.tensor([[1, 2, 3], [4, 5, 6]]), model_func=_model_func,
                                  return_scores=True, return_num_samples=True, return_variance=True,
                                  return_samples=True)
        res2 = explainer2.explain(instance=torch.tensor([[0, 1, 2]]),
                                  perturbable_mask=torch.tensor([[False, True, True]]),
                                  min_samples_per_feature=10,
                                  max_samples=100)

        EXPECTED_KEYS = ["importance", "var", "num_samples", "samples", "scores"]
        EXPECTED_TYPES = [torch.Tensor, torch.Tensor, torch.Tensor, list, list]
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
        def _model_func(data: torch.Tensor):
            return torch.randn((data.shape[0], 2))

        explainer = IMEExplainer(sample_data=torch.tensor([[1, 2, 3], [4, 5, 6]]), model_func=_model_func,
                                 return_scores=True, return_num_samples=True, return_variance=True,
                                 return_samples=True)
        res = explainer.explain(instance=torch.tensor([[0, 1, 2]]),
                                perturbable_mask=torch.tensor([[False, True, True]]),
                                min_samples_per_feature=10)

        self.assertEqual(res["num_samples"][0], 0)  # 0 samples taken for unperturbable feature
        self.assertIs(res["samples"][0], None)
        self.assertIs(res["scores"][0], None)
        self.assertEqual(res["num_samples"][1], 10)

    def test_required_samples(self):
        alpha = 0.05  # 95% CI

        # Not rigorously tested due to high influence of floating point errors when `max_abs_error` is low
        estimate1 = estimate_feature_samples(torch.tensor([0.0, 0.0, 0.0]), alpha, max_abs_error=0.1)
        self.assertEqual(estimate1.tolist(), [0, 0, 0])

        estimate2 = estimate_feature_samples(torch.tensor([1.0, 0.1, 5.0]), alpha, max_abs_error=1)
        self.assertEqual(estimate2.tolist(), [3, 0, 19])

        estimate3 = estimate_max_samples(torch.tensor([1.0, 0.1, 5.0]), alpha, max_abs_error=1)
        self.assertEqual(int(estimate3), 22)

    def test_exceptions_on_invalid_input(self):
        # No model function provided, either at instantiation time or at explanation time
        explainer1 = IMEExplainer(sample_data=torch.tensor([[1, 2, 3],
                                                            [4, 5, 6]]))
        self.assertRaises(ValueError,
                          lambda: explainer1.explain(instance=torch.tensor([[0, 1, 2]])))

        def _model_func(data: torch.Tensor):
            return torch.randn((data.shape[0], 2))

        # Specified a minimum of 10 samples per features, but specified only 20 max samples in total
        # (10 samples * 3 features = 30 samples, which is lower than 20 max samples)
        explainer2 = IMEExplainer(sample_data=torch.tensor([[1, 2, 3], [4, 5, 6]]),
                                  model_func=_model_func)
        self.assertRaises(AssertionError,
                          lambda: explainer2.explain(instance=torch.tensor([[0, 1, 2]]),
                                                     min_samples_per_feature=10,
                                                     max_samples=20))
