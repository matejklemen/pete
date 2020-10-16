import unittest

import torch

from explain_nlp.methods.ime_mlm import IMEMaskedLMExplainer


class TestIMEMaskedLMExplainer(unittest.TestCase):
    def test_return_options(self):
        """ Test that the correct things are returned based on keywords used at instantiation time """
        def _model_func(data: torch.Tensor):
            return torch.randn((data.shape[0], 2))

        explainer1 = IMEMaskedLMExplainer(model_func=_model_func)
        res1 = explainer1.explain(instance=torch.tensor([[0, 1, 2]]),
                                  min_samples_per_feature=10,
                                  max_samples=100,
                                  token_type_ids=torch.tensor([[0, 0, 0]]),
                                  attention_mask=torch.tensor([[1, 1, 1]]))

        # By default, return only feature importances and number of taken samples
        self.assertEqual(len(res1), 2)
        self.assertListEqual(list(res1.keys()), ["importance", "taken_samples"])
        self.assertIsInstance(res1["importance"], torch.Tensor)
        self.assertIsInstance(res1["taken_samples"], int)
        self.assertTrue(res1["importance"].shape[0] == 3)

        explainer2 = IMEMaskedLMExplainer(model_func=_model_func, return_scores=True, return_num_samples=True,
                                          return_variance=True, return_samples=True)
        res2 = explainer2.explain(instance=torch.tensor([[0, 1, 2]]),
                                  perturbable_mask=torch.tensor([[False, True, True]]),
                                  min_samples_per_feature=10,
                                  max_samples=100,
                                  token_type_ids=torch.tensor([[0, 0, 0]]),
                                  attention_mask=torch.tensor([[1, 1, 1]]))

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
        def _model_func(data: torch.Tensor):
            return torch.randn((data.shape[0], 2))

        explainer = IMEMaskedLMExplainer(model_func=_model_func, return_scores=True, return_num_samples=True,
                                         return_variance=True, return_samples=True)
        res = explainer.explain(instance=torch.tensor([[0, 1, 2]]),
                                perturbable_mask=torch.tensor([[False, True, True]]),
                                min_samples_per_feature=10,
                                token_type_ids=torch.tensor([[0, 0, 0]]),
                                attention_mask=torch.tensor([[1, 1, 1]]))

        self.assertEqual(res["num_samples"][0], 0)  # 0 samples taken for unperturbable feature
        self.assertIs(res["samples"][0], None)
        self.assertIs(res["scores"][0], None)
        self.assertEqual(res["num_samples"][1], 10)
