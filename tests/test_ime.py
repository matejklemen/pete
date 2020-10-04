import unittest
import torch
from explain_nlp.methods.ime import IMEExplainer, estimate_feature_samples, estimate_max_samples


class TestExactShapleyExplainer(unittest.TestCase):
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
