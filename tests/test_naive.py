import unittest

import numpy as np
import torch

from explain_nlp.methods.modeling import DummySentiment
from explain_nlp.methods.naive import ExactShapleyExplainer


def ideal_or2(data):
    """ Ideal OR classifier with two features. """
    assert data.shape[1] == 2
    probas = torch.zeros((data.shape[0], 2), dtype=torch.float32)
    results = torch.logical_or(data[:, [0]], data[:, [1]]).long().squeeze()
    probas[torch.arange(data.shape[0]), results] = 1.0
    return probas


class TestExactShapleyExplainer(unittest.TestCase):
    def setUp(self):
        self.explainer = ExactShapleyExplainer(model_func=ideal_or2,
                                               feature_values=[[0, 1], [0, 1]])

    def test_conditional_expectation(self):
        instance, label = torch.tensor([[1, 0]]), 1

        model_expectation = self.explainer.conditional_expectation(instance, fixed_features=torch.tensor([]))[label]
        self.assertEqual(float(model_expectation), 0.75)  # E(f(X1, X2))

        conditional_first = self.explainer.conditional_expectation(instance, fixed_features=torch.tensor([0]))[label]
        self.assertEqual(float(conditional_first), 1.0)  # E(f(X1, X2) | X1 = 1)

        conditional_all = self.explainer.conditional_expectation(instance, fixed_features=torch.tensor([0, 1]))[label]
        self.assertEqual(float(conditional_all), 1.0)  # E(f(X1, X2) | X1 = 1, X2 = 0)

    def test_prediction_difference(self):
        instance, label = torch.tensor([[1, 0]]), 1

        # having no features at fixed values should differ by 0 from the model expectation
        null_diff = self.explainer.prediction_difference(instance, fixed_features=torch.tensor([]))[label]
        self.assertEqual(float(null_diff), 0.75 - 0.75)

        diff_first_known = self.explainer.prediction_difference(instance, fixed_features=torch.tensor([0]))[label]
        self.assertEqual(float(diff_first_known), 1.0 - 0.75)

        diff_second_known = self.explainer.prediction_difference(instance, fixed_features=torch.tensor([1]))[label]
        self.assertEqual(float(diff_second_known), 0.5 - 0.75)

        diff_all_known = self.explainer.prediction_difference(instance, fixed_features=torch.tensor([0, 1]))[label]
        self.assertEqual(float(diff_all_known), 1.0 - 0.75)

    def test_feature_importance(self):
        instance, label = torch.tensor([[1, 0]]), 1

        first_importance = self.explainer.estimate_feature_importance(0, instance)[label]
        self.assertEqual(float(first_importance), 0.375)

        second_importance = self.explainer.estimate_feature_importance(1, instance)[label]
        self.assertEqual(float(second_importance), -0.125)

    def test_sentiment(self):
        # Same example as in examples/toy/sentiment.py, but we can only deterministically test exact Shapley values
        model = DummySentiment()
        labels = {"neg": 0, "pos": 1}
        feature_values = [
            [model.tok2id[t] for t in ["allegedly", "achingly", "amazingly", "astonishingly", "not", "very",
                                       "surprisingly"]],
            [model.tok2id[t] for t in ["good", "bad"]]
        ]
        query_sequence = "very good"

        exact_explainer = ExactShapleyExplainer(model_func=model.score, feature_values=feature_values)
        exact_res = exact_explainer.explain(model.to_internal([query_sequence])["input_ids"], label=labels["pos"])
        np.testing.assert_array_almost_equal(exact_res["importance"].tolist(), [0.16786, 0.33214], decimal=5)
