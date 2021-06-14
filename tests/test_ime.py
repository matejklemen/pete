import unittest
import torch
from explain_nlp.methods.ime import IMEExplainer
from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification


class TestIMEExplainer(unittest.TestCase):
    def setUp(self):
        self.model = InterpretableBertForSequenceClassification(
            tokenizer_name="lysandre/tiny-bert-random",
            model_name="lysandre/tiny-bert-random",
            device="cpu",
            max_seq_len=8
        )
        self.sample_data = self.model.to_internal([
            "The quick brown fox jumps over the lazy dog",
            "Short sequence",
            "My very photogenic mother died in a freak accident (picnic, lightning) when I was three, and, save for a pocket of warmth in the darkest past, nothing of her subsists within the hollows and dells of memory, over which, if you can still stand my style (I am writing under observation), the sun of my infancy had set: surely, you all know those redolent remnants of day suspended, with the midges, about some hedge in bloom or suddenly entered and traversed by the rambler, at the bottom of a hill, in the summer dusk; a furry warmth, golden midges."
        ])
        self.sample_input = self.model.to_internal(["I am John"])

    def test_sample_constraint_handling(self):
        """ Test different ways of providing sample constraints for IME explanations. """

        # [Providing min_samples_per_feature]
        samples_per_feature, total_samples = IMEExplainer._handle_sample_constraints(
            num_features=5, num_additional=0, used_feature_indices=[1, 2, 3],
            min_samples_per_feature=10
        )
        self.assertListEqual(samples_per_feature.tolist(), [0, 10, 10, 10, 0])
        self.assertEqual(total_samples, 30)

        samples_per_feature, total_samples = IMEExplainer._handle_sample_constraints(
            num_features=5, num_additional=3, used_feature_indices=[5, 7],
            min_samples_per_feature=3
        )
        self.assertListEqual(samples_per_feature.tolist(), [0, 0, 0, 0, 0, 3, 0, 3])
        self.assertEqual(total_samples, 6)

        # Taking less than 2 min samples per feature results in undefined variance, so it should raise an error
        self.assertRaises(AssertionError,
                          lambda: IMEExplainer._handle_sample_constraints(
                              num_features=5, num_additional=3, used_feature_indices=[5, 7],
                              min_samples_per_feature=0
                          ))

        # [Providing min_samples_per_feature, max_samples]
        samples_per_feature, total_samples = IMEExplainer._handle_sample_constraints(
            num_features=5, num_additional=0, used_feature_indices=[1, 2, 3],
            min_samples_per_feature=10, max_samples=37
        )
        self.assertListEqual(samples_per_feature.tolist(), [0, 10, 10, 10, 0])
        self.assertEqual(total_samples, 37)

        # Taking less max_samples than min_samples_per_feature for each feature should raise an error
        self.assertRaises(AssertionError,
                          lambda: IMEExplainer._handle_sample_constraints(
                              num_features=5, num_additional=0, used_feature_indices=[1, 2, 3],
                              min_samples_per_feature=10, max_samples=24
                          ))

        # [Providing exact_samples_per_feature]
        samples_per_feature, total_samples = IMEExplainer._handle_sample_constraints(
            num_features=5, num_additional=0, used_feature_indices=[1, 2, 3],
            exact_samples_per_feature=torch.tensor([[47, 5, 3, 1, 0]])
        )
        # Note that samples allocated to unused features are quietly ignored
        self.assertListEqual(samples_per_feature.tolist(), [0, 5, 3, 1, 0])
        self.assertEqual(total_samples, 9)

        # Requesting < 2 samples per feature when providing exact_samples_per_feature just raises a warning, as
        # this option assumes the user knows what they are providing as input
        with self.assertWarns(UserWarning):
            samples_per_feature, total_samples = IMEExplainer._handle_sample_constraints(
                num_features=5, num_additional=0, used_feature_indices=[1, 2, 3],
                exact_samples_per_feature=torch.tensor([[0, 1, 0, 1, 0]])
            )
            self.assertListEqual(samples_per_feature.tolist(), [0, 1, 0, 1, 0])
            self.assertEqual(total_samples, 2)

        # exact_samples_per_feature has to provide number of samples taken for each feature, even if it is zero
        self.assertRaises(AssertionError,
                          lambda: IMEExplainer._handle_sample_constraints(
                              num_features=5, num_additional=0, used_feature_indices=[1, 2, 3],
                              exact_samples_per_feature=torch.tensor([[0, 1, 0, 1]])
                          ))

        # [Providing min_samples_per_feature, max_samples, exact_samples_per_feature]
        # In case all three options are provided, exact_samples_per_feature quietly takes precedence
        samples_per_feature, total_samples = IMEExplainer._handle_sample_constraints(
            num_features=5, num_additional=0, used_feature_indices=[1, 2, 3],
            min_samples_per_feature=10, max_samples=37,
            exact_samples_per_feature=torch.tensor([[0, 3, 5, 5, 0]])
        )
        self.assertListEqual(samples_per_feature.tolist(), [0, 3, 5, 5, 0])
        self.assertEqual(total_samples, 13)

        # [general wrong inputs]
        # None of the possible constraint arguments are provided = can't allocate samples
        self.assertRaises(AssertionError,
                          lambda: IMEExplainer._handle_sample_constraints(
                              num_features=5, num_additional=0, used_feature_indices=[1, 2, 3]
                          ))

        # providing empty list of used feature indices implies that no feature is being estimated
        self.assertRaises(AssertionError,
                          lambda: IMEExplainer._handle_sample_constraints(
                              num_features=5, num_additional=0, used_feature_indices=[],
                              min_samples_per_feature=10
                          ))

    def test_unperturbable_feature(self):
        """ Test that features marked as unperturbable do not take up samples """
        explainer = IMEExplainer(sample_data=self.sample_data["input_ids"], model=self.model,
                                 return_scores=True, return_num_samples=True, return_samples=True)
        res = explainer.explain(instance=self.sample_input["input_ids"],
                                perturbable_mask=torch.tensor([[False, True, True, False, False, False, False, False]]),
                                min_samples_per_feature=10,
                                **self.sample_input["aux_data"])

        self.assertEqual(res["num_samples"][0], 0)  # 0 samples taken for unperturbable feature
        self.assertIs(res["samples"][0], None)
        self.assertIs(res["scores"][0], None)
        self.assertEqual(res["num_samples"][1], 10)

    def test_estimate_feature_importance_no_feature_groups(self):
        # Should pass without crashing: make sure feature groups get correctly constructed when not explicitly provided
        explainer = IMEExplainer(sample_data=self.sample_data["input_ids"], model=self.model,
                                 return_scores=True, return_num_samples=True, return_samples=True)

        explainer.estimate_feature_importance(idx_feature=1,
                                              instance=self.sample_input["input_ids"],
                                              num_samples=10,
                                              perturbable_mask=self.sample_input["perturbable_mask"],
                                              feature_groups=None,
                                              **self.sample_input["aux_data"])

    def test_exceptions_on_invalid_input(self):
        # Specified a minimum of 10 samples per features, but specified only 20 max samples in total
        # (10 samples * 3 features = 30 samples, which is lower than 20 max samples)
        explainer2 = IMEExplainer(sample_data=self.sample_data["input_ids"], model=self.model)
        self.assertRaises(AssertionError,
                          lambda: explainer2.explain(instance=self.sample_input["input_ids"],
                                                     min_samples_per_feature=10,
                                                     max_samples=20))
