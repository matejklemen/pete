import unittest
import torch
import numpy as np

from explain_nlp.methods.decoding import top_p_filtering, filter_unique, filter_allowed


class TestDecoding(unittest.TestCase):
    def setUp(self):
        # Predictions for a single position: shape [batch_size, vocab_size]
        self.probas = torch.tensor([
            [0.05, 0.1, 0.05, 0.03, 0.5, 0.27],
            [0.05, 0.27, 0.1, 0.05, 0.03, 0.5]
        ])
        self.logprobas = torch.log(self.probas)

    def test_topp_filtering(self):
        # top_p with p=1.0 = no-op
        filtered_logprobas = top_p_filtering(logits=self.logprobas.clone(), top_p=1.0)
        torch.testing.assert_allclose(torch.softmax(filtered_logprobas, dim=-1), self.probas)

        # top_p with p=0.0 = only keep the most likely option
        filtered_logprobas = top_p_filtering(logits=self.logprobas.clone(), top_p=0.0)
        filtered_probas = torch.softmax(filtered_logprobas, dim=-1)
        np.testing.assert_equal(torch.nonzero(filtered_probas, as_tuple=False).numpy(),
                                np.array([[0, 4], [1, 5]]))

        # top_p also includes (does not filter) the first token that goes beyond top_p
        filtered_logprobas = top_p_filtering(logits=self.logprobas.clone(), top_p=0.77)
        filtered_probas = torch.softmax(filtered_logprobas, dim=-1)
        np.testing.assert_equal(torch.nonzero(filtered_probas, as_tuple=False).numpy(),
                                np.array([[0, 1], [0, 4], [0, 5], [1, 1], [1, 2], [1, 5]]))

        filtered_logprobas = top_p_filtering(logits=self.logprobas.clone(), top_p=0.769)
        filtered_probas = torch.softmax(filtered_logprobas, dim=-1)
        np.testing.assert_equal(torch.nonzero(filtered_probas, as_tuple=False).numpy(),
                                np.array([[0, 4], [0, 5], [1, 1], [1, 5]]))

    def test_unique_filtering(self):
        filtered_logprobas = filter_unique(logits=self.logprobas.clone(),
                                           orig_values=torch.tensor([0, 5]))
        filtered_probas = torch.softmax(filtered_logprobas, dim=-1)
        # Make sure probability of token#1 (for first example) and token#5 (for second example) is 0.0
        torch.testing.assert_allclose(filtered_probas[[0, 1], [0, 5]], 0.0)
        np.testing.assert_equal(torch.nonzero(filtered_probas, as_tuple=False).numpy(),
                                np.array([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
                                          [1, 0], [1, 1], [1, 2], [1, 3], [1, 4]]))

    def test_allowed_filtering(self):
        # Make sure only tokens #0 and #3 are possible to sample
        filtered_logprobas = filter_allowed(logits=self.logprobas.clone(),
                                            allowed_values=torch.tensor([0, 3]))
        filtered_probas = torch.softmax(filtered_logprobas, dim=-1)
        np.testing.assert_equal(torch.nonzero(filtered_probas, as_tuple=False).numpy(),
                                np.array([[0, 0], [0, 3],
                                          [1, 0], [1, 3]]))

        # If all tokens are allowed, nothing should change
        filtered_logprobas = filter_allowed(logits=self.logprobas.clone(),
                                            allowed_values=torch.tensor([0, 1, 2, 3, 4, 5]))
        filtered_probas = torch.softmax(filtered_logprobas, dim=-1)
        torch.testing.assert_allclose(filtered_probas, self.probas)
