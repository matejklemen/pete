import unittest
from explain_nlp.visualizations.highlight import highlight_plot


class TestHighlight(unittest.TestCase):
    def test_exceptions_on_invalid_input(self):
        # too many labels given
        self.assertRaises(ValueError,
                          lambda: highlight_plot(sequences=[["i", "am", "iron", "man"]],
                                                 pred_labels=[0, 1],
                                                 importances=[[0.1, 0.0, 0.2, 0.8]]))

        # too many importances: provided 1 sequence and importances for 2 sequences
        self.assertRaises(ValueError,
                          lambda: highlight_plot(sequences=[["i", "am", "iron", "man"]],
                                                 pred_labels=[0],
                                                 importances=[[0.1, 0.0, 0.2, 0.8], [0.5, 0.5, 0.5, 0.5]]))

        # too many importances v2: provided 4 elements in first sequence, but only 3 importances
        self.assertRaises(ValueError,
                          lambda: highlight_plot(sequences=[["i", "am", "iron", "man"]],
                                                 pred_labels=[0],
                                                 importances=[[0.1, 0.0, 0.2]]))

    def test_happy_flow(self):
        # examples of correct calls, these just need to pass without raising any exceptions
        highlight_plot(sequences=[["i", "am", "iron", "man"]],
                       pred_labels=["movie"],
                       importances=[[0.1, 0.0, 0.2, 0.8]])

        # E.g. regression with non-textual data: not the main purpose of this package, but I don't judge
        highlight_plot(sequences=[[0, 8, 2], [4, 3.5, 1]],
                       pred_labels=[14.37, 18.5],
                       importances=[[0.0, 0.5, 1.2], [-2.5, 3.7, 12.1]])
