import os
import shutil
import unittest

from explain_nlp.visualizations.highlight import highlight_plot


class TestVisualization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_dir = os.path.sep.join(__file__.split(os.path.sep)[:-1])
        cls.temporary_cache_dir = os.path.join(test_dir, ".test_data")

        if os.path.exists(cls.temporary_cache_dir):
            shutil.rmtree(cls.temporary_cache_dir)

        if not os.path.exists(cls.temporary_cache_dir):
            os.makedirs(cls.temporary_cache_dir)

    def test_primary_units(self):
        highlight_plot(sequences=[["[BOS]", "This", "movie", "is", "very", "good", "."]],
                       importances=[[0.0, -0.05, 0.1, 0.01, 0.1, 0.3, -0.02]],
                       pred_labels=["clean"], actual_labels=["clean"],
                       custom_features=None,
                       path=os.path.join(self.temporary_cache_dir, "test_primary_units.html"))

    def test_bigger_units(self):
        # contiguous units
        highlight_plot(sequences=[["This", "movie", "is", "very", "good", "."]],
                       importances=[[0, 0, 0, 0, 0, 0,
                                     0.15, -0.02, 0.4, -0.03]],
                       pred_labels=["clean"], actual_labels=["clean"],
                       custom_features=[[[0, 1], [2], [3, 4], [5]]],
                       path=os.path.join(self.temporary_cache_dir, "test_bigger_units1.html"))

        # discontiguous units
        highlight_plot(sequences=[["This", "movie", "is", "very", "good", "."]],
                       importances=[[0, 0, 0, 0, 0, 0,
                                     0.15, -0.02, 0.4, -0.03]],
                       pred_labels=["clean"], actual_labels=["clean"],
                       custom_features=[[[0, 1, 4], [2], [3], [5]]],
                       path=os.path.join(self.temporary_cache_dir, "test_bigger_units2.html"))

        # large units that might break the visualization
        highlight_plot(sequences=[["Lorem", "ipsum", "dolor", "sit", "amet", ",", "consectetur", "adipiscing", "elit", ".",  # 10
                                   "Aliquam", "at", "laoreet", "elit", ".",  # 5
                                   "Sed", "dignissim", "aliquet", "velit", ",", "at", "placerat", "lacus", "volutpat", "ut", ".",  # 11
                                   "Vivamus", "blandit", "convallis", "nibh", "vel", "sollicitudin", ".",  # 7
                                   "Morbi", "at", "lectus", "facilisis", ",", "fermentum", "odio", "quis", ",", "laoreet", "velit", ".",  # 12
                                   "Praesent", "ut", "tempus", "ante", ",", "vitae", "interdum", "velit", ".",  # 9
                                   "Maecenas", "ornare", "neque", "tortor", ",", "eget", "faucibus", "quam", "suscipit", "vitae", "."]],  # 11
                       importances=[[0] * 65 + [0.15, -0.2, 0.1, -0.03, -0.4, 0.2, 0.1]],
                       pred_labels=["clean"], actual_labels=[None],
                       custom_features=[[list(range(0, 10)),
                                         list(range(10, 10 + 5)),
                                         list(range(15, 15 + 11)),
                                         list(range(26, 26 + 7)),
                                         list(range(33, 33 + 12)),
                                         list(range(45, 45 + 9)),
                                         list(range(54, 54 + 11))]],
                       path=os.path.join(self.temporary_cache_dir, "test_bigger_units3.html"))
