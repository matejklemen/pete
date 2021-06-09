import unittest

from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification


class TestModeling(unittest.TestCase):
    def setUp(self):
        self.model = InterpretableBertForSequenceClassification(tokenizer_name="bert-base-uncased",
                                                                model_name="bert-base-uncased",
                                                                batch_size=4,
                                                                max_seq_len=16,
                                                                device="cpu")

    def test_bert_word_alignment(self):
        # Single sequence
        pretokenized_example = [["Unbelieveable", ",", "Jeff"]]
        res = self.model.to_internal(pretokenized_example, is_split_into_units=True)

        self.assertListEqual(res["aux_data"]["alignment_ids"][0],
                             # ["[BOS]", "un", "##bel", "##ie", "##ve" "##able", "," , "Jeff", "[SEP]"]
                             [-1, 0, 0, 0, 0, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1])

        # Sequence pair
        pretokenized_example = [
            (["Does", "n't", "she", "study", "biology", "?"], ["No", ",", "she", "studies", "bionanotechnology"])
        ]
        res = self.model.to_internal(pretokenized_example, is_split_into_units=True)
        self.assertListEqual(res["aux_data"]["alignment_ids"][0],
                             # ["[CLS]", "Does", "n", "'", "t", "she", "study", "[SEP]", "No", ",", "she", "studies", "bio", "##nan", "##ote", "[SEP]"]
                             [-1, 0, 1, 1, 1, 2, 3, -1, 6, 7, 8, 9, 10, 10, 10, -1])

        # Batch
        pretokenized_examples = [
            ["A", "straight", "line", "may", "be", "the", "shortest", "distance", "between", "two", "points", ",",
             "but", "it", "is", "by", "no", "means", "the", "most", "interesting", "."],
            ["Memories", "can", "be", "vile", ".", "Repulsive", "little", "brutes", ",", "like", "children", "I",
             "suppose", ".", "But", "can" "we", "live", "without", "them", "?"]
        ]
        res = self.model.to_internal(pretokenized_examples, is_split_into_units=True)
        self.assertListEqual(res["aux_data"]["alignment_ids"],
                             [
                                 [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1],
                                 [-1, 0, 1, 2, 3, 4, 5, 5, 6, 7, 7, 8, 9, 10, 11, -1]
                             ])

    def test_skip_special_with_return_tokens(self):
        """ If return_tokens=True and skip_special_tokens=True, special tokens should be skipped instead of returning
        empty string"""
        seq = "unflinchingly bleak and desperate "

        encoded = self.model.to_internal([seq])
        decoded = self.model.from_internal(encoded["input_ids"], skip_special_tokens=True, return_tokens=True,
                                           **encoded["aux_data"])[0]

        self.assertNotEqual(decoded[-1], "")  # Make sure [SEP] did not get decoded into "", but that it is skipped

        decoded_ref = self.model.from_internal(encoded["input_ids"], skip_special_tokens=False, return_tokens=True,
                                               **encoded["aux_data"])[0]
        self.assertEqual(decoded_ref[-1], "[SEP]")
