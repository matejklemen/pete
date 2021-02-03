import unittest

from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification


class TestModeling(unittest.TestCase):
    def setUp(self):
        self.model = InterpretableBertForSequenceClassification(tokenizer_name="bert-base-uncased",
                                                                model_name="bert-base-uncased",
                                                                batch_size=4,
                                                                max_seq_len=16,
                                                                max_words=10,
                                                                device="cpu")

    def test_bert_word_alignment(self):
        # Single sequence
        pretokenized_example = [["Unbelieveable", ",", "Jeff"]]
        res = self.model.to_internal(pretokenized_text_data=pretokenized_example)

        self.assertListEqual(res["aux_data"]["alignment_ids"][0].tolist(),
                             # ["[BOS]", "un", "##bel", "##ie", "##ve" "##able", "," , "Jeff", "[SEP]"]
                             [-1, 0, 0, 0, 0, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1])

        # Sequence pair
        pretokenized_example = [
            (["Does", "n't", "she", "study", "biology", "?"], ["No", ",", "she", "studies", "bionanotechnology"])
        ]
        res = self.model.to_internal(pretokenized_text_data=pretokenized_example)
        self.assertListEqual(res["aux_data"]["alignment_ids"][0].tolist(),
                             # ["[CLS]", "Does", "n", "'", "t", "she", "study", "biology", "[SEP]", "No", ",", "she", "studies", "bio", "##nan", "[SEP]"]
                             [-1, 0, 1, 1, 1, 2, 3, 4, -1, 6, 7, 8, 9, 10, 10, -1])

        # Batch
        pretokenized_examples = [
            ["A", "straight", "line", "may", "be", "the", "shortest", "distance", "between", "two", "points", ",",
             "but", "it", "is", "by", "no", "means", "the", "most", "interesting", "."],
            ["Memories", "can", "be", "vile", ".", "Repulsive", "little", "brutes", ",", "like", "children", "I",
             "suppose", ".", "But", "can" "we", "live", "without", "them", "?"]
        ]
        res = self.model.to_internal(pretokenized_text_data=pretokenized_examples)
        self.assertListEqual(res["aux_data"]["alignment_ids"].tolist(),
                             [
                                 [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1],
                                 [-1, 0, 1, 2, 3, 4, 5, 5, 6, 7, 7, 8, 9, 10, 11, -1]
                             ])

    def test_bert_word_tokenization(self):
        """Test tokenization at word level, i.e. special tokens are given for each word, and words are tokenized into
        groups of subwords"""
        # Single sequence
        pretokenized_example = [["Unbelieveable", ",", "Jeff"]]
        res = self.model.words_to_internal(pretokenized_example)

        self.assertListEqual(res["input_ids"],
                             [[[101], [4895, 8671, 2666, 3726, 3085], [1010], [5076], [102], [0], [0], [0], [0], [0]]])
        self.assertListEqual(res["aux_data"]["attention_mask"].tolist(),
                             [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])

        # Sequence pair
        pretokenized_example = [
            (["Does", "n't", "she", "study", "biology", "?"], ["No", ",", "she", "studies", "bionanotechnology"])
        ]
        res = self.model.words_to_internal(pretokenized_example)
        self.assertListEqual(res["input_ids"],
                             [[[101], [2515], [1050, 1005, 1056], [2016], [2817], [102], [2053], [1010], [2016], [102]]])
        self.assertListEqual(res["aux_data"]["attention_mask"].tolist(),
                             [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        # Batch
        pretokenized_examples = [
            ["A", "straight", "line", "may", "be", "the", "shortest", "distance", "between", "two", "points", ",",
             "but", "it", "is", "by", "no", "means", "the", "most", "interesting", "."],
            ["Memories", "can", "be", "vile", ".", "Repulsive", "little", "brutes", ",", "like", "children", "I",
             "suppose", ".", "But", "can" "we", "live", "without", "them", "?"]
        ]
        res = self.model.words_to_internal(pretokenized_examples)
        # Lists of `max_words` WORDS, but the words may be composed of multiple subwords
        self.assertListEqual(res["input_ids"],
                             [
                                 [[101], [1037], [3442], [2240], [2089], [2022], [1996], [20047], [3292], [102]],
                                 [[101], [5758], [2064], [2022], [25047], [1012], [16360, 23004], [2210], [26128, 2015], [102]]
                             ])
        self.assertListEqual(res["words"],
                             [
                                 ['[CLS]', 'A', 'straight', 'line', 'may', 'be', 'the', 'shortest', 'distance', '[SEP]'],
                                 ['[CLS]', 'Memories', 'can', 'be', 'vile', '.', 'Repulsive', 'little', 'brutes', '[SEP]']
                             ])
