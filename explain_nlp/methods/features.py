from itertools import groupby
from typing import Tuple, Union, List

import stanza
import torch
from transformers import BertTokenizer


def stanza_bert_words(input_tokens: List[str], perturbable_mask: torch.Tensor,
                      raw_example: Union[str, Tuple[str, ...]],
                      pipe: stanza.Pipeline, lowercase=False):
    # print(input_tokens)
    flattened_tokens = []
    sentence_ids, curr_sentence_id = [], 0
    _raw_example = raw_example if isinstance(raw_example, tuple) else (raw_example,)
    for curr_seq in _raw_example:
        curr_doc = pipe(curr_seq)  # type: stanza.Document
        for curr_sent in curr_doc.sentences:
            for curr_token in curr_sent.tokens:
                flattened_tokens.append(curr_token.words[0].text.lower() if lowercase else curr_token.words[0].text)
                sentence_ids.append(curr_sentence_id)
            curr_sentence_id += 1

    subword_to_word = []
    subword_to_sent = []
    cursor_words = 0
    cursor_char = 0
    for i, (curr_unit, is_perturbable) in enumerate(zip(input_tokens, perturbable_mask)):
        if not is_perturbable:
            subword_to_word.append(-1)
            subword_to_sent.append(-1)
            continue

        curr_unit_norm = curr_unit[2:] if curr_unit.startswith("##") else curr_unit
        # Some tokens present in original sequence might have been truncated, skip those
        while cursor_words < len(flattened_tokens) and \
                flattened_tokens[cursor_words].find(curr_unit_norm, cursor_char) == -1:
            cursor_words += 1
            cursor_char = 0

        if cursor_words == len(flattened_tokens):
            raise ValueError(f"Could not find unit '{curr_unit_norm}' among tokens {flattened_tokens}")

        subword_position = flattened_tokens[cursor_words].find(curr_unit_norm, cursor_char)
        subword_to_word.append(cursor_words)
        subword_to_sent.append(sentence_ids[cursor_words])

        if subword_position + len(curr_unit_norm) == len(flattened_tokens[cursor_words]):
            cursor_words += 1
            cursor_char = 0
        else:
            cursor_char += len(curr_unit_norm)


    return {
        "word_ids": subword_to_word,
        "sentence_ids": subword_to_sent
    }


if __name__ == "__main__":
    example = ("Stretching high in the air, the player in the black shirt reaches for the football as the other stretches toward him to steal it away.",
               "A wide reciever is trying to make a catch.")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    do_lower = True

    if do_lower:
        example = (example[0].lower(), example[1].lower())

    bert_encoded = tokenizer.encode_plus(*example, return_special_tokens_mask=True, max_length=41, padding="max_length")
    input_tokens = tokenizer.convert_ids_to_tokens(bert_encoded["input_ids"])
    print(input_tokens)

    nlp = stanza.Pipeline(lang="en", processors="tokenize")
    # word_ids, sentence_ids
    res = stanza_bert_words(input_tokens, [not is_special for is_special in bert_encoded["special_tokens_mask"]],
                                     example, nlp, lowercase=do_lower)

    word_features = []
    for idx_word, word_units in groupby(enumerate(res["sentence_ids"]), lambda index_item: index_item[1]):
        if idx_word == -1:  # special tokens (unperturbable) -> don't include
            continue

        word_features.append(list(map(lambda tup: tup[0], word_units)))

    for curr_word in word_features:
        print([input_tokens[_i] for _i in curr_word])
