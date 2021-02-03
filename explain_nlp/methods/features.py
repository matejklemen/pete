from itertools import groupby
from typing import Tuple, Union, List

import stanza
import torch
from transformers import BertTokenizer

from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification


def extract_groups(feature_ids, ignore_index=-1):
    # e.g. [-1, 0, 0, 1, 0, 2] -> [[1, 2, 4], [3], [5]], assuming ignore_index=-1
    occurences = {}
    for i, el in enumerate(feature_ids):
        if el == ignore_index:
            continue

        curr_occurences = occurences.get(el, [])
        curr_occurences.append(i)
        occurences[el] = curr_occurences

    return list(occurences.values())


def stanza_word_features(raw_example: Union[str, Tuple[str, ...]], pipe: stanza.Pipeline,
                         do_depparse=False):
    """ Extract features for words inside `raw_example`. If a tuple is provided, examples are handled
    independently, but the word IDs do not reset. """
    global_word_id, global_sent_id = 0, 0
    eff_example = raw_example if isinstance(raw_example, tuple) else (raw_example,)
    ret_dict = {
        "words": [],
        "word_id_to_sent_id": {}
    }
    if do_depparse:
        ret_dict["word_id_to_head_id"] = {}
        ret_dict["word_id_to_deprel"] = {}
        ret_dict["word_id_to_depth"] = {}

    for curr_example in eff_example:
        doc = pipe(curr_example)  # type: stanza.Document

        for curr_sent in doc.sentences:
            for curr_word in curr_sent.words:
                ret_dict["words"].append(curr_word.text)
                ret_dict["word_id_to_sent_id"][global_word_id] = global_sent_id

                if do_depparse:
                    ret_dict["word_id_to_deprel"][global_word_id] = curr_word.deprel
                    # If word is a root, assign head word ID to itself
                    head_word_id = (global_word_id + (curr_word.head - curr_word.id)) if curr_word.head != 0 else global_word_id
                    ret_dict["word_id_to_head_id"][global_word_id] = head_word_id

                    # How far is the current word from ROOT in the dependency tree
                    head_id = curr_word.head
                    depth = 0
                    while head_id != 0:
                        head_id = curr_sent.words[head_id - 1].head  # head_id is 1-based as 0 = ROOT
                        depth += 1

                    ret_dict["word_id_to_depth"][global_word_id] = depth

                global_word_id += 1

            global_sent_id += 1

    return ret_dict


def depparse_custom_groups_1(head_ids, deprels):
    """ A very simple heuristic approach for grouping words into bigger units.
        Group some "supporting" words with their "main" words or group multi-word expressions."""
    num_words = len(head_ids)
    group_to_words = {i: [i] for i in range(num_words)}
    word_to_group = {i: i for i in range(num_words)}
    MERGED_DEPRELS = {"det", "det:predet", "aux", "flat", "flat:foreign", "flat:name", "compound"}

    for idx_word in range(num_words):
        word_group = word_to_group[idx_word]
        head_group = word_to_group[head_ids[idx_word]]
        if word_group == head_group:
            continue

        merge = deprels[idx_word] in MERGED_DEPRELS

        if merge:
            # Transfer units from current word's group to head's group
            for unit in group_to_words[word_group]:
                word_to_group[unit] = head_group
                group_to_words[head_group].append(unit)

            del group_to_words[word_group]

    return word_to_group


def depparse_custom_groups_2(head_ids, depths):
    """ Heuristic approach that groups together words in same subtree at depth >= 1. """
    num_words = len(head_ids)
    group_to_words = {i: [i] for i in range(num_words)}
    word_to_group = {i: i for i in range(num_words)}

    for idx_word in range(num_words):
        word_group = word_to_group[idx_word]
        head_group = word_to_group[head_ids[idx_word]]
        if word_group == head_group:
            continue

        merge = False

        if depths[head_ids[idx_word]] >= 1:
            merge = True

        if merge:
            # Transfer units from current word's group to head's group
            for unit in group_to_words[word_group]:
                word_to_group[unit] = head_group
                group_to_words[head_group].append(unit)

            del group_to_words[word_group]

    return word_to_group


if __name__ == "__main__":
    example = ("Unbelieveable, Jeff. What a goal!", "Peter Crouch scores and it's one nil for Stoke City.")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    do_depparse = True
    USED_PROCESSORS = "tokenize{}".format(",pos,lemma,depparse" if do_depparse else "")
    nlp = stanza.Pipeline(lang="en", processors=USED_PROCESSORS)

    model = InterpretableBertForSequenceClassification(
        model_name="/home/matej/Documents/embeddia/interpretability/ime-lm/resources/weights/snli_bert_uncased",
        tokenizer_name="/home/matej/Documents/embeddia/interpretability/ime-lm/resources/weights/snli_bert_uncased",
        batch_size=2,
        max_seq_len=41,
        device="cpu"
    )

    pretokenized_example = [
        (
            [word.text for sent in nlp(example[0]).sentences for word in sent.words],
            [word.text for sent in nlp(example[1]).sentences for word in sent.words]
        )
    ]
    encoded_input = model.to_internal(pretokenized_text_data=pretokenized_example)
    input_tokens = model.convert_ids_to_tokens(encoded_input["input_ids"])[0]
    res = stanza_word_features(example, pipe=nlp, do_depparse=do_depparse)

    word_ids = encoded_input["aux_data"]["alignment_ids"][0].tolist()
    custom_ids = [res["word_id_to_sent_id"].get(curr_word_id, -1) for curr_word_id in word_ids]

    if do_depparse:
        # Custom heuristics
        # word_to_custom_group = depparse_custom_groups_1(res["word_id_to_head_id"], res["word_id_to_deprel"])

        # Merge subtrees with root depth >= 1 together
        word_to_custom_group = depparse_custom_groups_2(res["word_id_to_head_id"], res["word_id_to_depth"])
        custom_ids = [word_to_custom_group.get(curr_word_id, -1) for curr_word_id in word_ids]

    custom_features = extract_groups(custom_ids)
    for curr_group in custom_features:
        print([input_tokens[_i] for _i in curr_group])
