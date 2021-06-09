from typing import Tuple, Union, List, Optional

import stanza


def extract_groups(group_ids: List[int], ignore_index: Optional[int] = -1):
    """ Converts a flattened representation of groups (list) into an unflattened one (list of lists).

    Args:
        group_ids: list
            Group ID for each unit (e.g. word)
        ignore_index:
            Index that marks a group ID to be ignored

    Examples:
        >>> extract_groups([-1, 0, 0, 1, 0, 2]) # returns [[1, 2, 4], [3], [5]]
    """
    occurences = {}
    for i, el in enumerate(group_ids):
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
                    # Ignore ROOT
                    if curr_word.head != 0:
                        head_word_id = (global_word_id + (curr_word.head - curr_word.id))
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
