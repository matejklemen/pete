from explain_nlp.methods.features import stanza_word_features, depparse_custom_groups_1, depparse_custom_groups_2, \
    extract_groups


def handle_features(custom_features: str = None,
                    **kwargs):
    """ Function for extraction of bigger word units for a single example.

    Args:
    -----
    custom_features: str
        Type of custom features to use (str) or `None` if you don't want to use custom features.
    word_ids: List[int]
        ID of word that the primary unit (e.g. subword) belongs to. These should align with words of `raw_example`,
        should be nonnegative and should NOT reset if `raw_example` contains more than one string
    raw_example: Union[str, Tuple[str, ...]]
        The string form of example from which features are to be extracted and aligned for use in model using `word_ids`
    pipe: stanza.Pipeline
        The model to use for extraction of features. Make sure to initialize it with the correct processors!
    """
    if custom_features is None:
        return None

    # Word IDs for each primary unit (e.g. subword) are required as they serve as an anchor to align features, obtained
    # on word level, with features used in model
    word_ids = kwargs["word_ids"]
    if custom_features == "words":
        feature_ids = word_ids
    elif custom_features == "sentences":
        res = stanza_word_features(
            raw_example=kwargs["raw_example"],
            pipe=kwargs["pipe"]
        )
        feature_ids = [res["word_id_to_sent_id"].get(curr_word_id, -1) for curr_word_id in word_ids]
    elif custom_features.startswith("depparse"):
        res = stanza_word_features(
            raw_example=kwargs["raw_example"],
            pipe=kwargs["pipe"],
            do_depparse=True
        )

        if custom_features == "depparse_simple":
            custom_groups = depparse_custom_groups_1(res["word_id_to_head_id"], res["word_id_to_deprel"])
        elif custom_features == "depparse_depth":
            custom_groups = depparse_custom_groups_2(res["word_id_to_head_id"], res["word_id_to_depth"])
        else:
            raise ValueError(f"Unrecognized option for custom_features: '{custom_features}'")

        feature_ids = [custom_groups.get(curr_word_id, -1) for curr_word_id in word_ids]
    else:
        raise NotImplementedError

    return extract_groups(feature_ids, ignore_index=-1)
