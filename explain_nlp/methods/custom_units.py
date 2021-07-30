from copy import deepcopy
from typing import Union, Tuple, Dict, List

import stanza

from explain_nlp.methods.features import stanza_word_features, extract_groups
from explain_nlp.methods.ime import IMEExplainer
from explain_nlp.methods.lime import LIMEExplainer
from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification
from explain_nlp.visualizations.highlight import highlight_plot


def criterion_abs_sum(new_importance, previous_importances):
    """ Returns True if absolute new importance is bigger than absolute sum of previous importances."""
    return abs(new_importance) > abs(sum(previous_importances))


class StanzaExplainer:
    stanza_pipe: stanza.Pipeline

    def word_tokenize(self, text_data: Union[str, Tuple[str, ...]]):
        is_pair = isinstance(text_data, tuple)

        if is_pair:
            pretokenized_example = (
                [word.text for sent in self.stanza_pipe(text_data[0]).sentences for word in sent.words],
                [word.text for sent in self.stanza_pipe(text_data[1]).sentences for word in sent.words]
            )
        else:
            pretokenized_example = [word.text for sent in self.stanza_pipe(text_data).sentences
                                    for word in sent.words]

        return pretokenized_example


class WordExplainer(StanzaExplainer):
    def __init__(self, explainer: Union[LIMEExplainer, IMEExplainer], stanza_pipeline: stanza.Pipeline,
                 return_steps=False):
        self.explainer = explainer
        self.stanza_pipe = stanza_pipeline
        self.return_steps = return_steps

    def explain_text(self, text_data: Union[str, Tuple[str, ...]],
                     **explainer_kwargs) -> Union[Dict, List]:
        assert "custom_features" not in explainer_kwargs
        pretokenized_example = self.word_tokenize(text_data)

        encoded_input = self.explainer.model.to_internal([pretokenized_example], is_split_into_units=True)
        word_ids = encoded_input["aux_data"]["alignment_ids"][0]
        feature_groups = extract_groups(word_ids)

        res = self.explainer.explain_text(text_data,
                                          pretokenized_text_data=pretokenized_example,
                                          custom_features=feature_groups,
                                          **explainer_kwargs)

        return [res] if self.return_steps else res


class SentenceExplainer(StanzaExplainer):
    def __init__(self, explainer: Union[LIMEExplainer, IMEExplainer], stanza_pipeline: stanza.Pipeline,
                 return_steps=False):
        self.explainer = explainer
        self.stanza_pipe = stanza_pipeline
        self.return_steps = return_steps

    def explain_text(self, text_data: Union[str, Tuple[str, ...]],
                     **explainer_kwargs) -> Union[Dict, List]:
        assert "custom_features" not in explainer_kwargs
        pretokenized_example = self.word_tokenize(text_data)

        encoded_input = self.explainer.model.to_internal([pretokenized_example], is_split_into_units=True)
        features = stanza_word_features(text_data, pipe=self.stanza_pipe, do_depparse=False)

        word_ids = encoded_input["aux_data"]["alignment_ids"][0]
        sent_ids = list(map(lambda curr_word_id: features["word_id_to_sent_id"].get(curr_word_id, -1), word_ids))
        feature_groups = extract_groups(sent_ids)

        res = self.explainer.explain_text(text_data,
                                          pretokenized_text_data=pretokenized_example,
                                          custom_features=feature_groups,
                                          **explainer_kwargs)

        return [res] if self.return_steps else res


class DependencyTreeExplainer(StanzaExplainer):
    def __init__(self, explainer: Union[LIMEExplainer, IMEExplainer], stanza_pipeline: stanza.Pipeline,
                 allow_sentence_explanations=False, return_steps=False):
        """ Explain a sequence with units determined by the dependency structure.

        Args:
            explainer:
                Used explanation method
            stanza_pipeline:
                A stanza.Pipeline object, initialized with the "dependency_parsing" processor
            allow_sentence_explanations:
                Whether to allow unit grouping to happen at depth 0, i.e. if the whole sentence can be treated as a
                single unit
            return_steps:
                Whether to return ALL steps of optimization instead of just the final result
        """
        self.explainer = explainer
        self.stanza_pipe = stanza_pipeline
        self.return_steps = return_steps
        self.min_considered_depth = 0 if allow_sentence_explanations else 1

        assert "depparse" in self.stanza_pipe.processors

    def truncate_text(self, text_data: Union[str, Tuple[str, ...]]):
        """ Truncates `text_data` so that when it is tokenized using the explained model's tokenizer, it's shorter than
        its max allowed length.

        Note that this is only required for DependencyTreeExplainer because in simpler explainers (word, sentence, ...)
        we can simply ignore the overflowing tokens. Here, this could lead to an improper dependency tree
        """
        pretokenized_example = self.word_tokenize(text_data)

        # Convert to internal (e.g. subword) representation, taking into account word boundaries
        encoded_input = self.explainer.model.to_internal([pretokenized_example], is_split_into_units=True)

        # Convert (possibly truncated) IDs back to str/tuple
        decoded_input = self.explainer.model.from_internal(encoded_input["input_ids"],
                                                           **encoded_input["aux_data"])

        return decoded_input[0]

    def explain_text(self, text_data: Union[str, Tuple[str, ...]],
                     **explainer_kwargs) -> Union[Dict, List]:
        assert "custom_features" not in explainer_kwargs

        truncated_text_data = self.truncate_text(text_data)
        pretokenized_example = self.word_tokenize(truncated_text_data)

        encoded_input = self.explainer.model.to_internal([pretokenized_example], is_split_into_units=True)
        features = stanza_word_features(truncated_text_data, pipe=self.stanza_pipe, do_depparse=True)
        # Map root of each subtree to its children
        parent_to_children = {}
        for id_word, id_head in features["word_id_to_head_id"].items():
            existing_ch = parent_to_children.get(id_head, [])
            existing_ch.append(id_word)
            parent_to_children[id_head] = existing_ch

        # Sort subtrees (roots) by (1) depth and (2) position in word, both ascending, i.e.
        #  we consider options by decreasing depth in left-to-right order
        merge_options = sorted([(parent, features["word_id_to_depth"][parent])
                                for parent in parent_to_children
                                if features["word_id_to_depth"][parent] >= self.min_considered_depth],
                               key=lambda tup: (-tup[1], tup[0]))

        word_ids = encoded_input["aux_data"]["alignment_ids"][0]
        feature_groups = extract_groups(word_ids)
        res = self.explainer.explain_text(truncated_text_data,
                                          pretokenized_text_data=pretokenized_example,
                                          custom_features=feature_groups,
                                          **explainer_kwargs)
        importances = res["importance"][-len(feature_groups):]
        group_to_importance = dict(zip(map(lambda group: tuple(group), res["custom_features"]), importances.tolist()))

        steps = [res]
        is_merged = {}
        for (curr_parent, curr_depth) in merge_options:
            new_feature_groups = deepcopy(feature_groups)
            children_importances = [group_to_importance[tuple(feature_groups[curr_parent])]]

            # Only attempt a merge if all children are merged
            attempt_merge = True
            for child in parent_to_children[curr_parent]:
                is_merged[child] = is_merged.get(child, True)
                attempt_merge &= is_merged[child]

                if is_merged[child]:
                    new_feature_groups[curr_parent] += feature_groups[child]
                    children_importances.append(group_to_importance[tuple(feature_groups[child])])
                    new_feature_groups[child] = []

            if not attempt_merge:
                is_merged[curr_parent] = False
                continue

            # Remove empty groups corresponding to children that were merged
            #  (these are left in original groups so that indexing works)
            clean_feature_groups = list(filter(lambda group: len(group) > 0, new_feature_groups))

            res = self.explainer.explain_text(truncated_text_data,
                                              pretokenized_text_data=pretokenized_example,
                                              custom_features=clean_feature_groups,
                                              **explainer_kwargs)
            new_importances = res["importance"][-len(clean_feature_groups):]
            new_group_to_importance = dict(zip(map(lambda group: tuple(group), res["custom_features"]),
                                               new_importances.tolist()))
            parent_importance = new_group_to_importance[tuple(new_feature_groups[curr_parent])]

            perform_merge = criterion_abs_sum(new_importance=parent_importance,
                                              previous_importances=children_importances)

            is_merged[curr_parent] = perform_merge
            if perform_merge:
                steps.append(res)
                group_to_importance = new_group_to_importance
                feature_groups = new_feature_groups

        return steps if self.return_steps else steps[-1]


if __name__ == "__main__":
    example = ("A shirtless man skateboards on a ledge.", "A man without a shirt.")
    EXPLAINED_LABEL = 0

    nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse")
    model = InterpretableBertForSequenceClassification(
        model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/classifiers/snli_bert_uncased",
        tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/classifiers/snli_bert_uncased",
        batch_size=2,
        max_seq_len=41,
        device="cpu"
    )
    lime_explainer = LIMEExplainer(model, return_samples=True, return_scores=True)
    explainer = DependencyTreeExplainer(explainer=lime_explainer, stanza_pipeline=nlp,
                                        return_steps=True)
    res = explainer.explain_text(text_data=example,
                                 label=EXPLAINED_LABEL,
                                 num_samples=100)

    if explainer.return_steps:
        highlight_plot(list(map(lambda step: step["input"], res)),
                       importances=list(map(lambda step: step["importance"].tolist(), res)),
                       pred_labels=["entailment"] * len(res),
                       actual_labels=["entailment"] * len(res),
                       path="tmp_custom_units.html",
                       custom_features=list(map(lambda step: step["custom_features"], res)))
    else:
        highlight_plot([res["input"]],
                       importances=[res["importance"].tolist()],
                       pred_labels=["entailment"],
                       actual_labels=["entailment"],
                       path="tmp_custom_units.html",
                       custom_features=[res["custom_features"]])
