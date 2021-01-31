from typing import Optional, Union, List

import torch

from explain_nlp.methods.generation import BertForMaskedLMGenerator, BertForControlledMaskedLMGenerator, \
    GPTLMGenerator, GPTControlledLMGenerator
from explain_nlp.methods.ime import IMEExplainer
from explain_nlp.methods.modeling import InterpretableModel, InterpretableBertForSequenceClassification
from explain_nlp.methods.utils import sample_permutations

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class DependentIMEMaskedLMExplainer(IMEExplainer):
    def __init__(self, model: InterpretableModel, generator: BertForMaskedLMGenerator,
                 confidence_interval: Optional[float] = None,  max_abs_error: Optional[float] = None,
                 return_variance: Optional[bool] = False, return_num_samples: Optional[bool] = False,
                 return_samples: Optional[bool] = False, return_scores: Optional[bool] = False,
                 criterion: Optional[str] = "squared_error", is_aligned_vocabulary: Optional[bool] = False):
        dummy_sample_data = torch.randint(5, (1, 1), dtype=torch.long)
        super().__init__(sample_data=dummy_sample_data, model=model, confidence_interval=confidence_interval,
                         max_abs_error=max_abs_error, return_variance=return_variance,
                         return_num_samples=return_num_samples, return_samples=return_samples,
                         return_scores=return_scores, criterion=criterion)

        self.model = self.model  # type: InterpretableBertForSequenceClassification
        self.generator = generator

        self.is_aligned_vocabulary = is_aligned_vocabulary

    def _transform_masks(self, _instance_tokens, _masked_instance_tokens):
        # Returns mask (True/False)!
        is_pair = isinstance(_instance_tokens, tuple)

        eff_instance_tokens, eff_masked_tokens = _instance_tokens, _masked_instance_tokens
        if not is_pair:
            eff_instance_tokens = (_instance_tokens,)
            eff_masked_tokens = (_masked_instance_tokens,)

        _generator_instance = [[] for _ in range(len(eff_instance_tokens))]
        for i, (all_orig_tok, all_mask_tok) in enumerate(zip(eff_instance_tokens, eff_masked_tokens)):
            for orig, masked in zip(all_orig_tok, all_mask_tok):
                transformed_tok = self.generator.tokenizer.tokenize(orig)

                # TODO: could probably do this on IDs and only operate on strings when really needed
                if masked == self.model.mask_token:
                    _generator_instance[i].extend([self.generator.mask_token] * len(transformed_tok))
                else:
                    _generator_instance[i].append(orig)

        if is_pair:
            _generator_instance = tuple([" ".join(curr_tokens) for curr_tokens in _generator_instance])
        else:
            _generator_instance = " ".join(_generator_instance[0])

        _generator_instance = self.generator.to_internal([_generator_instance])
        return _generator_instance["input_ids"][0] == self.generator.mask_token_id

    @torch.no_grad()
    def estimate_feature_importance(self, idx_feature: Union[int, List[int]], instance: torch.Tensor, num_samples: int,
                                    perturbable_mask: torch.Tensor, label: Optional[str] = None,
                                    feature_groups: Optional[List[List[int]]] = None, **modeling_kwargs):
        # Note: instance is currently supposed to be of shape [1, num_features]
        num_features = int(len(instance[0]))

        # TODO: this is temporary
        # if isinstance(idx_feature, int):
        #     print(f"Estimating importance of '{self.model.tokenizer.decode([instance[0, idx_feature]])}'")
        # else:
        #     print(f"Estimating importance of '{self.model.tokenizer.decode(instance[0, idx_feature])}'")

        if feature_groups is not None:
            eff_feature_groups = feature_groups
            idx_superfeature = feature_groups.index(idx_feature)
        else:
            eff_feature_groups = torch.arange(num_features)[perturbable_mask[0]]
            idx_superfeature = eff_feature_groups.tolist().index(idx_feature)

        # Permuted POSITIONS of (super)features inside `eff_feature_groups`
        indices = sample_permutations(upper=len(eff_feature_groups),
                                      indices=torch.arange(len(eff_feature_groups)),
                                      num_permutations=num_samples)
        feature_pos = torch.nonzero(indices == idx_superfeature, as_tuple=False)

        if hasattr(self.generator, "label_weights"):
            randomly_selected_label = torch.multinomial(self.generator.label_weights, num_samples=num_samples, replacement=True)
            randomly_selected_label = [self.generator.control_labels_str[i] for i in randomly_selected_label]
        else:
            randomly_selected_label = [None] * num_samples

        # is_masked[i] = perturbed features for i-th sample
        is_masked = torch.zeros((num_samples, num_features), dtype=torch.bool)
        for idx_sample in range(num_samples):
            curr_feature_pos = int(feature_pos[idx_sample, 1])
            changed_features = self.indexer(eff_feature_groups, indices[idx_sample, curr_feature_pos:])

            is_masked[idx_sample, changed_features] = True

        # Obtain index of observed (estimated) feature in generator's representation
        instance_tokens = self.model.from_internal_precise(instance, skip_special_tokens=False)["decoded_data"][0]
        instance_copy = instance.clone()
        instance_copy[0, idx_feature] = self.model.mask_token_id
        instance_copy_masked_tokens = \
            self.model.from_internal_precise(instance_copy, skip_special_tokens=False)["decoded_data"][0]

        gen_idx_feature = torch.nonzero(self._transform_masks(instance_tokens, instance_copy_masked_tokens),
                                        as_tuple=False).flatten()

        if self.is_aligned_vocabulary:
            instance_str = self.model.from_internal(instance)
            instance_generator = self.generator.to_internal(instance_str)

            # If generator has a longer max seq. len than model, mark the diff as "unmasked" (not to be generated)
            length_diff = self.generator.max_seq_len - self.model.max_seq_len
            is_masked = torch.hstack((is_masked, torch.zeros((num_samples, length_diff), dtype=torch.bool)))
        else:
            # Find out which tokens are masked after converting to generator representation
            gen_is_masked = []
            for idx_sample in range(num_samples):
                masked_instance = instance.clone()
                masked_instance[0, is_masked[idx_sample]] = self.model.tokenizer.mask_token_id
                masked_instance_tokens = self.model.from_internal_precise(masked_instance,
                                                                          skip_special_tokens=False)["decoded_data"][0]

                gen_is_masked.append(self._transform_masks(instance_tokens, masked_instance_tokens))

            is_masked = torch.stack(gen_is_masked)
            instance_tokens = self.model.from_internal_precise(instance, skip_special_tokens=False)["decoded_data"][0]
            instance_str = tuple(" ".join(s_tok) for s_tok in instance_tokens) \
                if isinstance(instance_tokens, tuple) else " ".join(instance_tokens)
            instance_generator = self.generator.to_internal([instance_str])

        all_examples = self.generator.generate_masked_samples(instance_generator["input_ids"],
                                                              generation_mask=is_masked,
                                                              idx_observed_feature=gen_idx_feature,
                                                              control_labels=randomly_selected_label,
                                                              **instance_generator["aux_data"])

        text_examples = self.generator.from_internal(all_examples)
        model_examples = self.model.to_internal(text_examples)

        # print("Final: ")
        # for i in range(2 * num_samples):
        #     print(f"({randomly_selected_label[i // 2]}) {self.generator.from_internal(all_examples[[i]])}")
        # print("-----")

        scores = self.model.score(model_examples["input_ids"], **model_examples["aux_data"])
        scores_with = scores[::2]
        scores_without = scores[1::2]
        assert scores_with.shape[0] == scores_without.shape[0]
        diff = scores_with - scores_without

        results = {
            "diff_mean": torch.mean(diff, dim=0),
            "diff_var": torch.var(diff, dim=0)
        }

        if self.return_samples:
            results["samples"] = all_examples

        if self.return_scores:
            results["scores"] = scores

        return results


if __name__ == "__main__":
    model = InterpretableBertForSequenceClassification(tokenizer_name="bert-base-uncased",
                                                       model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/snli_bert_uncased",
                                                       batch_size=2,
                                                       device="cpu",
                                                       max_seq_len=41)
    # generator = BertForControlledMaskedLMGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert_snli_clm_best",
    #                                                model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert_snli_clm_best",
    #                                                control_labels=["<ENTAILMENT>", "<NEUTRAL>", "<CONTRADICTION>"],
    #                                                batch_size=2,
    #                                                device="cpu",
    #                                                strategy="top_p",
    #                                                top_p=0.99,
    #                                                generate_cover=False)
    # generator = BertForMaskedLMGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert-base-uncased-snli-mlm",
    #                                      model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert-base-uncased-snli-mlm",
    #                                      batch_size=10,
    #                                      device="cpu",
    #                                      strategy="top_p",
    #                                      top_p=0.999,
    #                                      monte_carlo_dropout=True)
    # generator = GPTLMGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/gpt_snli_lm_maxseqlen42",
    #                            model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/gpt_snli_lm_maxseqlen42",
    #                            batch_size=10,
    #                            max_seq_len=42,
    #                            device="cpu",
    #                            strategy="top_p",
    #                            top_p=0.99)
    generator = GPTControlledLMGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/gpt_snli_clm_maxseqlen42",
                                         model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/gpt_snli_clm_maxseqlen42",
                                         control_labels=["<ENTAILMENT>", "<NEUTRAL>", "<CONTRADICTION>"],
                                         batch_size=2,
                                         device="cpu",
                                         strategy="top_p",
                                         top_p=0.99)
    # generator = BertForMaskedLMGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert-base-uncased-snli-mlm-ls01",
    #                                      model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert-base-uncased-snli-mlm-ls01",
    #                                      batch_size=2,
    #                                      device="cpu",
    #                                      strategy="top_p",
    #                                      top_p=0.99)
    # generator = BertForMaskedLMGenerator(tokenizer_name="bert-base-uncased",
    #                                      model_name="bert-base-uncased",
    #                                      batch_size=2,
    #                                      device="cpu",
    #                                      strategy="top_p",
    #                                      top_p=0.999)

    # generator = TrigramForMaskedLMGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/3gramlm-snli-base-uncased",
    #                                         model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/3gramlm-snli-base-uncased",
    #                                         batch_size=10,
    #                                         max_seq_len=41,
    #                                         device="cpu",
    #                                         strategy="top_p",
    #                                         top_p=0.99)

    explainer = DependentIMEMaskedLMExplainer(model=model,
                                              generator=generator,
                                              return_samples=True,
                                              return_scores=True,
                                              return_variance=True,
                                              return_num_samples=True,
                                              is_aligned_vocabulary=False)

    seq = ("A shirtless man skateboards on a ledge.", "A man without a shirt")
    res = explainer.explain_text(seq, label=0, min_samples_per_feature=10)
    print(f"Sum of importances: {sum(res['importance'])}")
    for curr_token, curr_imp, curr_var in zip(res["input"], res["importance"], res["var"]):
        print(f"{curr_token} = {curr_imp: .4f} (var: {curr_var: .4f})")
