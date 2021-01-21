import os
import sys
from typing import Optional, Union, Tuple, List

import torch
import logging

from explain_nlp.methods.generation import BertForMaskedLMGenerator, BertForControlledMaskedLMGenerator
from explain_nlp.methods.ime import IMEExplainer
from explain_nlp.methods.modeling import InterpretableModel, InterpretableBertForSequenceClassification
from explain_nlp.methods.utils import sample_permutations, extend_tensor

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
)


class DependentIMEMaskedLMExplainer(IMEExplainer):
    """ TODO: this will only work for BERT as of now (and implies the use of same vocabulary)
         I need to think about conversion from model repr. into text and then into generator repr. and make sure
         nothing gets lost or wrongly shifted. E.g. ["Wrong", "##ly"] vs. ["Wrongly"] representation
    """
    def __init__(self, model: InterpretableModel, generator: BertForMaskedLMGenerator,
                 confidence_interval: Optional[float] = None,  max_abs_error: Optional[float] = None,
                 return_variance: Optional[bool] = False, return_num_samples: Optional[bool] = False,
                 return_samples: Optional[bool] = False, return_scores: Optional[bool] = False,
                 criterion: Optional[str] = "squared_error"):
        dummy_sample_data = torch.randint(5, (1, 1), dtype=torch.long)
        super().__init__(sample_data=dummy_sample_data, model=model, confidence_interval=confidence_interval,
                         max_abs_error=max_abs_error, return_variance=return_variance,
                         return_num_samples=return_num_samples, return_samples=return_samples,
                         return_scores=return_scores, criterion=criterion)

        self.generator = generator

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

        # Custom features present
        if feature_groups is not None:
            eff_feature_groups = feature_groups
            # Convert group of features to a new, "artificial" superfeature
            idx_superfeature = feature_groups.index(idx_feature)
        # Use regular, "primary" units (e.g. subwords)
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

        # is_masked[i] = perturbed features (EXCLUDING observed feature)
        is_masked = torch.zeros((num_samples, num_features), dtype=torch.bool)
        observed_feature = torch.tensor([idx_feature] if isinstance(idx_feature, int) else idx_feature).repeat((num_samples, 1))
        for idx_sample in range(num_samples):
            curr_feature_pos = int(feature_pos[idx_sample, 1])
            # Get indices of perturbed primary units (e.g. subwords)
            changed_features = self.indexer(eff_feature_groups, indices[idx_sample, curr_feature_pos + 1:])
            # print(f"Sample#{idx_sample}: {changed_features}")

            is_masked[idx_sample, changed_features] = True

        # Convert tokens from model representation into text, remapping model's MASK representation into generator's
        instance_str = self.model.from_internal(instance)
        instance_generator = self.generator.to_internal(instance_str)  # "input_ids", "perturbable_mask", "aux_data" -> ["token_type_ids", "attention_mask"]

        # ---------------------------------------------
        # -- GENERATOR-SPECIFIC LOGIC (TO BE MOVED) ---
        # ---------------------------------------------
        num_batches = (num_samples + self.generator.batch_size - 1) // self.generator.batch_size

        eff_input_ids = instance_generator["input_ids"].repeat((num_samples, 1))
        eff_token_type_ids = instance_generator["aux_data"]["token_type_ids"]
        eff_attention_mask = instance_generator["aux_data"]["attention_mask"]
        eff_is_masked = is_masked  # done this way just so I remember the expected inputs
        eff_observed_feature = observed_feature

        is_controlled = all(curr_selected is not None for curr_selected in randomly_selected_label)
        if is_controlled:
            # Make room for control label at start of sequence (at pos. 1)
            eff_input_ids = extend_tensor(eff_input_ids)
            eff_token_type_ids = extend_tensor(eff_token_type_ids)
            eff_attention_mask = extend_tensor(eff_attention_mask)

            eff_is_masked = extend_tensor(eff_is_masked)
            eff_is_masked[:, 1] = False

            encoded_control_labels = self.generator.tokenizer.encode(randomly_selected_label, add_special_tokens=False)
            eff_input_ids[:, 1] = torch.tensor(encoded_control_labels)
            eff_attention_mask[:, 1] = 1

            # squeezing in a control signal shifts feature positions by one
            eff_observed_feature = observed_feature + 1

        original_input_ids = eff_input_ids[0].clone()
        eff_token_type_ids = eff_token_type_ids.repeat((self.generator.batch_size, 1)).to(self.generator.device)
        eff_attention_mask = eff_attention_mask.repeat((self.generator.batch_size, 1)).to(self.generator.device)

        mask_size = 1
        num_observed_chunks = (eff_observed_feature.shape[1] + mask_size - 1) // mask_size
        num_total_chunks = (num_features + mask_size - 1) // mask_size

        for idx_batch in range(num_batches):
            s_b, e_b = idx_batch * self.generator.batch_size, (idx_batch + 1) * self.generator.batch_size
            curr_inputs = eff_input_ids[s_b: e_b]
            curr_observed = eff_observed_feature[s_b: e_b]
            curr_batch_size = curr_inputs.shape[0]
            _batch_indexer = torch.arange(curr_batch_size)

            # First stage: predict the observed feature, which must/should be different from its original value
            for idx_chunk in range(num_observed_chunks):
                s_c, e_c = idx_chunk * mask_size, (idx_chunk + 1) * mask_size
                feats_to_predict = curr_observed[:, s_c: e_c]  # [curr_batch_size, mask_size]
                curr_mask_size = feats_to_predict.shape[1]
                original_values = curr_inputs[_batch_indexer.unsqueeze(1), feats_to_predict]
                curr_inputs[_batch_indexer.unsqueeze(1), feats_to_predict] = self.generator.mask_token_id

                res = self.generator.generator(input_ids=curr_inputs.to(self.generator.device),
                                               token_type_ids=eff_token_type_ids[:curr_batch_size],
                                               attention_mask=eff_attention_mask[:curr_batch_size])
                # Only operate on logits of masked features
                logits = res["logits"][_batch_indexer.unsqueeze(1), feats_to_predict]
                # Make original values unsamplable
                # logits[_batch_indexer.unsqueeze(1),
                #        torch.arange(curr_mask_size).repeat((curr_batch_size, 1)),
                #        original_values] = -float("inf")

                preds = []
                for pos in range(curr_mask_size):
                    # preds.append(self.generator.decoding_strategy(logits[:, pos, :]))
                    probas = torch.softmax(logits[:, pos, :], dim=-1)
                    preds.append(torch.multinomial(probas, num_samples=1))
                preds = torch.cat(preds, dim=1).cpu()

                curr_inputs[_batch_indexer.unsqueeze(1), feats_to_predict] = preds

            curr_masked = eff_is_masked[s_b: e_b]
            for idx_masked_chunk in range(num_total_chunks):
                s_c, e_c = idx_masked_chunk * mask_size, (idx_masked_chunk + 1) * mask_size
                is_feature_masked = curr_masked[:, s_c: e_c]  # [curr_batch_size, mask_size]
                curr_mask_size = is_feature_masked.shape[1]

                if not torch.any(is_feature_masked):
                    continue

                curr_inputs[:, s_c: e_c][is_feature_masked] = self.generator.tokenizer.mask_token_id
                # print("Before: ")
                # for i in range(curr_batch_size):
                #     print(self.generator.tokenizer.decode(curr_inputs[i]))

                res = self.generator.generator(input_ids=curr_inputs.to(self.generator.device),
                                               token_type_ids=eff_token_type_ids[:curr_batch_size],
                                               attention_mask=eff_attention_mask[:curr_batch_size])
                for pos in range(curr_mask_size):
                    logits = res["logits"][:, s_c + pos, :]
                    # preds = self.generator.decoding_strategy(logits)[:, 0].cpu()
                    probas = torch.softmax(logits, dim=-1)
                    preds = torch.multinomial(probas, num_samples=1)[:, 0].cpu()

                    curr_inputs[:, s_c + pos][is_feature_masked[:, pos]] = preds[is_feature_masked[:, pos]]

                # print("After:")
                # for i in range(curr_batch_size):
                #     print(self.generator.tokenizer.decode(curr_inputs[i]))
                # print("")

        all_examples = eff_input_ids.repeat((2, 1))
        # With original feature
        all_examples[::2] = eff_input_ids
        all_examples[torch.arange(0, 2 * num_samples, 2).unsqueeze(1), eff_observed_feature] = \
            original_input_ids[eff_observed_feature]

        # Without
        all_examples[1::2] = eff_input_ids

        valid_tokens = torch.ones(all_examples.shape[1], dtype=torch.bool)
        # Control signals are not necessary valid tokens inside model
        if is_controlled:
            valid_tokens[1] = False
            all_examples = all_examples[:, valid_tokens]

        modeling_kwargs = {
            "token_type_ids": eff_token_type_ids[0: 1, valid_tokens].cpu(),
            "attention_mask": eff_attention_mask[0: 1, valid_tokens].cpu()
        }

        scores = self.model.score(all_examples, **modeling_kwargs)
        scores_with = scores[::2]
        scores_without = scores[1::2]
        assert scores_with.shape[0] == scores_without.shape[0]
        diff = scores_with - scores_without

        # print("Final: ")
        # for i in range(2 * num_samples):
        #     print(f"({scores[i][label]: .3f} {randomly_selected_label[i // 2]}) {self.generator.from_internal(all_examples[[i]])}")
        #     print("")
        # print("-----")

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
                                                       device="cpu")
    # generator = BertForControlledMaskedLMGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert_snli_clm_best",
    #                                                model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert_snli_clm_best",
    #                                                control_labels=["<ENTAILMENT>", "<NEUTRAL>", "<CONTRADICTION>"],
    #                                                batch_size=2,
    #                                                device="cpu",
    #                                                strategy="greedy",
    #                                                top_k=5,
    #                                                generate_cover=False)
    generator = BertForMaskedLMGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert-base-uncased-snli-mlm",
                                         model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert-base-uncased-snli-mlm",
                                         batch_size=2,
                                         device="cpu",
                                         strategy="greedy")
    # generator = BertForMaskedLMGenerator(tokenizer_name="bert-base-uncased",
    #                                      model_name="bert-base-uncased",
    #                                      batch_size=2,
    #                                      device="cpu",
    #                                      strategy="greedy")

    explainer = DependentIMEMaskedLMExplainer(model=model,
                                              generator=generator,
                                              return_samples=True,
                                              return_scores=True,
                                              return_variance=True,
                                              return_num_samples=True)

    seq = ("A shirtless man skateboards on a ledge.", "A man without a shirt")
    res = explainer.explain_text(seq, label=0, min_samples_per_feature=10)
    print(f"Sum of importances: {sum(res['importance'])}")
    for curr_token, curr_imp, curr_var in zip(res["input"], res["importance"], res["var"]):
        print(f"{curr_token} = {curr_imp: .4f} (var: {curr_var: .4f})")
