import os
import sys
from typing import Optional, Union, Tuple

import torch
import logging

from explain_nlp.methods.decoding import greedy_decoding, top_p_decoding
from explain_nlp.methods.generation import SampleGenerator, BertForMaskedLMGenerator
from explain_nlp.methods.ime import IMEExplainer
from explain_nlp.methods.modeling import InterpretableModel, InterpretableBertForSequenceClassification
from explain_nlp.methods.utils import sample_permutations

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
)


class DependentIMEMaskedLMExplainer(IMEExplainer):
    """ TODO: this will only work for BERT as of now.
        I need to think about conversion from model repr. into text and then into generator repr. and make sure
        nothing gets lost or wrongly shifted.
    """
    def __init__(self, model: InterpretableModel, generator: BertForMaskedLMGenerator,
                 confidence_interval: Optional[float] = None,  max_abs_error: Optional[float] = None,
                 return_variance: Optional[bool] = False, return_num_samples: Optional[bool] = False,
                 return_samples: Optional[bool] = False, return_scores: Optional[bool] = False,
                 verbose: Optional[bool] = False, controlled: Optional[bool] = False):
        # IME requires sampling data so we give it dummy data and later override it with generated data
        dummy_sample_data = torch.empty((0, 0), dtype=torch.long)
        super().__init__(sample_data=dummy_sample_data, model=model, confidence_interval=confidence_interval,
                         max_abs_error=max_abs_error, return_variance=return_variance,
                         return_num_samples=return_num_samples, return_samples=return_samples,
                         return_scores=return_scores)

        self.verbose = verbose
        self.generator = generator
        logging.info(f"Devices: [model] {model.device}, [generator] {generator.device}")

        self.controlled = controlled
        self.hardcoded_labels = ["entailment", "neutral", "contradiction"]
        if self.controlled:
            print("Warning: labels for controlled LM are hardcoded at the moment (SNLI)!")
            # TODO: very very very hacked (hardcoded) labelset for SNLI
            assert all([f"<{label.upper()}>" in generator.tokenizer.all_special_tokens
                        for label in self.hardcoded_labels])

    def estimate_feature_importance(self, idx_feature: int, instance: torch.Tensor, num_samples: int,
                                    perturbable_mask: torch.Tensor, label=None, **modeling_kwargs):
        if self.controlled:
            assert label is not None
            # TODO: very very very hacked (hardcoded) labelset for SNLI just for proof of concept
            str_label = self.hardcoded_labels[label]

        # Note: instance is currently supposed to be of shape [1, num_features]
        num_features = int(instance.shape[1])
        perturbable_inds = torch.arange(num_features)[perturbable_mask[0]]

        indices = sample_permutations(upper=num_features, indices=perturbable_inds,
                                      num_permutations=num_samples)
        feature_pos = torch.nonzero(indices == idx_feature, as_tuple=False)

        samples = instance.repeat((2 * num_samples, 1))
        for idx_sample in range(num_samples):
            curr_feature_pos = int(feature_pos[idx_sample, 1])

            # With feature `idx_feature` set
            samples[2 * idx_sample, indices[idx_sample, curr_feature_pos + 1:]] = self.model.tokenizer.mask_token_id

            # With feature `idx_feature` randomized
            samples[2 * idx_sample + 1, indices[idx_sample, curr_feature_pos:]] = self.model.tokenizer.mask_token_id

        # Convert tokens from model representation into text, remapping model's MASK representation into generator's
        decoded_tokens = self.model.convert_ids_to_tokens(samples)
        remapped_decoded_tokens = []
        for curr_sample in decoded_tokens:
            remapped = []
            for tok in curr_sample:
                new_tok = tok
                if tok == self.model.tokenizer.mask_token:
                    new_tok = self.generator.tokenizer.mask_token

                remapped.append(new_tok)
            remapped_decoded_tokens.append(remapped)

        # TODO: standardize - method that doesnt modify tokens, just joins them, possibly doing fancy spacing tricks
        text_data = [self.model.tokenizer.convert_tokens_to_string(tokens) for tokens in remapped_decoded_tokens]

        # Prepare also unmasked data for use in generator (to provide context)
        instance_text = self.model.from_internal(instance)

        # Re-add a dummy label to seed controlled MLM as it gets removed when converting from internal representation:
        # this will be overriden by control signal(s) later
        if self.controlled:
            if isinstance(instance_text[0], tuple):
                instance_text = [(f"<{str_label.upper()}> {instance_text[0][0]}", instance_text[0][1])]
            else:
                instance_text = [f"<{str_label.upper()}> {instance_text}"]

        generator_instances = self.generator.to_internal(instance_text)
        generator_instances = {
            "input_ids": generator_instances["input_ids"].repeat((2 * num_samples, 1)),
            "aux_data": {k: v.repeat((2 * num_samples, 1)) for k, v in generator_instances["aux_data"].items()}
        }
        generator_masked_ids = []
        for data in text_data:
            # TODO: this is model-specific
            encoded = self.generator.tokenizer.encode(data, add_special_tokens=False)
            if self.controlled:
                rand_label = self.hardcoded_labels[int(torch.randint(len(self.hardcoded_labels), ()))]
                enc_lbl = self.generator.tokenizer.encode([f"<{rand_label.upper()}>"], add_special_tokens=False)[0]
                encoded[1] = enc_lbl

            generator_masked_ids.append(encoded)

        # Account for the possible changes in number of tokens:
        # After this, `input_ids` and `generator_masked_ids` will differ in length by 1 (the latter is longer),
        # but this doesn't affect the generation process because we are only iterating over length of `input_ids`
        generator_masked_ids = self.generator.tokenizer.pad({"input_ids": generator_masked_ids}, padding="longest")
        generator_masked_ids = torch.tensor(generator_masked_ids["input_ids"])

        input_ids = generator_instances["input_ids"]  # raw encoded (unmasked) input
        aux_data = generator_instances["aux_data"]

        if self.generator.top_p is None:
            def decoding_strategy(logits, ensure_diff_from):
                return greedy_decoding(logits, ensure_diff_from)
        else:
            def decoding_strategy(logits, ensure_diff_from):
                return top_p_decoding(logits, self.generator.top_p, ensure_diff_from)

        num_features = int(input_ids.shape[1])
        num_batches = (input_ids.shape[0] + self.generator.batch_size - 1) // self.generator.batch_size

        # Shuffle the order of generation
        generation_order = sample_permutations(upper=num_features, indices=perturbable_inds,
                                               num_permutations=2*num_samples)
        all_ex_indexer = torch.arange(generation_order.shape[0])
        for idx_masked in range(perturbable_inds.shape[0]):
            curr_indices = generation_order[:, idx_masked]
            curr_masked = \
                generator_masked_ids[all_ex_indexer, curr_indices] == self.generator.tokenizer.mask_token_id

            if not torch.any(curr_masked):
                continue

            # Mask current tokens
            input_ids[all_ex_indexer, curr_indices] = generator_masked_ids[all_ex_indexer, curr_indices]

            for idx_batch in range(num_batches):
                s_batch = idx_batch * self.generator.batch_size
                e_batch = min((idx_batch + 1) * self.generator.batch_size, input_ids.shape[0])

                curr_input_ids = input_ids[s_batch: e_batch]  # Note: view!
                curr_gen_indices = curr_indices[s_batch: e_batch]
                curr_batch_size = curr_input_ids.shape[0]
                curr_aux_data = {k: v[s_batch: e_batch].to(self.generator.device) for k, v in aux_data.items()}

                logits = self.generator.generator(curr_input_ids.to(self.generator.device), **curr_aux_data,
                                                  return_dict=True)["logits"]  # [B, seq_len, |V|]

                # Ensure a different token from the source
                # Note: assuming here that batch size is a multiple of 2
                every_second = torch.zeros(curr_batch_size, dtype=torch.bool)
                every_second[1::2] = True
                is_estimated_feature = torch.logical_and(curr_gen_indices == idx_feature, every_second)
                if torch.any(is_estimated_feature):
                    batch_indexer = torch.arange(curr_batch_size)[is_estimated_feature]
                    feature_indexer = curr_gen_indices[is_estimated_feature]
                    logits[batch_indexer, feature_indexer, input_ids[0, idx_feature]] = -float("inf")

                override_mask = curr_masked[s_batch: e_batch]
                preds = decoding_strategy(logits[torch.arange(curr_batch_size), curr_indices[s_batch: e_batch]],
                                          ensure_diff_from=None)  # [B, 1]
                preds = preds[override_mask]

                override_row = torch.arange(curr_batch_size)[override_mask]
                override_col = curr_indices[s_batch: e_batch][override_mask]

                curr_input_ids[override_row, override_col] = preds[:, 0].cpu()

            # After predicting the current feature for all samples using random control signals, set the "correct"
            # control signal (e.g. ground truth label)
            # TODO: fix this
            # if self.controlled and idx_masked == idx_feature:
            #     gen_encoded_label = self.generator.tokenizer.encode([f"<{str_label.upper()}>"],
            #                                                         add_special_tokens=False)[0]
            #     input_ids[:, 1] = gen_encoded_label
            #     generator_masked_ids[:, 1] = gen_encoded_label

        if self.verbose:
            logging.info(f"Estimating feature #{idx_feature}")
            logging.info("Before: ")
            for i in range(input_ids.shape[0]):
                logging.info(self.generator.tokenizer.decode(generator_masked_ids[i]))

            logging.info("After: ")
            for i in range(input_ids.shape[0]):
                logging.info(self.generator.tokenizer.decode(input_ids[i]))
            logging.info("")

        # Remove the auxilary token, used for controlled LM, as it isn't a valid model token
        if self.controlled:
            valid_tokens = torch.ones(input_ids[0].shape[0], dtype=torch.bool)
            valid_tokens[1] = False
            input_ids = input_ids[:, valid_tokens]
            modeling_kwargs = {k: v[:, valid_tokens] for k, v in modeling_kwargs.items()}

        scores = self.model.score(input_ids, **modeling_kwargs)
        scores_with = scores[::2]
        scores_without = scores[1::2]
        assert scores_with.shape[0] == scores_without.shape[0]
        diff = scores_with - scores_without

        results = {
            "diff_mean": torch.mean(diff, dim=0),
            "diff_var": torch.var(diff, dim=0)
        }

        if self.return_samples:
            results["samples"] = input_ids

        if self.return_scores:
            results["scores"] = scores

        return results

    def explain_text(self, text_data: Union[str, Tuple[str, ...]], label: Optional[int] = 0,
                     min_samples_per_feature: Optional[int] = 100, max_samples: Optional[int] = None):
        # TODO (maybe): remove the dummy token afterwards and shift all data accordingly so that the output is consistent with other methods? (i.e. that there isn't an additional token here)
        # If using controlled MLM, add a dummy token in front, which wont be perturbed, so that we don't need to
        # be careful about the offsets caused by additional control token(s)
        if self.controlled:
            _text_data = (f"{self.model.tokenizer.pad_token} {text_data[0]}", text_data[1]) if isinstance(text_data, tuple) \
                else f"{self.model.tokenizer.pad_token} {text_data}"
        else:
            _text_data = text_data

        # Convert instance being interpreted to representation of interpreted model
        model_instance = self.model.to_internal([_text_data])
        if self.controlled:
            # dummy token will be manually changed
            model_instance["perturbable_mask"][0, 1] = False

        # TODO: the control token is removed from generated samples, but not from the text being explained
        res = super().explain(model_instance["input_ids"], label, perturbable_mask=model_instance["perturbable_mask"],
                              min_samples_per_feature=min_samples_per_feature, max_samples=max_samples,
                              **model_instance["aux_data"])
        res["input"] = self.model.convert_ids_to_tokens(model_instance["input_ids"])[0]

        return res


if __name__ == "__main__":
    model = InterpretableBertForSequenceClassification(tokenizer_name="bert-base-uncased",
                                                       model_name="/home/matej/Documents/embeddia/interpretability/ime-lm/examples/weights/snli_bert_uncased",
                                                       batch_size=2,
                                                       device="cpu")
    generator = BertForMaskedLMGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/ime-lm/examples/weights/bert_snli_clm_best",
                                         model_name="/home/matej/Documents/embeddia/interpretability/ime-lm/examples/weights/bert_snli_clm_best",
                                         batch_size=2,
                                         device="cpu")

    explainer = DependentIMEMaskedLMExplainer(model=model,
                                              generator=generator,
                                              return_samples=True,
                                              return_scores=True,
                                              return_variance=True,
                                              return_num_samples=True,
                                              verbose=True,
                                              controlled=True)

    seq = ("A patient is being worked on by doctors and nurses.", "A man is sleeping.")
    res = explainer.explain_text(seq, label=2, min_samples_per_feature=20)
    print(res)
    for curr_token, curr_imp, curr_var in zip(res["input"], res["importance"], res["var"]):
        print(f"{curr_token} = {curr_imp: .4f} (var: {curr_var: .4f})")
