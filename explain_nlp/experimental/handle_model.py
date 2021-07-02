from typing import Union

from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification, \
    InterpretableXLMRobertaForSequenceClassification


def load_model(**kwargs) -> Union[InterpretableBertForSequenceClassification,
                                  InterpretableXLMRobertaForSequenceClassification]:
    model_type = kwargs["model_type"]
    model_params = {
        "model_name": kwargs["model_name"], "tokenizer_name": kwargs["tokenizer_name"],
        "batch_size": kwargs.get("batch_size", 32), "max_seq_len": kwargs["max_seq_len"],
        "device": kwargs.get("device", "cpu")
    }

    if model_type == "bert_sequence":
        return InterpretableBertForSequenceClassification(**model_params)
    elif model_type == "xlmr_sequence":
        return InterpretableXLMRobertaForSequenceClassification(**model_params)
    else:
        raise NotImplementedError(f"Unsupported model type: '{model_type}'")
