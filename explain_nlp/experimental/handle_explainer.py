from explain_nlp.experimental.core import MethodType
from explain_nlp.methods.dependent_ime_mlm import DependentIMEMaskedLMExplainer
from explain_nlp.methods.ime import IMEExplainer, SequentialIMEExplainer, WholeWordIMEExplainer
from explain_nlp.methods.ime_mlm import IMEMaskedLMExplainer


def load_explainer(method: str, model, confidence_interval, max_abs_error,
                   return_model_scores, return_generated_samples, **kwargs):
    if method in {"ime", "sequential_ime", "whole_word_ime"}:
        method_type = MethodType.IME

        used_sample_data = kwargs["used_sample_data"]
        if method == "ime":
            explainer_cls = IMEExplainer
        elif method == "sequential_ime":
            explainer_cls = SequentialIMEExplainer
        elif method == "whole_word_ime":
            explainer_cls = WholeWordIMEExplainer
        else:
            raise ValueError(f"'Unrecognized method: '{method}'")

        method = explainer_cls(sample_data=used_sample_data, model=model,
                               confidence_interval=confidence_interval, max_abs_error=max_abs_error,
                               return_scores=return_model_scores, return_num_samples=True,
                               return_samples=return_generated_samples, return_variance=True)
    elif method == "ime_mlm":
        method_type = MethodType.INDEPENDENT_IME_MLM
        method = IMEMaskedLMExplainer(model=model, generator=kwargs["generator"],
                                      confidence_interval=confidence_interval, max_abs_error=max_abs_error,
                                      num_generated_samples=kwargs["num_generated_samples"],
                                      return_scores=return_model_scores, return_num_samples=True,
                                      return_samples=return_generated_samples, return_variance=True)
    elif method == "ime_dependent_mlm":
        method_type = MethodType.DEPENDENT_IME_MLM
        method = DependentIMEMaskedLMExplainer(model=model, generator=kwargs["generator"],
                                               confidence_interval=confidence_interval, max_abs_error=max_abs_error,
                                               return_scores=return_model_scores, return_num_samples=True,
                                               return_samples=return_generated_samples, return_variance=True,
                                               controlled=kwargs["controlled"],
                                               seed_start_with_ground_truth=kwargs["seed_start_with_ground_truth"],
                                               reset_seed_after_first=kwargs["reset_seed_after_first"],
                                               verbose=kwargs.get("verbose", False))
    else:
        raise NotImplementedError(f"Unsupported method: '{method}'")

    return method, method_type
