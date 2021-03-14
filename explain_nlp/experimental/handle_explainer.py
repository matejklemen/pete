from explain_nlp.experimental.core import MethodType
from explain_nlp.methods.dependent_ime_mlm import DependentIMEMaskedLMExplainer
from explain_nlp.methods.hybrid import HybridIMEExplainer
from explain_nlp.methods.ime import IMEExplainer
from explain_nlp.methods.ime_mlm import IMEMaskedLMExplainer
from explain_nlp.methods.lime import LIMEExplainer, LIMEMaskedLMExplainer


def load_explainer(method: str, model, confidence_interval, max_abs_error,
                   return_model_scores, return_generated_samples, **kwargs):
    if method == "ime":
        method_type = MethodType.IME
        method = IMEExplainer(sample_data=kwargs["used_sample_data"], model=model,
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
                                               return_samples=return_generated_samples, return_variance=True)
    elif method == "ime_hybrid":
        method_type = MethodType.DEPENDENT_IME_MLM
        method = HybridIMEExplainer(model=model, generator=kwargs["generator"],
                                    gen_sample_data=kwargs["used_sample_data"], data_weights=kwargs["data_weights"],
                                    confidence_interval=confidence_interval, max_abs_error=max_abs_error,
                                    return_scores=return_model_scores, return_num_samples=True,
                                    return_samples=return_generated_samples, return_variance=True)
    elif method == "lime":
        method_type = MethodType.LIME
        method = LIMEExplainer(model=model, kernel_width=kwargs["kernel_width"],
                               return_samples=return_generated_samples, return_scores=return_model_scores)
    elif method == "lime_lm":
        method_type = MethodType.LIME_LM
        method = LIMEMaskedLMExplainer(model=model, generator=kwargs["generator"], kernel_width=kwargs["kernel_width"],
                                       return_samples=return_generated_samples, return_scores=return_model_scores)
    else:
        raise NotImplementedError(f"Unsupported method: '{method}'")

    return method, method_type
