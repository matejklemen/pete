from explain_nlp.experimental.core import MethodType
from explain_nlp.methods.ime import IMEExplainer
from explain_nlp.methods.ime_lm import IMEInternalLMExplainer, IMEExternalLMExplainer, IMEHybridExplainer
from explain_nlp.methods.lime import LIMEExplainer
from explain_nlp.methods.lime_lm import LIMEMaskedLMExplainer


def load_explainer(**kwargs):
    """ An extremely free-for-all method that loads the desired explanation method based on experiment arguments """
    method_class, method = kwargs["method_class"], kwargs["method"]
    model = kwargs["model"]
    return_model_scores = kwargs.get("return_model_scores", False)
    return_generated_samples = kwargs.get("return_generated_samples", False)

    if method_class == "ime":
        sample_constraints = {}
        if kwargs["experiment_type"] == "accurate_importances":
            sample_constraints["confidence_interval"] = kwargs["confidence_interval"]
            sample_constraints["max_abs_error"] = kwargs["max_abs_error"]

        if method == "ime":
            method_type = MethodType.IME
            method = IMEExplainer(
                sample_data=kwargs["used_sample_data"], model=model, return_scores=return_model_scores,
                return_num_samples=True, return_samples=return_generated_samples,
                **sample_constraints
            )
        elif method == "ime_elm":
            method_type = MethodType.INDEPENDENT_IME_MLM
            method = IMEExternalLMExplainer(
                model=model, generator=kwargs["generator"], num_generated_samples=kwargs["num_generated_samples"],
                return_scores=return_model_scores, return_num_samples=True, return_samples=return_generated_samples,
                **sample_constraints
            )
        elif method == "ime_ilm":
            method_type = MethodType.DEPENDENT_IME_MLM
            method = IMEInternalLMExplainer(
                model=model, generator=kwargs["generator"], return_scores=return_model_scores, return_num_samples=True,
                return_samples=return_generated_samples, shared_vocabulary=kwargs["shared_vocabulary"],
                **sample_constraints
            )
        elif method == "ime_hybrid":
            method_type = MethodType.DEPENDENT_IME_MLM
            method = IMEHybridExplainer(
                model=model, generator=kwargs["generator"], sample_data_generator=kwargs["used_sample_data"],
                data_weights=kwargs["data_weights"], return_scores=return_model_scores, return_num_samples=True,
                return_samples=return_generated_samples, shared_vocabulary=kwargs["shared_vocabulary"],
                **sample_constraints
            )
        else:
            raise NotImplementedError(f"Unsupported method: '{method}'")
    else:
        if method == "lime":
            method_type = MethodType.LIME
            method = LIMEExplainer(
                model=model, kernel_width=kwargs["kernel_width"], return_samples=return_generated_samples,
                return_scores=return_model_scores
            )
        elif method == "lime_lm":
            method_type = MethodType.LIME_LM
            method = LIMEMaskedLMExplainer(
                model=model, generator=kwargs["generator"], kernel_width=kwargs["kernel_width"],
                return_samples=return_generated_samples, return_scores=return_model_scores,
                shared_vocabulary=kwargs["shared_vocabulary"]
            )
        else:
            raise NotImplementedError(f"Unsupported method: '{method}'")

    return method, method_type
