from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import torch
import numpy as np
import shap
from explain_nlp.methods.ime import IMEExplainer, estimate_max_samples
from explain_nlp.methods.modeling import InterpretableModel


class InterpretableLR(InterpretableModel):
    def __init__(self, skl_model):
        self.skl_model = skl_model

    def score(self, input_ids: torch.Tensor, **aux_data):
        np_data = input_ids.numpy()
        class_probas = self.skl_model.predict(np_data).reshape([-1, 1])
        return torch.from_numpy(class_probas)


if __name__ == "__main__":
    dataset = load_boston()

    X = pd.DataFrame(dataset["data"], columns=dataset["feature_names"])
    X = X[["LSTAT", "INDUS", "NOX", "PTRATIO", "RM", "TAX", "DIS", "AGE"]]
    y = pd.Series(dataset["target"], name="target")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=0.7)
    model = LinearRegression(normalize=True)
    model.fit(X_tr, y_tr)
    test_preds = model.predict(X_te)

    rmse = mean_squared_error(y_te, y_pred=test_preds) ** 0.5
    rmse_dummy = mean_squared_error(y_te, y_pred=[y_tr.mean() for _ in range(y_te.shape[0])]) ** 0.5
    print(f"[Mean regressor] RMSE: {rmse_dummy: .4f}")
    print(f"[Learned regressor] RMSE: {rmse: .4f}")

    chosen_instance = X_te.iloc[[0]].values
    ground_truth = y_te.iloc[[0]]
    predicted = test_preds[0]

    print("Explaining instance: ")
    print(chosen_instance)
    print(f"[Predicted] {predicted: .4f}")
    print(f"[Actual] {float(ground_truth): .4f}")

    # Our implementation (part 1: estimate required number of samples to satisfy the max allowed error constraint)
    custom_implementation = IMEExplainer(model=InterpretableLR(skl_model=model),
                                         sample_data=torch.from_numpy(X_tr.values),
                                         return_variance=True)
    res = custom_implementation.explain(torch.from_numpy(chosen_instance),
                                        min_samples_per_feature=1000, max_samples=8*1000)
    # Note: for estimation of number of samples, we need the variance of differences, not variance of importances
    required_max_samples = estimate_max_samples(res["var"] * res["taken_samples"],
                                                alpha=(1 - 0.99), max_abs_error=0.1)
    print(f"Required max samples: {required_max_samples}")

    # SHAP implementation of IME (seems to be a modified version of the original algorithm though)
    shap_implementation = shap.SamplingExplainer(model=model.predict, data=X_tr)
    shap_importances = shap_implementation.explain(chosen_instance, min_samples_per_feature=1000,
                                                   nsamples=int(required_max_samples))

    # Our implementation (part 2: obtain the actual importances)
    res = custom_implementation.explain(torch.from_numpy(chosen_instance),
                                        min_samples_per_feature=1000,
                                        max_samples=required_max_samples)

    for rank, (idx_lr, idx_shap, idx_custom) in enumerate(zip(np.argsort(np.abs(model.coef_))[::-1],
                                                              np.argsort(np.abs(shap_importances)).tolist()[::-1],
                                                              torch.argsort(torch.abs(res["importance"])).tolist()[::-1]), start=1):
        print(f"#{rank}")
        print(f"\t[LR COEFS] {X.columns[idx_lr]}, importance: {model.coef_[idx_lr]:.3f}")
        print(f"\t[SHAP implementation] {X.columns[idx_shap]}, importance: {shap_importances[idx_shap]: .3f}")
        print(f"\t[Our implementation] {X.columns[idx_custom]}, importance: {res['importance'][idx_custom]: .3f} "
              f"(+-{torch.sqrt(res['var'][idx_custom]): .3f})")
