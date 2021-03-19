import numpy as np


def fidelity(model_scores: np.ndarray, surrogate_scores: np.ndarray):
    """ Measures how well a surrogate model approximates the original model.
    Assuming the scores are probabilities, fidelity ranges from 0.5 (worst) to 1.0 (best).

    Args:
        model_scores:
            Scores, predicted by the interpreted model
        surrogate_scores:
            Scores, predicted by the surrogate model
    """
    return 1.0 / (np.abs(model_scores - surrogate_scores) + 1.0)


if __name__ == "__main__":
    print(fidelity(np.array([1.0, 0.5, 0.2, 0.4]), np.array([0.0, 0.5, 0.4, 0.2])))
