import os
import numpy as np
from argparse import ArgumentParser

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="snli",
                    choices=["snli"])
parser.add_argument("--mini_experiment_dir", type=str, default="debug/control")
parser.add_argument("--discriminator_model", type=str, default="mlp",
                    choices=["logistic_regression", "random_forest", "mlp"])

parser.add_argument("--random_seed", type=int, default=None)


if __name__ == "__main__":
    args = parser.parse_args()
    assert os.path.exists(args.mini_experiment_dir), \
        "--mini_experiment_dir must point to a valid directory. Please run embedding.py first in order to create it"

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    sample_features = np.load(os.path.join(args.mini_experiment_dir, "sample.npy"))
    other_features = np.load(os.path.join(args.mini_experiment_dir, "other.npy"))
    features = np.concatenate((sample_features, other_features))
    # 0 = original distribution, 1 = approximated distribution
    labels = np.concatenate((np.zeros(sample_features.shape[0]), np.ones(other_features.shape[0])))

    num_examples = labels.shape[0]
    rand_indices = np.random.permutation(num_examples)
    train_indices = rand_indices[:int(0.8 * num_examples)]
    dev_indices = rand_indices[int(0.8 * num_examples): int(0.9 * num_examples)]
    test_indices = rand_indices[int(0.9 * num_examples):]

    train_X, train_y = features[train_indices], labels[train_indices]
    dev_X, dev_y = features[dev_indices], labels[dev_indices]
    test_X, test_y = features[test_indices], labels[test_indices]

    print(f"Label distribution:"
          f"\ttrain: {np.unique(train_y, return_counts=True)}\n"
          f"\tdev: {np.unique(dev_y, return_counts=True)}\n"
          f"\ttest: {np.unique(test_y, return_counts=True)}")

    fixed_params = {"random_state": args.random_seed}
    if args.discriminator_model == "logistic_regression":
        model_class = LogisticRegression
        param_values = [
            {"C": 0.01}, {"C": 0.02}, {"C": 0.05},
            {"C": 0.1}, {"C": 0.2}, {"C": 0.5},
            {"C": 1.0}, {"C": 2.0}, {"C": 5.0}
        ]
    elif args.discriminator_model == "random_forest":
        model_class = RandomForestClassifier
        fixed_params.update({"n_jobs": -1})
        param_values = [
            {"n_estimators": 10}, {"n_estimators": 20}, {"n_estimators": 50},
            {"n_estimators": 100}, {"n_estimators": 200}, {"n_estimators": 500},
            {"n_estimators": 1000}, {"n_estimators": 2000}, {"n_estimators": 5000}
        ]
    elif args.discriminator_model == "mlp":
        model_class = MLPClassifier
        fixed_params.update({"early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 5})
        param_values = [
            {"hidden_layer_sizes": [100, 100], "alpha": 1e-4},
            {"hidden_layer_sizes": [100, 100], "alpha": 2e-4},
            {"hidden_layer_sizes": [100, 100], "alpha": 5e-4},
            {"hidden_layer_sizes": [200, 200], "alpha": 1e-4},
            {"hidden_layer_sizes": [200, 200], "alpha": 2e-4},
            {"hidden_layer_sizes": [200, 200], "alpha": 5e-4},
            {"hidden_layer_sizes": [500, 500], "alpha": 1e-4},
            {"hidden_layer_sizes": [500, 500], "alpha": 2e-4},
            {"hidden_layer_sizes": [500, 500], "alpha": 5e-4}
        ]
    else:
        raise NotImplementedError()

    best_params = None
    best_dev_acc = 0.0

    for curr_params in param_values:
        model = model_class(**fixed_params, **curr_params)
        model.fit(train_X, train_y)

        dev_preds = model.predict(dev_X)
        dev_acc = accuracy_score(y_true=dev_y, y_pred=dev_preds)

        if dev_acc > best_dev_acc:
            best_params = curr_params
            best_dev_acc = dev_acc

        print(f"{curr_params} -> dev_accuracy={dev_acc:.4f}")

    print(f"Best dev settings: {best_params}, dev_accuracy={best_dev_acc:.4f}")

    model = model_class(**fixed_params, **best_params)
    model.fit(train_X, train_y)
    test_preds = model.predict(test_X)
    dev_acc = accuracy_score(y_true=test_y, y_pred=test_preds)
