from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

if __name__ == "__main__":
    np.random.seed(17)
    dist1_file = pd.read_csv("snli_train5000_orig_bert_embeddings.csv")
    dist2_file = pd.read_csv("snli_train5000_ime_bert_embeddings_r3.csv")

    assert dist1_file.shape[0] == dist2_file.shape[0]
    assert dist1_file.shape[1] == dist2_file.shape[1]

    num_examples = dist1_file.shape[0]
    indices = np.random.permutation(num_examples)
    TRAIN_SIZE, DEV_SIZE = 0.7, 0.1  # test set = rest
    train_rest_bnd = int(TRAIN_SIZE * num_examples)
    dev_test_bnd = train_rest_bnd + int((num_examples - train_rest_bnd) * (DEV_SIZE / (1 - TRAIN_SIZE)))

    train_X = np.vstack((dist1_file.iloc[indices[:train_rest_bnd]].values,
                         dist2_file.iloc[indices[:train_rest_bnd]].values))
    train_y = np.concatenate((np.zeros(train_rest_bnd), np.ones(train_rest_bnd)))

    shuf_inds = np.random.permutation(train_X.shape[0])
    train_X = train_X[shuf_inds]
    train_y = train_y[shuf_inds]

    dev_X = np.vstack((dist1_file.iloc[indices[train_rest_bnd: dev_test_bnd]].values,
                       dist2_file.iloc[indices[train_rest_bnd: dev_test_bnd]].values))
    dev_y = np.concatenate((np.zeros(dev_test_bnd - train_rest_bnd), np.ones(dev_test_bnd - train_rest_bnd)))

    test_X = np.vstack((dist1_file.iloc[indices[dev_test_bnd:]].values,
                       dist2_file.iloc[indices[dev_test_bnd:]].values))
    test_y = np.concatenate((np.zeros(num_examples - dev_test_bnd), np.ones(num_examples - dev_test_bnd)))

    best_n, best_acc = None, 0.0
    for n_estimators in [50, 100, 250, 500, 1000]:
        forest = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
        forest.fit(train_X, train_y)

        preds = forest.predict(dev_X)
        acc = accuracy_score(y_true=dev_y, y_pred=preds)
        print(f"n_estimators={n_estimators}, dev acc: {acc: .4f}")

        if acc > best_acc:
            best_acc = acc
            best_n = n_estimators

    combined_X = np.vstack((train_X, dev_X))
    combined_y = np.concatenate((train_y, dev_y))

    final_forest = RandomForestClassifier(n_estimators=best_n, n_jobs=-1)
    final_forest.fit(combined_X, combined_y)

    preds = forest.predict(test_X)
    acc = accuracy_score(y_true=test_y, y_pred=preds)
    print(f"[TEST SET] n_estimators={best_n}, test acc: {acc: .4f}")
