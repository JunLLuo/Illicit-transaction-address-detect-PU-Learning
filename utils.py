import copy
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit


def get_eval_metrics(y_true, y_pred, to_print=True):
    cm_ = confusion_matrix(y_true, y_pred, labels=[0, 1])
    FP = cm_.sum(axis=0) - np.diag(cm_)
    FN = cm_.sum(axis=1) - np.diag(cm_)
    TP = np.diag(cm_)
    TN = cm_.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    P = ((TP + FP) / (TP + FP + FN + TN))
    T = ((TP + FN) / (TP + FP + FN + TN))

    PUM = (TPR * TPR) / P
    PUF1 = (TPR * TPR) / (2 * P)

    F1 = (2 * PPV[1] * TPR[1]) / (PPV[1] + TPR[1])

    if to_print:
        print(f'Precision: {np.round(PPV[1], 3)}')
        print(f'Recall: {np.round(TPR[1], 3)}')
        print(f'F1: {np.round(F1, 3)}')
        print(f'PUM: {np.round(PUM[1], 3)},   P: {P}')
        print(f'FPR: {np.round(FPR[1], 3)}')
        print(f'PUF1: {np.round(PUF1[1], 3)},   P:{P}   T: {T}')

    return PPV[1], TPR[1], PUM[1], PUF1[1], F1


def binarize_labels(raw_labels, mode = 'negative'):
    new_labels = []
    if mode == 'negative':
        for l in raw_labels:
            if l == 1:
                new_labels.append(1)
            else:
                new_labels.append(0)
    elif mode == 'positive':
        for l in raw_labels:
            if l in [1, 2]:
                new_labels.append(1)
            else:
                new_labels.append(0)
    return new_labels


def get_class_preds(y_pred, y_test):
    all_classes = [0, 1, 2]
    class_indices = {}
    for c in all_classes:
        class_i_in_y_test = []
        for i, y_ in enumerate(y_test):
            if y_ == c:
                class_i_in_y_test.append(i)
        class_indices[c] = class_i_in_y_test

    class_preds = {}
    for i, y_pred_each in enumerate(y_pred):
        for c, class_i_in_y_test in class_indices.items():
            if i in class_i_in_y_test:
                class_preds.setdefault(c, []).append(y_pred_each)

    return class_indices, class_preds


def train_test_split(X, y, rnd_seed, test_size=0.2):
    """
    split the features and the labels according to the indices
    :param test_size: [0-1] how much ratio of data for testing
    :param X: feature set, should be array or list
    :param y: labels, should be array or list
    :param rnd_seed: random seed
    """
    # generate indices for the train and test set
    indices = [i for i in range(len(y))]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=rnd_seed)
    sss.get_n_splits(indices, y)
    train_indices, test_indices = next(sss.split(indices, y))

    # train/test split
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]

    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, X_test, y_train, y_test


def split_and_label_pu_data(start_, how_many_unlabeled_positive, raw_labels, embeddings_):
    rnd_seed = 42

    if how_many_unlabeled_positive == 0:
        labels_ = raw_labels
    else:
        labels_ = copy.deepcopy(raw_labels)
        labels_[start_:start_ + how_many_unlabeled_positive] = [2] * how_many_unlabeled_positive

    X_train, X_test, y_train, y_test = train_test_split(embeddings_, labels_, rnd_seed)

    y_train = binarize_labels(y_train, 'negative')
    y_test_neg = binarize_labels(y_test, 'negative')
    y_test_pos = binarize_labels(y_test, 'positive')

    print(f'y_train: {Counter(y_train)}')
    print(f'y_test_neg: {Counter(y_test_neg)}')
    print(f'y_test_pos: {Counter(y_test_pos)}')

    class_i_in_y_test = []
    for i, y_ in enumerate(y_test):
        if y_ == 2:
            class_i_in_y_test.append(i)

    print(f'class 2 unlabeled positive: {len(class_i_in_y_test)}/{len(y_test)}')

    return X_train, X_test, y_train, y_test, y_test_neg, y_test_pos, labels_