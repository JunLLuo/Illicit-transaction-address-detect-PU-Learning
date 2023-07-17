import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn import svm
from pulearn import ElkanotoPuClassifier, BaggingPuClassifier
from utils import get_class_preds, get_eval_metrics


def get_LR_results(X_train, X_test, y_train, y_test, y_test_neg, y_test_pos, rnd_seed = 42):
    lr_clf = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1e5, random_state=rnd_seed)
    lr_clf.fit(X_train, y_train)

    y_pred = lr_clf.predict(X_test)

    print('LR')
    PPV_before, TPR_before, PUM_before, PUF1_before, F1_before = get_eval_metrics(y_test_neg, y_pred)
    PPV_after, TPR_after, PUM_after, PUF1_after, F1_after = get_eval_metrics(y_test_pos, y_pred)

    class_indices, class_preds = get_class_preds(y_pred, y_test)
    all_classes = list(Counter(y_test).keys())
    for c in all_classes:
        print(f'class {c} pred: {sorted(Counter(class_preds[c]).items(), key=lambda x: x[0])}')
    print()

    return PPV_before, TPR_before, PUM_before, PUF1_before, F1_before, PPV_after, TPR_after, PUM_after, PUF1_after, F1_after


def get_BaggingPu_results(X_train, X_test, y_train, y_test, y_test_neg, y_test_pos):
    svc = svm.SVC(degree=3, kernel='rbf', C=10, gamma=0.4)

    pu_estimator = BaggingPuClassifier(base_estimator=svc, n_estimators=50, n_jobs=-1)
    pu_estimator.fit(np.asarray(X_train), np.asarray(y_train))

    y_pred = pu_estimator.predict(X_test)
    for i in range(len(y_pred)):
        if y_pred[i] == -1:
            y_pred[i] = 0

    print('BaggingPU')
    PPV_before, TPR_before, PUM_before, PUF1_before, F1_before = get_eval_metrics(y_test_neg, y_pred)
    PPV_after, TPR_after, PUM_after, PUF1_after, F1_after = get_eval_metrics(y_test_pos, y_pred)

    class_indices, class_preds = get_class_preds(y_pred, y_test)
    all_classes = list(Counter(y_test).keys())
    for c in all_classes:
        print(f'class {c} pred: {sorted(Counter(class_preds[c]).items(), key=lambda x: x[0])}')
    print()

    return PPV_before, TPR_before, PUM_before, PUF1_before, F1_before, PPV_after, TPR_after, PUM_after, PUF1_after, F1_after


def get_Elkanoto_PU_results(X_train, X_test, y_train, y_test, y_test_neg, y_test_pos):
    svc = svm.SVC(C=10, kernel='rbf', degree=3, gamma=0.4, probability=True)

    elkanoto_pu_estimator = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.1)
    elkanoto_pu_estimator.fit(np.asarray(X_train), np.asarray(y_train))

    y_pred = elkanoto_pu_estimator.predict(X_test)
    for i in range(len(y_pred)):
        if y_pred[i] == -1:
            y_pred[i] = 0

    print('Elkanoto_PU')
    PPV_before, TPR_before, PUM_before, PUF1_before, F1_before = get_eval_metrics(y_test_neg, y_pred)
    PPV_after, TPR_after, PUM_after, PUF1_after, F1_after = get_eval_metrics(y_test_pos, y_pred)

    class_indices, class_preds = get_class_preds(y_pred, y_test)
    all_classes = list(Counter(y_test).keys())
    for c in all_classes:
        print(f'class {c} pred: {sorted(Counter(class_preds[c]).items(), key=lambda x: x[0])}')
    print()

    return PPV_before, TPR_before, PUM_before, PUF1_before, F1_before, PPV_after, TPR_after, PUM_after, PUF1_after, F1_after
