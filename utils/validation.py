from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.model_selection import cross_validate
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import StratifiedKFold

from collections import defaultdict

import numpy as np


def validation(model, X, y, method="cv", cv=5, problem="cls"):
    """
    :param problem:
    :param model: model to evaluate
    :param X: features
    :param y: labels
    :param method: type of evaluation
    :param cv: number of folds
    :return: return the usual scores for variety of methods
    cross validation
    cross validation time series
    cross validation
    """

    if problem == "cls":
        scoring = ["accuracy", "precision", "recall", "f1"]
        scoring_fun = {"accuracy": accuracy_score,
                       "f1": f1_score,
                       "precision": precision_score,
                       "recall": recall_score}

    else:
        # actually is not neg_metric I leaved it as that for convenience
        # when the result will be shown it will be mean_squared_error and mean absolute error
        scoring = ["neg_mean_absolute_error", "neg_mean_squared_error"]
        scoring_fun = {"neg_mean_squared_error": mean_squared_error,
                       "mean_absolute_error": mean_absolute_error}

    if method == 'cv':
        cv_scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        if problem == "cls":

            scoring = ["test_" + metric for metric in scoring]
            print_scores(scoring, cv_scores)
        else:
            for metric in cv_scores:
                cv_scores[metric] = -cv_scores[metric]

            print_scores(scoring, cv_scores)

    if method == "ts":
        ts_scores = defaultdict(list)
        tscv = TimeSeriesSplit(n_splits=cv)

        for train, test in tscv.split(X):

            X_train, X_test, y_train, y_test = X.loc[train], X.loc[test], y.loc[train], y.loc[test]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            for metric in scoring:
                ts_scores[metric].append(scoring_fun[metric](y_pred, y_test))

        for metric in scoring:
            ts_scores[metric] = np.array(ts_scores[metric])
        print_scores(scoring, ts_scores)

    if problem == "cls":
        if method == "ss":
            ss_scores = defaultdict(list)
            skf = StratifiedKFold(n_splits=cv)

            for train, test in skf.split(X, y):

                X_train, X_test, y_train, y_test = X.loc[train], X.loc[test], y.loc[train], y.loc[test]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                for metric in scoring:
                    ss_scores[metric].append(scoring_fun[metric](y_pred, y_test))

                print_scores(scoring, ss_scores)


def print_scores(scoring, dict_scores):
    for metric in scoring:
        print(text_result(metric, dict_scores[metric]))


def text_result(metric, scores):
    return metric.capitalize() + ": " + "{mean:0.2f} (+/- {confidence:0.2f})".format(mean=scores.mean(),
                                                                                     confidence=2 * scores.std())