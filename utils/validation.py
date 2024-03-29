from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import StratifiedKFold

import utils.Constants as Consts

from collections import defaultdict

import numpy as np


def validation_test(model, X, y, problem="cls", test_size=0.2, random_stat=0):
    """
    :param model: a machine learning model
    :param X: features
    :param y: labels
    :param problem: type of problem either classification or regression
    :param test_size: test size
    :param random_stat: random stat 
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_stat)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scoring = Consts.SCORING[problem]
    scoring_fun = Consts.SCORING_FUNCS[problem]
    for metric in scoring:
        print(metric.capitalize() + ": {score:0.2f}".format(score=scoring_fun(y_pred, y_test)))


def validation_cv(model, X, y, method="cv", cv=5, problem="cls"):
    """
    Validation function independent of method, scoring and scoring functions are to be defined in the Constants.py file

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
    scoring = Consts.SCORING[problem]
    scoring_fun = Consts.SCORING_FUNCS[problem]
    scoring_sk = Consts.SKSCORING[problem]

    if method == 'cv':
        cv_scores = cross_validate(model, X, y, cv=TimeSeriesSplit(n_splits=cv), scoring=scoring_sk, n_jobs=Consts.CPU_COUNT)

        scores = defaultdict(list)

        for new_metric, metric in zip(scoring, scoring_sk):
            tranf = Consts.SCORE_TRANSF
            if metric in tranf:
                scores[new_metric] = tranf[metric](cv_scores["test_" + metric])
            else:
                scores[new_metric] = cv_scores["test_" + metric]

        print_scores(scoring, scores)

    if method == "ts":
        splitter_scores(model, X, y, TimeSeriesSplit, scoring, scoring_fun, n_splits=cv)

    if problem == "cls":
        if method == "ss":
            splitter_scores(model, X, y, StratifiedKFold, scoring, scoring_fun, n_splits=cv)


def splitter_scores(model, X, y, splitter_class, scoring, scoring_fun, n_splits=5):
    """
    :param model: a machine learning model
    :param X: features
    :param y: labels
    :param splitter_class: Class for splitting dataset
    :param scoring: scoring names
    :param scoring_fun: scoring functions
    :param n_splits: number of splits
    :return: scores with cross validation
    """
    scores = defaultdict(list)
    splt = splitter_class(n_splits=n_splits)

    for train, test in splt.split(X, y):
        X_train, X_test, y_train, y_test = X.loc[train], X.loc[test], y.loc[train], y.loc[test]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        for metric in scoring:
            scores[metric].append(scoring_fun[metric](y_pred, y_test))

    print_scores(scoring, scores)


def print_scores(scoring, dict_scores):
    for metric in scoring:
        print(text_result(metric, dict_scores[metric]))


def text_result(metric, scores):
    scores = np.array(scores)
    return metric.capitalize() + ": " + "{mean:0.2f} (+/- {confidence:0.2f})".format(mean=scores.mean(),
                                                                                     confidence=2 * scores.std())
