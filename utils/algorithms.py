from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from util.interact import load_json

def model(X_train=None, y_train=None, name="random forest", search='randomize search', type='cs'):
    """
    Return a trained model, paramaters are to be specified

    :param type: type of the machine learning algorihm either classification or regression, "cs" for classification,
    anything for regression
    :param X_train: Training dataset
    :param parameters: parameters
    :param name: name of the model for classification or regression
    :param search: search method to find the best parameters
    :return: a trained model to be evaluated
    """
    models = dict()
    param_grid = load_json("parameters/param_grid.json")

    if type == "cs":
        models = {
            "random forest": RandomForestClassifier(),
            "naive bayes": GaussianNB(),
        }
    else:
        models = {}

    if type == "cs":
        param_grid = param_grid["cs"]

    else:
        param_grid = param_grid["re"]

    clf = models[name]

    search_methods = {'randomized search': {RandomizedSearchCV(clf, param_distributions=param_grid, scoring='')},
                      'grid search': {GridSearchCV(clf, param_grid=param_grid, scoring='')}
                      }

    search_method = search_methods[search]