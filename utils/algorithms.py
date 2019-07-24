
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

def model( X_train=None, parameters=None, name = "random forest", search='randomize search',):
    models = {
        "random forest": RandomForestClassifier(),
        "naive bayes":GaussianNB(),
        "":,
        "":,
        "":,
            }
