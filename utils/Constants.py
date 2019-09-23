from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

PARAM_GRID = {
    "cls": {

    },
    "reg": {

    }
}

MODELS = {
    "cls": {
        "random forest": RandomForestClassifier(),
        "naive bayes": GaussianNB(),
    },
    "reg": {
        "ridge": Ridge(),
    }
}

SCORING = {
    "cls":  ["accuracy", "precision", "recall", "f1"],
    "reg": ["neg_mean_absolute_error", "neg_mean_squared_error"]
}

SCORING_FUNC = {
    "cls":  {
            "accuracy": accuracy_score,
            "f1": f1_score,
            "precision": precision_score,
            "recall": recall_score,
    },
    "reg": {
            "neg_mean_squared_error": mean_squared_error,
            "mean_absolute_error": mean_absolute_error,
    }
}