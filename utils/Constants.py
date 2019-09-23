from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import multiprocessing

# for methods that support multiprocessing
CPU_COUNT = multiprocessing.cpu_count()

# Grid parameters for grid search and randomized search
PARAM_GRID = {
    "cls": {

    },
    "reg": {

    }
}

# Supported machine learning models
MODELS = {
    "cls": {
        "random forest": RandomForestClassifier(),
        "naive bayes": GaussianNB(),
    },
    "reg": {
        "ridge": Ridge(),
    }
}

# scoring methods respect the naming precised by Sklearn
SCORING = {
    "cls": ["accuracy", "precision", "recall", "f1"],
    "reg": ["neg_mean_absolute_error", "neg_mean_squared_error"]
}

# scoring functions
SCORING_FUNCS = {
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