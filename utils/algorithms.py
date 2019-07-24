
from sklearn.ensemble import RandomForestClassifier

def random_forest(X, y):
    rf = RandomForestClassifier()
    rf.fit(X, y)
    