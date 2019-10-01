from pandas.plotting import scatter_matrix
import matplotlib as plt
import numpy as np


def na_count(data, column):
    for name, na_sum in data[column].isna().sum().todict():
        print("Percentage of NaN count for column {}: {:0.2f}%".format(name, 100*na_sum / len(data)))


def corr_column(data, column):
    corr_matrix = data.corr()
    corr_col = corr_matrix[column].sort_values(ascending=False)
    dic_corr = corr_col.to_dict()
    print("correlation between " + column + " and " + ", ".join(data.columns))

    for k, v in dic_corr.items():
        print(k.capitalize() + " {:.2f}".format(v))
    return corr_col


def describe(data, num_columns=None):
    desc = data.describe()
    sm = None
    if num_columns is not None:
        scatter_matrix(data[num_columns], figsize=(12, 8))

    return desc, sm 