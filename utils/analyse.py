from pandas.plotting import scatter_matrix
import matplotlib as plt
import numpy as np


def na_count(data, column):
    print("NaN counr: {:0.2f}".format(np.sum(data[column]) / len(data)))


def corr_column(data, column):
    corr_matrix = data.corr()
    corr_matrix[column].sort_values(ascending=False)


def describe(data, num_columns=None):
    print(data.describe())

    scatter_matrix(data[num_columns], figsize=(12, 8))
