from pandas.plotting import scatter_matrix
import matplotlib as plt
import numpy as np


def na_count(data, column):
    for name, na_sum in data[column].isna().sum().todict():
        print("Percentage of NaN count for column {}: {:0.2f}%".format(name, 100*na_sum / len(data)))


def corr_column(data, column):
    corr_matrix = data.corr()
    corr_matrix[column].sort_values(ascending=False)


def describe(data, num_columns=None):
    print(data.describe())

    scatter_matrix(data[num_columns], figsize=(12, 8))
