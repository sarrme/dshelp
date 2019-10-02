import os
import urllib.request
import pandas as pd
from pandas.api.types import is_string_dtype


def detect_type_missing_values(data):
    num_columns = []
    cat_columns = []
    missing_values = list()
    data_copy = data.copy().astype('str')
    n = len(data_copy)
    for column in data.columns:
        string_columns = data_copy[column].loc[~data_copy[column].str.contains('^\d+$|^\d+[\.,]$|^\d+[\.,]\d+$')]
        perc = len(string_columns) / n

        if perc < 0.5:
            num_columns.append(column)
            missing_values.extend(list(string_columns))
        else:
            cat_columns.append(column)

    return num_columns, cat_columns, missing_values


def fetch_data(save_path, url=None, sep=",", header=None):
    """
    :param header:
    :param sep:
    :param url: url to dataset
    :param save_path: save directory
    :return:
    """
    if url is not None:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        if not os.path.isfile(save_path):
            urllib.request.urlretrieve(url, save_path)

    data = pd.read_csv(save_path, sep=sep, header=header)
    num_columns, cat_columns, missing_values = detect_type_missing_values(data)
    data = pd.read_csv(save_path, sep=sep, header=header, na_values=missing_values)

    for column in num_columns:
        if is_string_dtype(data[column]):
            data[column] = data[column].str.replace(',', '.').astype("float")
    return data, num_columns, cat_columns
