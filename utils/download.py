import os
import urllib.request
import pandas as pd


def fetch_data(url, save_path, parameters):
    """
    :param url: url to dataset
    :param save_path: save directory
    :return:
    """

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    if not os.path.isfile(save_path):
        urllib.request.urlretrieve(url, save_path)

    data = pd.read_csv(save_path, sep=";", header=None, na_values="", )

    return data
