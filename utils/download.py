import os
import urllib.request
import pandas as pd


def fetch_data(url, save_path, parameters=None):
    """
    :param url: url to dataset
    :param save_path: save directory
    :param parameters: is a dictionary that contains some of most important read_csv parameters
    :return:
    """
    # default
    sep = ";"
    header = None
    na_values = ""

    # specified by the user
    if parameters is not None:
        sep, header, na_values = parameters.values()
    
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    if not os.path.isfile(save_path):
        urllib.request.urlretrieve(url, save_path)

    data = pd.read_csv(save_path, sep=sep, header=header, na_values=na_values, )

    return data
