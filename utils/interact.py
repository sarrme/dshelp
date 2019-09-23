import os
import sys
import json

from collections import OrderedDict


# current path where all the data will be stored
def current_path():
    """
    :return: path to file independent of the dev directory
    """
    pathname = os.path.dirname(sys.argv[0])
    path = os.path.abspath(pathname)
    return path


# load json a json file and create one if it doesn't exist
def load_json(path_file):
    """
    :param path_file: path to a json file
    :return: data contained in the json file
    """
    data = dict()

    try:
        json_file = open(path_file)
        data = json.load(json_file)
        json_file.close()

    except IOError:
        outfile = open(path_file, 'w')
        json.dump(data, outfile)
        outfile.close()

    return data


# save Python dictionary as a json file
def save_json(path_file, data):
    """
    :param path_file: path to a json file
    :param data: data to save in the json file
    :return: None
    """
    outfile = open(path_file, 'w')
    json.dump(data, outfile, indent=4)
    outfile.close()


# create a folder if it doesn't exist
def create_save_directory(newpath):
    """
    :param newpath: create a  directory
    :return: None
    """
    if not os.path.exists(newpath):
        os.makedirs(newpath)
