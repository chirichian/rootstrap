"""Utility function cross-project."""

import os
import pandas as pd

DATA_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__),'data/'))

import pdb
pdb.set_trace()

def data_path(*joins):
    """
    Data path getter

    Args:
        joins: Extra path to be join with the data path

    Returns:
        os.path data path
    """
    print(os.path.join(DATA_PATH, *joins))
    return os.path.join(DATA_PATH, *joins)


def load_data(file_name='International_Report_Passengers.csv'):
    """
    Client file loader

    Args:
        file_name: (optional). Str. Filename to load

    Returns:
        pd.DataFrame with client data
    """
    csv_path = data_path(file_name)
    df = pd.read_csv(csv_path)
    return df


