import pandas as pd

def read_csv(path, sep=None):
    table = pd.read_csv(path, sep)
    return table



