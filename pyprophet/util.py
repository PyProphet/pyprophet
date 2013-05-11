class Bunch(dict):
    __getattr__ = dict.__getitem__
