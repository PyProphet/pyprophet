import json

from util import bunchify

def write_config(cf, path):
    with open(path, "w") as fp:
        json.dump(cf, fp, indent=4)

def read_config(path):
    with open(path, "r") as fp:
        return bunchify(json.load(fp))

def update(cf, key, value):
    if "." in key:
        key_path, final_key = key.rsplit(".", 1)
        for sub_key in key_path.split("."):
            cf = cf.get(sub_key)
            if cf is None:
                raise KeyError("invalid key %s" % key)
        if final_key not in cf:
            raise KeyError("invalid key %s" % key)
    else:
        final_key = key
    cf[final_key] = value

def iter_key_pathes(cf):
    assert isinstance(cf, dict)
    for k, v in cf.items():
        if isinstance(v, dict):
            for p in iter_key_pathes(v):
                yield "%s.%s" % (k, p)
        else:
            yield k

def get_key_pathes(cf):
    return list(iter_key_pathes(cf))
