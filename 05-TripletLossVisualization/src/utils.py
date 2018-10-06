import json


def get_params(json_path):
    print("utils jason path:{}".format(json_path))
    with open(json_path) as f:
        params = json.load(f)
    return params