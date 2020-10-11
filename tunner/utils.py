import json
import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

# Hyper setting
ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))


# load json
def load_done_configs(name):
    output_dir = os.path.join(ROOT_DIR, name)
    return filter(lambda x: os.path.exists(output_dir + '/' + x + '/result.json'), os.listdir(output_dir))


def param_json2list(json_file_name, keys):
    with open(json_file_name + '/param.json', 'r') as f:
        config = json.load(f)
    res = []
    for key in keys:
        res.append('--' + key + ' ' + str(config[key]))
    return res


# load the generate config
def load_configs(path):
    configs = []
    config_paths = os.listdir(path)
    for config_path in config_paths:
        with open(path + '/' + config_path, 'r') as f:
            configs.append((path + '/' + config_path, f.read()))
    return configs
