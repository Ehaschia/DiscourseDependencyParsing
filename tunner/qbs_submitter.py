import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)
from collections import deque
from tunner.utils import ROOT_DIR
import time
import subprocess

script_path = ROOT_DIR + '/scripts/'
cli_prefix = 'sh ' + script_path
skip = {}


def submitter(configs):
    # avoid submit to a same node
    config_map = dict()
    for config in configs:
        config_map[int(config.split('.')[0])] = config
    keys = sorted(list(config_map.keys()))
    for key in keys:
        time.sleep(3)
        print('qsub ' + script_path + config_map[key])
        subprocess.call('qsub ' + script_path + config_map[key], shell=True)

if __name__ == '__main__':
    configs = deque(os.listdir(script_path))
    print(len(configs))
    submitter(configs)
