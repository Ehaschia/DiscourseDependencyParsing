import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import json
from tunner.utils import ROOT_DIR
import csv

output_dir = ROOT_DIR + '/output/'
finished_list = list(filter(lambda x: os.path.exists(output_dir + x + '/result.json'), os.listdir(output_dir)))
outer = open(output_dir + 'all.csv', 'w')
writer = csv.writer(outer)

keys = []

for finished_name in finished_list:
    with open(output_dir + finished_name + '/param.json') as f:
        params = json.load(f)
    with open(output_dir + finished_name + '/result.json') as f:
        result = json.load(f)
    param_keys = params.keys()
    keys = list(param_keys)

writer.writerow(keys + ['Epoch', 'Dev', 'Test'])

for finished_name in finished_list:
    with open(output_dir + finished_name + '/param.json') as f:
        params = json.load(f)
    with open(output_dir + finished_name + '/result.json') as f:
        result = json.load(f)
    to_write = []
    for key in keys:
        to_write.append(params[key])
    to_write.append(result['Epoch'])
    to_write.append(result['Dev'])
    to_write.append(result['Test'])
    writer.writerow(to_write)

outer.close()