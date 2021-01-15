import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)
from copy import copy
from tunner.utils import ROOT_DIR, param_json2list, load_done_configs
from os import listdir
from os.path import isfile, join

bash_prefix = '#!/usr/bin/env bash\n'
python = '/home/zhanglw/bin/anaconda3/envs/dci/bin/python'
cuda = 'CUDA_VISIBLE_DEVICES'
dir_path = './dndmv_configs/'
program = 'dndmv_main.py'
class BaseGenerator:
    def generate(self, load_path, batch):
        onlyfiles = [f for f in listdir(load_path) if isfile(join(load_path, f))]
        onlyfiles = sorted(onlyfiles)
        b = [load_path + '/batch_runner' + str(i) + '.sh' for i in range(batch)]
        clis = [[] for _ in range(batch)]
        for idx, file_name in enumerate(onlyfiles):
            cuda_idx = idx % batch + 1
            run_prefix = cuda + '=' + str(cuda_idx) + ' ' + python + ' ' + program
            clis[idx%batch].append(run_prefix + ' --config ' + dir_path + file_name + '\n')

        for f_name, cli in zip(b, clis):
            print(f_name)
            with open(f_name, 'w') as f:
                s = ''.join(cli)
                f.write(s)

        # with open(load_path + '/batch_runner.sh', 'w') as f:
        #     f.write(bash_prefix)
        #     for idx, file_name in enumerate(onlyfiles):
        #         run_prefix = cuda + '=' + str(idx%4) + ' ' + python + ' ' + program
        #         if (idx+1) % batch == 0:
        #             f.write(run_prefix +' --config ' + dir_path + file_name + '\n')
        #             f.write('sleep 5\n')
        #         else:
        #             f.write(run_prefix + ' --config ' + dir_path + file_name + '\n')
        #             f.write('sleep 5\n')

if __name__ == '__main__':
    generator = BaseGenerator()
    generator.generate(ROOT_DIR + '/dndmv_configs', 3)
