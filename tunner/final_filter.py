import os
import json
from configparser import ConfigParser


# load every epoch and find best
def main():
    path = './output/dndmv/'
    dirs = os.listdir(path)

    file_list = []
    for dir in dirs:
        if os.path.exists(path + dir + '/' + 'final.json'):
            file_list.append(dir)

    # load every training log and find best test uas
    file_tuple = [] # (name, uas, config)
    for dir in file_list:
        with open(path + dir + '/' + 'training.log', 'r') as f:
            cur_best_uas = 0.0
            for line in f.readlines():
                if line.find('test.uas=') != -1 and line.find('random') == -1 and line.find('final') == -1 and line.find('loss') == -1:
                    test_uas = float(line.strip().split('test.uas=')[-1])
                    if test_uas > cur_best_uas:
                        cur_best_uas = test_uas
        config_file = [f for f in os.listdir(path + '/' +dir) if os.path.isfile(path + '/' +dir + '/' + f) and f.endswith('ini')]
        parser = ConfigParser()
        # only one config file
        parser.read(path + '/' + dir + '/' + config_file[0])
        # parser to dict
        cfg = dict((name, value)
                   for section in parser.sections()
                   for name, value in parser.items(section))
        file_tuple.append((dir, cur_best_uas, cfg))
    # collect config
    all_config = {}
    for tup in file_tuple:
        cfg = tup[-1]
        for key, value in cfg.items():
            if key not in all_config:
                all_config[key] = set()
            all_config[key].add(value)
    # get needed keys
    needed_keys = []
    for key in all_config.keys():
        if len(all_config[key]) > 1:
            needed_keys.append(key)
    with open(path + 'final_result.log', 'w') as f:
        # collect used data
        f.write('name\t' + '\t'.join(needed_keys) + '\t' + 'uas\n')
        for instance in file_tuple:
            cfg = instance[-1]
            values = []
            for key in needed_keys:
                values.append(cfg[key])
            values = [instance[0]] + values + [str(instance[1])]
            f.write('\t'.join(values))
            f.write('\n')

if __name__ == '__main__':
    # path = './output/dndmv/'
    # dirs = os.listdir(path)
    #
    #
    # res = []
    # for dir in dirs:
    #     if os.path.exists(path + dir + '/' + 'final.json'):
    #         data = json.load(open(path + dir + '/' + 'final.json'))
    #         res.append((dir, float(data['final_uas'])))
    #
    # res = sorted(res, key=lambda x:x[1])
    # print(res[0])
    # print(res[-1])
    main()