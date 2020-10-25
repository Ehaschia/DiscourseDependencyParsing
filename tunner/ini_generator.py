import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)
from copy import copy
from tunner.utils import ROOT_DIR, param_json2list, load_done_configs

root = ROOT_DIR + '/dndmv_configs'
if not os.path.exists(root):
    os.makedirs(root)

# parameters
parameters = {
    "dmv_mode": ["\'tdr\'"],
    "nn_mode": ["\'tdr\'"],
    "num_lex": ["0"],
    "cv": ["2"],
    "e_step_mode": ["\'viterbi\'"],
    "count_smoothing": ["1e-1"],
    "param_smoothing": ["1e-1"],
    "dim_pos_emb": ["20", "40"],
    "dim_word_emb": ["1536"],
    "dim_valence_emb": ["20"],
    "dim_deprel_emb": ["10"],
    "dim_relation_emb ": ["10"],
    "dim_hidden": ["200", "100"],
    "dim_pre_out_decision": ["32"],
    "dim_pre_out_child": ["64"],
    "dim_pre_out_root": ["64"],
    "lstm_dim_in": ["200"],
    "lstm_dim_out": ["32"],
    "lstm_dropout": ["0.0"],
    "lstm_layers": ["1"],
    "lstm_bidirectional": ["True"],

    "activation_func": ["\'relu\'"],
    "dropout": ["0.3"],
    "freeze_word_emb": ["False"],
    "freeze_pos_emb": ["True", "False"],
    "freeze_out_pos_emb": ["False"],
    "use_emb_as_w": ["False"],
    "end2end": ["True", "False"],
    "lr": ["0.001"],
    "e_batch_size": ["256", "128", "64", "32"],
    "m_batch_size": ["256", "128", "64", "32"],
    "clip_grad": ["5."],
    "epoch": ["100"],
    "epoch_init": ["10"],
    "epoch_nn": ["5"],
    "neural_stop_criteria": ["1e-3"],
    "same_len": ["False"],
    "shuffle": ["2"],
    "drop_last": ["False"],
    "max_len_train": ["60"],
    "max_len_eval": ["60"],
    "min_len_train": ["1"],
    "num_worker": ["3"],
    "device": ["\'cuda\'"],
    "use_pair": ["False"],
    "max_len": ["60"],
    "initial_tree_sampling": ["\'RB_RB_RB\'"],
    "encoder": ["\'bert\'"],
    "kcluster": ["50"],
    "share_valence_emb": ["True"],
    "pca": ["False"],
    "markov_label": ["gaussian", "nice"],
}

keys = list(parameters.keys())
configs = []


def generate_qbs_prefix():
    return '[hyperparams]\n'


def dfs(config, idx):
    for value in parameters[keys[idx]]:
        new_config = copy(config)
        new_config.append(keys[idx] + '=' + value)
        if idx + 1 == len(keys):
            configs.append(new_config)
        else:
            dfs(new_config, idx + 1)


def config_generate(config):
    return '\n'.join(config)


def configs_generate(configs, func):
    for idx in range(len(configs)):
        with open(root + '/dndmv_' + str(idx).zfill(len(str(len(configs)))) + '.ini', 'w') as f:
            f.write(func(configs[idx]))


def pbs_config_generate(config):
    return generate_qbs_prefix() + config_generate(config)


if __name__ == '__main__':
    dfs([], 0)
    configs_generate(configs, pbs_config_generate)
