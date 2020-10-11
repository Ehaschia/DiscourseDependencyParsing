import argparse
import os
import time
from datetime import datetime
from typing import Dict, List
from ast import literal_eval

# import chainer
# import chainer.functions as F
# from chainer import cuda, optimizers, serializers
import torch

import dataloader
import utils
from models.dndmv.dmv import DMV
from models.dndmv.dndmv import DiscriminativeNeuralDMV
# from models.dndmv.trainer.online_em_trainer import OnlineEMTrainer
from models.dndmv.trainer.e2e_trainer import OnlineEMTrainer
from utils.data import ScidtbDataset, ScidtbDatasetWithEmb


def dndmv_main(args):
    ##################
    # Arguments
    model_name = "debug"

    path_config = args.config
    trial_name = None


    if trial_name is None or trial_name == "None":
        trial_name = utils.get_current_time()

    ##################
    # Path setting
    cfg = dndmv_config(utils.Config(path_config))

    basename = "%s.%s.%s" \
            % (model_name,
               utils.get_basename_without_ext(path_config),
               trial_name)

    path_log = os.path.join(cfg["results"], basename + ".training.log")
    path_train = os.path.join(cfg["results"], basename + ".training.jsonl")
    path_valid = os.path.join(cfg["results"], basename + ".validation.jsonl")
    path_snapshot = os.path.join(cfg["results"], basename + ".model")
    path_pred = os.path.join(cfg["results"], basename + ".evaluation.arcs")
    path_eval = os.path.join(cfg["results"], basename + ".evaluation.json")

    utils.set_logger(path_log)

    ##################
    # Random seed
    utils.set_seed(trial_name)

    ##################
    # Log so far
    utils.writelog("model_name=%s" % model_name)
    utils.writelog("path_config=%s" % path_config)
    utils.writelog("trial_name=%s" % trial_name)

    utils.writelog("path_log=%s" % path_log)
    utils.writelog("path_train=%s" % path_train)
    utils.writelog("path_valid=%s" % path_valid)
    utils.writelog("path_snapshot=%s" % path_snapshot)
    utils.writelog("path_pred=%s" % path_pred)
    utils.writelog("path_eval=%s" % path_eval)

    ##################
    # Data preparation

    train_dataset, dev_dataset, test_dataset = load_scidtb(cfg)


    ##################
    # Model preparation

    converter = lambda x, y: y

    # dmv and dndmv
    dmv = DMV(cfg, cfg["dmv_mode"]).cuda()
    dmv.train()
    nn = DiscriminativeNeuralDMV(cfg, {}, cfg["nn_mode"]).cuda()
    # nn = DiscriminativeNeuralDMV(dict_cfg, {"word": torch.from_numpy(initialW)}, cfg.getstr("nn_mode")).cuda()
    nn.optimizer = torch.optim.Adam(nn.parameters(), cfg["lr"])

    # build trainer
    trainer = OnlineEMTrainer(cfg, dmv, nn, converter, train_dataset, dev_dataset, test_dataset)

    # train
    uas_dmv, ll_dmv = trainer.evaluate(test_dataset, prefer_nn=False)
    # uas_nn, ll_nn = trainer.evaluate(test_dataset, prefer_nn=True)

    # trainer.init_train(cfg.getint("epoch_init"))
    trainer.init_train_v2(train_dataset, cfg["epoch_init"], True)
    trainer.train(cfg["epoch"], stop_hook=trainer.default_stop_hook)

    # evaluate
    dmv.load_state_dict(torch.load(trainer.workspace / 'best_ll' / 'dmv'))
    nn.load_state_dict(torch.load(trainer.workspace / 'best_ll' / 'nn'))
    final_uas, final_ll = trainer.evaluate(test_dataset, prefer_nn=True)



def dndmv_config(config: utils.Config) -> Dict:
    cfg = dict((name, literal_eval(value))
               for section in config.parser.sections()
               for name, value in config.parser.items(section))
    cfg["workspace"] = f'output/dndmv/{datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")}'
    while os.path.exists(cfg["workspace"]):
        cfg["workspace"] += 'r'
    return cfg

def load_scidtb(cfg: Dict):
    ##################
    # Data preparation
    begin_time = time.time()

    vocab_word = utils.read_vocab(os.path.join(cfg["data"], "scidtb-vocab", "words.vocab.txt"))
    vocab_postag = utils.read_vocab(os.path.join(cfg["data"], "scidtb-vocab", "postags.vocab.txt"))
    vocab_deprel = utils.read_vocab(os.path.join(cfg["data"], "scidtb-vocab", "deprels.vocab.txt"))
    vocab_relation = utils.read_vocab(os.path.join(cfg["data"], "scidtb-vocab", "relations.coarse.vocab.txt"))

    train = dataloader.read_scidtb("train", "", relation_level="coarse-grained")
    test = dataloader.read_scidtb("test", "gold", relation_level="coarse-grained")
    dev = dataloader.read_scidtb("dev", "gold", relation_level="coarse-grained")
    # build for debug
    remove_root(train)
    remove_root(test)
    remove_root(dev)
    train_dataset = ScidtbDatasetWithEmb(train, vocab_word=vocab_word, vocab_postag=vocab_postag, vocab_deprel=vocab_deprel,
                                         vocab_relation=vocab_relation, encoder=cfg["encoder"])
    test_dataset = ScidtbDatasetWithEmb(test, vocab_word=vocab_word, vocab_postag=vocab_postag, vocab_deprel=vocab_deprel,
                                        vocab_relation=vocab_relation, encoder=cfg["encoder"])
    dev_dataset = ScidtbDatasetWithEmb(dev, vocab_word=vocab_word, vocab_postag=vocab_postag, vocab_deprel=vocab_deprel,
                                       vocab_relation=vocab_relation, encoder=cfg["encoder"])
    utils.writelog("Build Kmeans cluster")
    std = train_dataset.norm_embed(None)
    dev_dataset.norm_embed(std)
    test_dataset.norm_embed(std)
    kmeas = train_dataset.kmeans(cfg["kcluster"], 42)
    train_dataset.kmeans_label(kmeas)
    dev_dataset.kmeans_label(kmeas)
    test_dataset.kmeans_label(kmeas)
    end_time = time.time()
    utils.writelog("Loaded the corpus. %f [sec.]" % (end_time - begin_time))

    cfg["num_tag"] = len(vocab_postag)
    cfg["num_pos"] = len(vocab_postag)
    cfg["num_word"] = len(vocab_word)
    cfg["num_deprel"] = len(vocab_deprel)
    cfg["num_relation"] = len(vocab_relation)

    # cfg cluster

    return train_dataset, dev_dataset, test_dataset

def remove_root(dataset: List[utils.DataInstance]):
    for ins in dataset:
        # print(ins)
        ins.edus = ins.edus[1:]
        ins.edus_head = ins.edus_head[1:]
        ins.edus_postag = ins.edus_postag[1:]
        ins.edu_ids = ins.edu_ids[1:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    try:
        dndmv_main(args)
    except Exception as e:
        utils.logger.error(e, exc_info=True)