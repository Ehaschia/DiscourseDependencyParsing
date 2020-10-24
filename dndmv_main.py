import argparse
import os
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from ast import literal_eval

# import chainer
# import chainer.functions as F
# from chainer import cuda, optimizers, serializers
import jsonlines
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
    cfg = dndmv_config(utils.Config(path_config))

    path_log = os.path.join(cfg["workspace"], "training.log")
    path_final = os.path.join(cfg["workspace"], "final.json")
    utils.set_logger(path_log)
    # move setting
    shutil.copy(path_config, cfg["workspace"])

    ##################
    # Random seed
    utils.set_seed(cfg["workspace"])
    ##################
    # Data preparation

    train_dataset, dev_dataset, test_dataset, kmeans = load_scidtb(cfg)

    ##################
    # Model preparation

    converter = lambda x, y: y

    # dmv and dndmv
    dmv = DMV(cfg, cfg["dmv_mode"]).cuda()
    dmv.train()
    # pos embed (current ignore)
    # pos_embed = kmeans.cluster_centers_
    nn = DiscriminativeNeuralDMV(cfg, {}, cfg["nn_mode"]).cuda()
    # nn = DiscriminativeNeuralDMV(dict_cfg, {"word": torch.from_numpy(initialW)}, cfg.getstr("nn_mode")).cuda()
    nn.optimizer = torch.optim.Adam(nn.parameters(), cfg["lr"])

    # build trainer
    trainer = OnlineEMTrainer(cfg, dmv, nn, converter, train_dataset, dev_dataset, test_dataset)

    # train
    uas_dmv, ll_dmv = trainer.evaluate(test_dataset, prefer_nn=False)
    utils.writelog(f"random dmv test.uas={uas_dmv}, ll={ll_dmv}")
    uas_nn, ll_nn = trainer.evaluate(test_dataset, prefer_nn=True)
    utils.writelog(f"random dmv test.uas={uas_nn}, ll={ll_dmv}")

    # trainer.init_train(cfg.getint("epoch_init"))
    trainer.init_train_v2(train_dataset, cfg["epoch_init"], True)
    trainer.train(cfg["epoch"], stop_hook=trainer.uas_stop_hook)

    # evaluate
    dmv.load_state_dict(torch.load(trainer.workspace / 'best_ll' / 'dmv'))
    nn.load_state_dict(torch.load(trainer.workspace / 'best_ll' / 'nn'))
    best_ll_uas, best_ll_ll = trainer.evaluate(test_dataset, prefer_nn=True)
    utils.writelog(f"final ll best dmv test.uas={best_ll_uas}, ll={best_ll_ll}")

    dmv.load_state_dict(torch.load(trainer.workspace / 'best_uas' / 'dmv'))
    nn.load_state_dict(torch.load(trainer.workspace / 'best_uas' / 'nn'))
    best_uas_uas, best_uas_ll = trainer.evaluate(test_dataset, prefer_nn=True)
    utils.writelog(f"final uas best dmv test.uas={best_uas_uas}, ll={best_uas_ll}")

    final_uas = jsonlines.Writer(open(path_final, "w"), flush=True)
    final_uas.write({"final_uas": max(best_ll_uas, best_uas_uas)})
    final_uas.close()


def dndmv_config(config: utils.Config) -> Dict:
    cfg = dict((name, literal_eval(value))
               for section in config.parser.sections()
               for name, value in config.parser.items(section))
    cfg["workspace"] = f'output/dndmv/{datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")}'
    while os.path.exists(cfg["workspace"]):
        cfg["workspace"] += 'r'
    utils.make_sure_dir_exists(Path(cfg["workspace"]))
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
    # utils.writelog("Build Kmeans cluster")
    std = train_dataset.norm_embed(None)
    dev_dataset.norm_embed(std)
    test_dataset.norm_embed(std)

    # kcluster labels
    # kmeans = train_dataset.kmeans(cfg["kcluster"], 42)
    # train_dataset.kmeans_label(kmeans)
    # dev_dataset.kmeans_label(kmeans)
    # test_dataset.kmeans_label(kmeans)

    # load markov label
    kmeans = None
    tag2ids = utils.load_markov_label('data/appendix/tags/', cfg['markov_label'], train_dataset, None)
    utils.load_markov_label('data/appendix/tags/', cfg['markov_label'], dev_dataset, tag2ids)
    utils.load_markov_label('data/appendix/tags/', cfg['markov_label'], test_dataset, tag2ids)
    end_time = time.time()
    utils.writelog("Loaded the corpus. %f [sec.]" % (end_time - begin_time))

    utils.writelog("Update markov label number: {}".format(len(tag2ids)))
    cfg['cluster'] = len(tag2ids)

    cfg["num_tag"] = len(vocab_postag)
    cfg["num_pos"] = len(vocab_postag)
    cfg["num_word"] = len(vocab_word)
    cfg["num_deprel"] = len(vocab_deprel)
    cfg["num_relation"] = len(vocab_relation)

    # cfg cluster

    # clean
    train_dataset.clean_encoder()
    dev_dataset.clean_encoder()
    test_dataset.clean_encoder()

    return train_dataset, dev_dataset, test_dataset, kmeans

def remove_root(dataset: List[utils.DataInstance]):
    for ins in dataset:
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