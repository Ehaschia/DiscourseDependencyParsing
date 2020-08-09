import argparse
import os
import time
from datetime import datetime
from typing import Dict

# import chainer
# import chainer.functions as F
# from chainer import cuda, optimizers, serializers
import torch

import dataloader
import utils
from models.dndmv.dmv import DMV
from models.dndmv.dndmv import DiscriminativeNeuralDMV
from models.dndmv.trainer.online_em_trainer import OnlineEMTrainer
from utils.data import ScidtbDataset


def dndmv_main(args):
    ##################
    # Arguments
    gpu = args.gpu
    model_name = args.model
    path_config = args.config
    trial_name = args.name
    actiontype = args.actiontype
    max_epoch = args.max_epoch
    dev_size = args.dev_size
    initial_tree_sampling = args.initial_tree_sampling

    # Check
    assert actiontype in ["train", "evaluate", "unsup_train"]
    if actiontype == "train":
        assert max_epoch > 0

    if trial_name is None or trial_name == "None":
        trial_name = utils.get_current_time()

    ##################
    # Path setting
    cfg = utils.Config(path_config)

    basename = "%s.%s.%s" \
            % (model_name,
               utils.get_basename_without_ext(path_config),
               trial_name)

    if actiontype == "train":
        path_log = os.path.join(cfg.getpath("results"), basename + ".training.log")
    elif actiontype == "unsup_train":
        path_log = os.path.join(cfg.getpath("results"), basename + ".training.log")
    else:
        path_log = os.path.join(cfg.getpath("results"), basename + ".evaluation.log")
    path_train = os.path.join(cfg.getpath("results"), basename + ".training.jsonl")
    path_valid = os.path.join(cfg.getpath("results"), basename + ".validation.jsonl")
    path_snapshot = os.path.join(cfg.getpath("results"), basename + ".model")
    path_pred = os.path.join(cfg.getpath("results"), basename + ".evaluation.arcs")
    path_eval = os.path.join(cfg.getpath("results"), basename + ".evaluation.json")

    utils.set_logger(path_log)

    ##################
    # Random seed
    utils.set_seed(trial_name)

    ##################
    # Log so far
    utils.writelog("gpu=%d" % gpu)
    utils.writelog("model_name=%s" % model_name)
    utils.writelog("path_config=%s" % path_config)
    utils.writelog("trial_name=%s" % trial_name)
    utils.writelog("actiontype=%s" % actiontype)
    utils.writelog("max_epoch=%s" % max_epoch)
    utils.writelog("dev_size=%s" % dev_size)

    utils.writelog("path_log=%s" % path_log)
    utils.writelog("path_train=%s" % path_train)
    utils.writelog("path_valid=%s" % path_valid)
    utils.writelog("path_snapshot=%s" % path_snapshot)
    utils.writelog("path_pred=%s" % path_pred)
    utils.writelog("path_eval=%s" % path_eval)

    ##################
    # Data preparation
    begin_time = time.time()

    vocab_word = utils.read_vocab(os.path.join(cfg.getpath("data"), "scidtb-vocab", "words.vocab.txt"))
    vocab_postag = utils.read_vocab(os.path.join(cfg.getpath("data"), "scidtb-vocab", "postags.vocab.txt"))
    vocab_deprel = utils.read_vocab(os.path.join(cfg.getpath("data"), "scidtb-vocab", "deprels.vocab.txt"))
    vocab_relation = utils.read_vocab(os.path.join(cfg.getpath("data"), "scidtb-vocab", "relations.coarse.vocab.txt"))
    train_dataset = ScidtbDataset(dataloader.read_scidtb("train", "", relation_level="coarse-grained"),
                                  vocab_word=vocab_word, vocab_postag=vocab_postag, vocab_deprel=vocab_deprel,
                                  vocab_relation=vocab_relation)
    test_dataset = ScidtbDataset(dataloader.read_scidtb("test", "gold", relation_level="coarse-grained"),
                                 vocab_word=vocab_word, vocab_postag=vocab_postag, vocab_deprel=vocab_deprel,
                                 vocab_relation=vocab_relation)
    dev_dataset = ScidtbDataset(dataloader.read_scidtb("dev", "gold", relation_level="coarse-grained"),
                                vocab_word=vocab_word, vocab_postag=vocab_postag, vocab_deprel=vocab_deprel,
                                vocab_relation=vocab_relation)

    end_time = time.time()
    utils.writelog("Loaded the corpus. %f [sec.]" % (end_time - begin_time))


    ##################
    # Model preparation
    # cuda.get_device(gpu).use()
    # device = torch.device("cuda")

    dict_cfg = dndmv_config(cfg)
    dict_cfg["num_tag"] = len(vocab_postag)
    dict_cfg["num_pos"] = len(vocab_postag)
    dict_cfg["num_word"] = len(vocab_word)
    dict_cfg["num_deprel"] = len(vocab_deprel)
    dict_cfg["num_relation"] = len(vocab_relation)

    converter = lambda x, y: y

    # Initialize a model
    initialW = utils.read_word_embedding_matrix(
        path=cfg.getpath("pretrained_word_embeddings"),
        dim=dict_cfg["dim_word_emb"],
        vocab=vocab_word,
        scale=0.0)

    # dmv and dndmv
    dmv = DMV(dict_cfg, cfg.getstr("dmv_mode")).cuda()
    dmv.train()
    nn = DiscriminativeNeuralDMV(dict_cfg, {"word": torch.from_numpy(initialW)}, cfg.getstr("nn_mode")).cuda()
    nn.optimizer = torch.optim.Adam(nn.parameters(), cfg.getstr("lr"))

    # build trainer
    trainer = OnlineEMTrainer(dict_cfg, dmv, nn, converter, train_dataset, dev_dataset, test_dataset)

    # train
    # uas_dmv, ll_dmv = trainer.evaluate(test_dataset, prefer_nn=False)
    # uas_nn, ll_nn = trainer.evaluate(test_dataset, prefer_nn=True)

    # TODO change the init_train to using right branch tree
    # How To do that?
    # trainer.init_train(cfg.getint("epoch_init"))
    trainer.train(cfg.getint("epoch"), stop_hook=trainer.default_stop_hook)

    # evaluate
    dmv.load_state_dict(torch.load(trainer.workspace / 'best_ll' / 'dmv'))
    nn.load_state_dict(torch.load(trainer.workspace / 'best_ll' / 'nn'))
    final_uas, final_ll = trainer.evaluate(test_dataset, prefer_nn=True)



def dndmv_config(config: utils.Config, ) -> Dict:
    cfg = {}

    # model hyperparameter
    cfg["dmv_mode"] = config.getstr("dmv_mode")
    cfg["nn_mode"] = config.getstr("nn_mode")
    cfg["num_lex"] = config.getint("num_lex")

    cfg["cv"] = config.getint("cv")
    cfg["e_step_mode"] = config.getstr("e_step_mode")
    cfg["count_smoothing"] = config.getfloat("count_smoothing")
    cfg["param_smoothing"] = config.getfloat("param_smoothing")

    cfg["dim_word_emb"] = config.getint("dim_word_emb")
    cfg["dim_pos_emb"] = config.getint("dim_pos_emb")
    cfg["dim_valence_emb"] = config.getint("dim_valence_emb")
    cfg["dim_deprel_emb"] = config.getint("dim_deprel_emb")
    cfg["dim_relation_emb"] = config.getint("dim_relation_emb")
    cfg["dim_hidden"] = config.getint("dim_hidden")

    cfg["dim_pre_out_decision"] = config.getint("dim_pre_out_decision")
    cfg["dim_pre_out_child"] = config.getint("dim_pre_out_child")
    cfg["dim_pre_out_root"] = config.getint("dim_pre_out_root")

    cfg["lstm_dim_in"] = config.getint("lstm_dim_in")
    cfg["lstm_dim_out"] = config.getint("lstm_dim_out")
    cfg["lstm_dropout"] = config.getfloat("lstm_dropout")
    cfg["lstm_layers"] = config.getint("lstm_layers")
    cfg["lstm_bidirectional"] = config.getbool("lstm_bidirectional")


    cfg["activation_func"] = config.getstr("activation_func")
    cfg["dropout"] = config.getstr("dropout")

    cfg["freeze_word_emb"] = config.getbool("freeze_word_emb")
    cfg["freeze_pos_emb"] = config.getbool("freeze_pos_emb")
    cfg["freeze_out_pos_emb"] = config.getbool("freeze_out_pos_emb")

    cfg["use_emb_as_w"] = config.getbool("use_emb_as_w")

    cfg["share_valence_emb"] = config.getint("cv") == 2

    # train config
    cfg["end2end"] = config.getbool("end2end")
    cfg["lr"] = config.getfloat("lr")
    cfg["e_batch_size"] = config.getint("e_batch_size")
    cfg["m_batch_size"] = config.getint("m_batch_size")

    cfg["epoch"] = config.getint("epoch")
    cfg["epoch_init"] = config.getint("epoch_init")
    cfg["epoch_nn"] = config.getint("epoch_nn")
    cfg["neural_stop_criteria"] = config.getfloat("neural_stop_criteria")

    cfg["same_len"] = config.getbool("same_len")
    cfg["shuffle"] = config.getint("shuffle")
    cfg["drop_last"] = config.getbool("drop_last")
    cfg["max_len_train"] = config.getint("max_len_train")
    cfg["max_len_eval"] = config.getint("max_len_eval")
    cfg["min_len_train"] = config.getint("min_len_train")
    cfg["num_worker"] = config.getint("num_worker")
    cfg["device"] = config.getstr("device")
    cfg["workspace"] = f'output/dndmv/{datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")}'
    while os.path.exists(cfg["workspace"]):
        cfg["workspace"] += 'r'

    # data config
    cfg["use_pair"] = config.getbool("use_pair")
    cfg["max_len"] = config.getint("max_len")
    cfg["vocab"] = config.getpath("vocab")

    cfg["word_emb"] = config.getpath("word_emb")
    cfg["out_pos_emb"] = config.getpath("out_pos_emb")
    cfg["pos_emb"] = config.getpath("pos_emb")

    cfg["pretrained_ds"] = config.getpath("pretrained_ds")
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--actiontype", type=str, required=True)
    parser.add_argument("--max_epoch", type=int, default=-1)
    parser.add_argument("--dev_size", type=int, default=-1)
    parser.add_argument("--initial_tree_sampling", type=str, default="RB_RB_RB")
    args = parser.parse_args()
    try:
        dndmv_main(args)
    except Exception as e:
        utils.logger.error(e, exc_info=True)