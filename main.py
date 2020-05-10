import argparse
import os
import time

import chainer
from chainer import cuda, serializers
import numpy as np

import utils

import dataloader
import models
import decoders
import training
import parsing
import metrics

def main(args):
    ##################
    # Arguments
    gpu = args.gpu
    model_name = args.model
    path_config = args.config
    trial_name = args.name
    actiontype = args.actiontype
    max_epoch = args.max_epoch
    dev_size = args.dev_size

    # Check
    assert actiontype in ["train", "evaluate"]
    if actiontype == "train":
        assert max_epoch > 0

    if trial_name is None or trial_name == "None":
        trial_name = utils.get_current_time()

    ##################
    # Path setting
    config = utils.Config(path_config)

    basename = "%s.%s.%s" \
            % (model_name,
               utils.get_basename_without_ext(path_config),
               trial_name)

    if actiontype == "train":
        path_log = os.path.join(config.getpath("results"), basename + ".training.log")
    else:
        path_log = os.path.join(config.getpath("results"), basename + ".evaluation.log")
    path_train = os.path.join(config.getpath("results"), basename + ".training.jsonl")
    path_valid = os.path.join(config.getpath("results"), basename + ".validation.jsonl")
    path_snapshot = os.path.join(config.getpath("results"), basename + ".model")
    path_pred = os.path.join(config.getpath("results"), basename + ".evaluation.ctrees")
    path_eval = os.path.join(config.getpath("results"), basename + ".evaluation.json")

    utils.set_logger(path_log)

    ##################
    # Random seed
    random_seed = trial_name
    random_seed = utils.hash_string(random_seed)
    np.random.seed(random_seed)
    cuda.cupy.random.seed(random_seed)

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

    utils.writelog("random_seed=%d" % random_seed)

    ##################
    # Data preparation
    begin_time = time.time()

    train_databatch = dataloader.read_rstdt("train", relation_level="coarse-grained", with_root=True)
    test_databatch = dataloader.read_rstdt("test", relation_level="coarse-grained", with_root=True)
    vocab_word = utils.read_vocab(os.path.join(config.getpath("data"), "rstdt-vocab", "words.vocab.txt"))
    vocab_postag = utils.read_vocab(os.path.join(config.getpath("data"), "rstdt-vocab", "postags.vocab.txt"))
    vocab_deprel = utils.read_vocab(os.path.join(config.getpath("data"), "rstdt-vocab", "deprels.vocab.txt"))
    vocab_relation = utils.read_vocab(os.path.join(config.getpath("data"), "rstdt-vocab", "relations.coarse.vocab.txt"))
    # vocab_nuclearity = utils.read_vocab(os.path.join(config.getpath("data"), "rstdt-vocab", "nuclearities.vocab.txt"))

    end_time = time.time()
    utils.writelog("Loaded the corpus. %f [sec.]" % (end_time - begin_time))

    ##################
    # Hyper parameters
    word_dim = config.getint("word_dim")
    postag_dim = config.getint("postag_dim")
    deprel_dim = config.getint("deprel_dim")
    lstm_dim = config.getint("lstm_dim")
    mlp_dim = config.getint("mlp_dim")
    batch_size = config.getint("batch_size")
    weight_decay = config.getfloat("weight_decay")
    gradient_clipping = config.getfloat("gradient_clipping")
    optimizer_name = config.getstr("optimizer_name")

    utils.writelog("word_dim=%d" % word_dim)
    utils.writelog("postag_dim=%d" % postag_dim)
    utils.writelog("depre_dim=%d" % deprel_dim)
    utils.writelog("lstm_dim=%d" % lstm_dim)
    utils.writelog("mlp_dim=%d" % mlp_dim)
    utils.writelog("batch_size=%d" % batch_size)
    utils.writelog("weight_decay=%f" % weight_decay)
    utils.writelog("gradient_clipping=%f" % gradient_clipping)
    utils.writelog("optimizer_name=%s" % optimizer_name)

    ##################
    # Model preparation
    cuda.get_device(gpu).use()

    # Initialize a model
    initialW = utils.read_word_embedding_matrix(
                        path=config.getpath("pretrained_word_embeddings"),
                        dim=word_dim,
                        vocab=vocab_word,
                        scale=0.0)
    if model_name == "arcfactoredmodel":
        template_feature_extractor1 = models.TemplateFeatureExtractor1(
                                                databatch=train_databatch,
                                                vocab_relation=vocab_relation)
        template_feature_extractor2 = models.TemplateFeatureExtractor2()
        utils.writelog("Template feature (1) size=%d" % template_feature_extractor1.feature_size)
        utils.writelog("Template feature (2) size=%d" % template_feature_extractor2.feature_size)
        if actiontype == "train":
            for template in template_feature_extractor1.templates:
                dim = template_feature_extractor1.template2dim[template]
                utils.writelog("Template feature (1) #%s %s" % (dim, template))
            for template in template_feature_extractor2.templates:
                dim = template_feature_extractor2.template2dim[template]
                utils.writelog("Template feature (2) #%s %s" % (dim, template))
        model = models.ArcFactoredModel(
                        vocab_word=vocab_word,
                        vocab_postag=vocab_postag,
                        vocab_deprel=vocab_deprel,
                        vocab_relation=vocab_relation,
                        word_dim=word_dim,
                        postag_dim=postag_dim,
                        deprel_dim=deprel_dim,
                        lstm_dim=lstm_dim,
                        mlp_dim=mlp_dim,
                        initialW=initialW,
                        template_feature_extractor1=template_feature_extractor1,
                        template_feature_extractor2=template_feature_extractor2)
    else:
        raise ValueError("Invalid model_name=%s" % model_name)
    utils.writelog("Initialized the model ``%s''" % model_name)

    # Load pre-trained parameters
    if actiontype != "train":
        serializers.load_npz(path_snapshot, model)
        utils.writelog("Loaded trained parameters from %s" % path_snapshot)

    model.to_gpu(gpu)

    ##################
    # Decoder preparation
    decoder = decoders.IncrementalEisnerDecoder()

    ##################
    # Training / evaluation
    if actiontype == "train":
        with chainer.using_config("train", True):
            if dev_size > 0:
                # Training with cross validation
                train_databatch, dev_databatch = dataloader.randomsplit(
                                                                n_dev=dev_size,
                                                                databatch=train_databatch)
                with open(os.path.join(config.getpath("results"), basename + ".valid_gold.arcs"), "w") as f:
                    for arcs in dev_databatch.batch_arcs:
                        arcs = ["%s-%s-%s" % (h,d,r) for (h,d,r) in arcs]
                        f.write("%s\n" % " ".join(arcs))
            else:
                # Training with the full training set
                dev_databatch = None
            training.train(
                model=model,
                decoder=decoder,
                max_epoch=max_epoch,
                batch_size=batch_size,
                weight_decay=weight_decay,
                gradient_clipping=gradient_clipping,
                optimizer_name=optimizer_name,
                train_databatch=train_databatch,
                dev_databatch=dev_databatch,
                path_train=path_train,
                path_valid=path_valid,
                path_snapshot=path_snapshot,
                path_pred=os.path.join(config.getpath("results"), basename + ".valid_pred.arcs"),
                path_gold=os.path.join(config.getpath("results"), basename + ".valid_gold.arcs"))

    elif actiontype == "evaluate":
        with chainer.using_config("train", False), chainer.no_backprop_mode():
            # Test
            parsing.parse(model=model,
                          decoder=decoder,
                          databatch=test_databatch,
                          path_pred=path_pred)
            scores = metrics.attachment_scores(
                        pred_path=path_pred,
                        gold_path=os.path.join(config.getpath("data"), "rstdt", "wsj", "test", "gold.arcs"))
            scores["LAS"] *= 100.0
            scores["UAS"] *= 100.0
            utils.write_json(path_eval, scores)
            utils.writelog(utils.pretty_format_dict(scores))

    utils.writelog("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--actiontype", type=str, required=True)
    parser.add_argument("--max_epoch", type=int, default=-1)
    parser.add_argument("--dev_size", type=int, default=-1)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        utils.logger.error(e, exc_info=True)

