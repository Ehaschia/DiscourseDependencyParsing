import argparse
import random
from datetime import datetime

from torch import optim
import torch
from torch.optim.lr_scheduler import ExponentialLR

from model import treebank_loader, utility
from model.unsup_model import NCRFAE, KMeansNCRFAE, KMeansBiaffineNCRFAE
from model.dep_evalutator import evaluate_uas, evaluate
import numpy as np
from sys import exit
from time import time
import utils


def filter_shorter_than(data, k):
    return list(filter(lambda x: len(x) <= k, data))

def filter_longer_than(data, k):
    return list(filter(lambda x: len(x) > k, data))

# used to choose learning method
def get_method(batch, args, epoch, supervied=False):
    if supervied:
        return 'supervised_forward_' + args.method
    if len(batch[0]) < args.partition_length:
        method = 'forward_' + args.method
    else:
        method = 'forward_head_selection'
    if epoch < args.init_epoch or len(batch[0]) < 10:
        method = 'init_' + method
    return method

def init_parameter():
    parser = argparse.ArgumentParser(description='Unsupervised Learning with NCRFAE')

    discourse_corpus = ('scidtb', 'rstdt')
    parser.add_argument('--corpus', type=str, default='scidtb', choices=discourse_corpus + ('wsj',))
    parser.add_argument('--gpu', type=int, default=0, help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--conf', '-c', default='configs/unsup_config.ini', help='path to config file')
    parser.add_argument('--log_name', default=f'semi_output/semi_{utils.get_current_time()}')
    args = parser.parse_args()
    args = utils.Config(args.conf).update(vars(args))
    return args


def main():
    args = init_parameter()

    # set log
    utils.set_logger(args.log_name)
    utils.writelog(str(args))


    # random seed
    utils.set_seed(utils.get_current_time())

    # args information
    corpus = args.corpus

    # data loading
    if corpus == 'scidtb':
        utils.writelog('\t Load {}'.format(corpus))
        # update n_rels in loader
        train, dev, test = treebank_loader.load_scidtb_dataset(args=args)
        utils.writelog("\tLoad embedding: {}".format(
            args.pretrained_model + '_' + args.corpus + '_' + args.encode_method + '.pkl'))
        utils.load_embedding(
            './data/embedding/' + args.pretrained_model + '_' + args.corpus + '_' + args.encode_method + '.pkl',
            [train, dev, test])
    elif corpus == 'rstdt':
        utils.writelog('\t Load {}'.format(corpus))
        # update n_rels in loader
        train, _, test = treebank_loader.load_rstdt_dataset(args=args)
        dev = []
        utils.writelog("\tLoad embedding: {}".format(
            args.pretrained_model + '_' + args.corpus + '_' + args.encode_method + '.pkl'))
        utils.load_embedding(
            './data/embedding/' + args.pretrained_model + '_' + args.corpus + '_' + args.encode_method + '.pkl',
            [train, test])
    else:
        raise NotImplementedError("Error Corpus!")


    all_train = train
    random.shuffle(all_train)
    all_train_size = len(all_train)
    unsupervised_part = int(all_train_size * (1-args.supvised_part))
    supervised_train = all_train[unsupervised_part:]
    for inst in supervised_train:
        inst.supervised = True

    if args.max_length > 0:
        # alert here the train is the unsupervised part
        unsupervised_train = all_train[:unsupervised_part]
        short_train = filter_shorter_than(unsupervised_train, args.max_length)
        long_train = filter_longer_than(unsupervised_train, args.max_length)

        # split long train instance to small instance by paragraph
        if len(long_train) > 0:
            split_train = []
            for inst in long_train:
                r_embed = inst.r_embed
                split_inst = utils.split_discourse(inst.raw, r_embed)
                split_train += split_inst
            split_train = filter_longer_than(split_train, 2)
            split_train = filter_shorter_than(split_train, args.max_length)
            train = short_train + split_train + supervised_train

    USE_GPU = False
    device = torch.device('cpu')
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
        USE_GPU = True

    if args.norm:
        # alert, norm the embedding, the embedding of splitted samples are also be normed.
        utils.writelog("\tNorm std")
        if corpus == 'rstdt':
            utils.norm_embedding(all_train, test)
        else:
            utils.norm_embedding(all_train, dev, test)

    utils.writelog('Building model')
    if args.model == 'kmeansncrfae':
        model = KMeansNCRFAE(args)
    elif args.model == 'kmeansbiaffinencrfae':
        model = KMeansBiaffineNCRFAE(args)
    else:
        raise NotImplementedError("Not implement method")
    model.calculate_kmeans(all_train)
    model.kmeans_label_predict(train)
    model.kmeans_label_predict(all_train)
    model.kmeans_label_predict(dev)
    model.kmeans_label_predict(test)
    model.init_embedding()
    model.set_prob(model.get_gold_prob_mat(supervised_train))
    exit(0)
    if USE_GPU:
        model.to(device)

    gd_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr, (args.mu, args.nu),
                              args.epsilon, weight_decay=args.l2reg)
    scheduler = ExponentialLR(gd_optimizer, args.decay ** (1 / args.decay_steps))

    utils.writelog("Before training:")

    utils.writelog("\tObj: {}".format(evaluate(model, train)))
    # print('\tTRAIN-{}-UAS: {}'.format(K, evaluate_uas(model, train_K)))
    max_dev = evaluate_uas(model, dev)
    utils.writelog('\tDEV-UAS: {}'.format(max_dev))
    max_test = evaluate_uas(model, test)
    utils.writelog('\tTEST-UAS: {}'.format(max_test))
    max_dev = 0.0
    final = (0, 0.0, 0.0)
    for epoch in range(args.epochs):
        begin_time = time()
        utils.writelog("EPOCH-{}".format(epoch))
        model.train()
        random.shuffle(train)
        batches = utility.construct_batches(train, args.batch_size)
        random.shuffle(batches)
        for i, batch in enumerate(batches):
            gd_optimizer.zero_grad()
            sub_batches = utility.construct_batches_by_length(batch, args.batch_size)
            loss = 0
            for sub_batch in sub_batches:
                loss += len(sub_batch) * model(sub_batch, method=get_method(sub_batch, args, epoch))
            loss = loss / len(batch)
            loss.backward()
            gd_optimizer.step()
        scheduler.step()
        # utils.writelog("\tAfter GD: {}".format(evaluate(model, train)))
        if len(dev) != 0:
            current_dev = evaluate_uas(model, dev)
            utils.writelog('\tDEV-UAS: {}'.format(current_dev))
        else:
            current_dev = 0.0
        # Test
        # test_uas = evaluate_uas(model, test)
        # utils.writelog('\tTEST-UAS: {}'.format(test_uas))
        if current_dev > max_dev:
            # max_test = test_uas
            max_dev = current_dev
            test_uas = evaluate_uas(model, test)
            final = (epoch, current_dev, test_uas)
            utils.writelog('\tTEST-UAS: {}'.format(test_uas))
        utils.writelog("Time-Used:\t{}".format(time() - begin_time))
    utils.writelog('FINAL-BEST-EPOCH-{}-DEV: {} TEST: {}'.format(*final))


if __name__ == '__main__':
    main()
