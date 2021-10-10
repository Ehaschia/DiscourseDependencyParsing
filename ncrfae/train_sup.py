import argparse
import random

from torch import optim
from torch.optim.lr_scheduler import ExponentialLR

from model import treebank_loader, utility
from model.sup_model import *
from model.biaffine_model import Model, BiaffineAE
from model.unsup_model import NCRFAE
from datetime import datetime
import numpy as np

import utils


@torch.no_grad()
def evaluate_acc(model: NCRFAE, dataset: List[SCIDTBSentence]) -> Tuple[float, float]:
    if len(dataset) == 0:
        return 0.0, 0.0
    uas_num_correct, las_num_correct = 0, 0
    num_tokens = 0.0
    model.eval()
    arcs, rels = model.decoding(dataset)
    # trees = model.decoding(batch, enable_prior=False).detach().cpu().numpy()

    for i,inst in enumerate(dataset):
        inst_len = len(inst.raw.edu_ids)
        arcs_gold = np.array(inst.heads)
        rels_gold = np.array(inst.rels)

        arcs_correct = arcs_gold[1:] == arcs[i][1:inst_len]
        # here instance not add head, detail in dep_dataset.py line 85
        rels_correct = rels_gold == rels[i][1:inst_len]

        uas_num_correct += np.sum(arcs_correct)
        las_num_correct += np.sum(arcs_correct & rels_correct)
        num_tokens += (len(arcs[i]) - 1)

    uas_acc = uas_num_correct / num_tokens
    las_acc = las_num_correct / num_tokens
    return uas_acc, las_acc


@torch.no_grad()
def evaluate_eisner_acc(model: NCRFAE, dataset: List[SCIDTBSentence]) -> Tuple[float, float]:
    if len(dataset) == 0:
        return 0.0, 0.0
    uas_num_correct, las_num_correct = 0, 0
    num_tokens = 0.0
    model.eval()
    arcs, rels = model.eisner_decoding(dataset)
    # trees = model.decoding(batch, enable_prior=False).detach().cpu().numpy()

    for i,inst in enumerate(dataset):
        inst_len = len(inst.raw.edu_ids)
        arcs_gold = np.array(inst.heads)
        rels_gold = np.array(inst.rels)

        arcs_correct = arcs_gold[1:] == arcs[i][1:inst_len]
        # here instance not add head, detail in dep_dataset.py line 85
        rels_correct = rels_gold == rels[i][1:inst_len]

        uas_num_correct += np.sum(arcs_correct)
        las_num_correct += np.sum(arcs_correct & rels_correct)
        num_tokens += (len(arcs[i]) - 1)

    uas_acc = uas_num_correct / num_tokens
    las_acc = las_num_correct / num_tokens
    return uas_acc, las_acc



@torch.no_grad()
def evaluate(model, dataset):
    model.eval()
    loss = model(dataset)
    return loss


def init_parameter():
    parser = argparse.ArgumentParser(description='Learning with NCRFAE')

    discourse_corpus = ('scidtb', 'rstdt')
    parser.add_argument('--corpus', type=str, default='scidtb', choices=discourse_corpus + ('wsj',))
    parser.add_argument('--gpu', type=int, default=0, help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--conf', '-c', default='configs/test_config.ini', help='path to config file')
    parser.add_argument('--log_name', default=f'sup_output/su_{utils.get_current_time()}')

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
    utils.writelog("Pre-processing")

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

    USE_GPU = False
    device = torch.device('cpu')
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
        USE_GPU = True

    utils.writelog('Building model')
    # model = NCRF(args)
    if args.model == 'biaffine':
        model = Model(args)
    elif args.model == 'biaffineae':
        model = BiaffineAE(args)
        model.calculate_kmeans(train)
        model.kmeans_label_predict(train)
        model.kmeans_label_predict(dev)
        model.kmeans_label_predict(test)
        model.set_prob(model.get_gold_prob_mat(train))
    else:
        raise NotImplementedError('Not implement temp.')
    if USE_GPU:
        model.to(device)

    gd_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr, (args.mu, args.nu), args.epsilon, weight_decay=args.l2reg)
    scheduler = ExponentialLR(gd_optimizer, args.decay ** (1 / args.decay_steps))
    utils.writelog("Before training:")
    # Train
    if args.norm:
        utils.writelog("\tNorm std")
        if corpus == 'rstdt':
            utils.norm_embedding(train, test)
        else:
            utils.norm_embedding(train, dev, test)
    utils.writelog("\tObj: {}".format(evaluate(model, train)))
    # print('\tTRAIN-{}-UAS: {}'.format(K, evaluate_uas(model, train_K)))
    utils.writelog('\tDEV-UAS: {}'.format(evaluate_acc(model, dev)))
    max_dev_uas, max_dev_las = 0.0, 0.0
    final = (0, max_dev_uas, max_dev_las, 0.0, 0.0, 0.0, 0.0)
    for epoch in range(args.epochs):
        utils.writelog("EPOCH-{}".format(epoch))
        model.train()
        random.shuffle(train)
        batches = utility.construct_batches(train, args.batch_size)
        random.shuffle(batches)
        batch_loss = 0.0
        for i, batch in enumerate(batches):
            gd_optimizer.zero_grad()
            loss = model(batch)
            batch_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            gd_optimizer.step()
            scheduler.step()

        utils.writelog("\tAfter GD: {}".format(batch_loss))
        current_dev_uas, current_dev_las = evaluate_acc(model, dev)
        utils.writelog('\tDEV-UAS: {}, LAS: {}'.format(current_dev_uas, current_dev_las))

        # # Test
        if current_dev_uas > max_dev_uas or current_dev_las > max_dev_las: # or epoch +1 == args.epochs:
            test_uas, test_las = evaluate_acc(model, test)
            esiner_test_uas, esiner_test_las = evaluate_eisner_acc(model, test)
            max_dev_uas = current_dev_uas
            max_dev_las = current_dev_las
            utils.writelog('\tTEST-UAS:{} LAS:{}'.format(test_uas, test_las))
            final = (epoch, max_dev_uas, max_dev_las, test_uas, test_las, esiner_test_uas, esiner_test_las)
    if len(dev) == 0:
        test_uas, test_las = evaluate_acc(model, test)
        esiner_test_uas, esiner_test_las = evaluate_eisner_acc(model, test)
        final = (args.epochs-1, max_dev_uas, max_dev_las, test_uas, test_las, esiner_test_uas, esiner_test_las)
    utils.writelog('FINAL-BEST-EPOCH-{} DEV-UAS: {} DEV-LAS:{} TEST-UAS: {} TEST-LAS: {} Eisner-TEST-UAS: {} Eisner-TEST-LAS: {}'.format(*final))

if __name__ == '__main__':
    main()
