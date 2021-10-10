import argparse
import random
from datetime import datetime

import math
from copy import deepcopy
import torch
from torch import optim

from model import treebank_loader, utility
from model.unsup_model import NCRFAE, KMeansNCRFAE
from model.dep_evalutator import evaluate_uas, evaluate
import numpy as np
from collections import Counter
from sys import exit
import pickle
from scipy.special import softmax
from model.eisner import IncrementalEisnerDecoder
from treesamplers import TreeSampler
from utils import utils


def init_arg_parser():
    parser = argparse.ArgumentParser(description='Learning with NCRFAE')

    discourse_corpus = ('scidtb', 'rstdt')
    parser.add_argument('--corpus', type=str, default='scidtb', choices=discourse_corpus + ('wsj',))
    parser.add_argument('--gpu', type=int, default=0, help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size (10)')
    parser.add_argument('--hidden', type=int, default=50, help='hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout ratio')
    parser.add_argument('--epoch', type=int, default=50, help='maximum epoch number')
    parser.add_argument('--layers', type=int, default=1, help='number of LSTM layers')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.0, help='decay ratio of learning rate')
    parser.add_argument('--update', choices=['sgd', 'adadelta', 'adam'], default='adadelta', help='optimization method')
    parser.add_argument('--l2reg', type=float, default=0.001, help='L2 weight decay')
    parser.add_argument('--act_func', choices=['sigmoid', 'tanh', 'relu', 'gelu'], default='relu')
    parser.add_argument('--encode_method', type=str, choices=['edu', 'minus', 'avg_pooling', 'max_pooling',
                                                              'mean', 'endpoint', 'diffsum', 'coherent', 'attention'],
                        default='endpoint')
    parser.add_argument('--finetune', type=bool, default=False)
    parser.add_argument('--random_seed', type=int, default=40)
    parser.add_argument('--norm', type=bool, default=True)
    parser.add_argument('--init_epoch', type=int, default=20)
    parser.add_argument('--kcluster', type=int, default=50)
    parser.add_argument('--word_embed', type=bool, default=False)
    parser.add_argument('--pretrained_model', type=str, default='bert')
    parser.add_argument('--smooth', choices=['uniform', 'additive'], default='additive')
    parser.add_argument('--alpha', type=float, default=0.00001)
    parser.add_argument('--tree', choices=['gold', 'rightbranching'], default='rightbranching')
    return parser


def additive_smoothing(cnts, alpha):
    total_cnt = sum(cnts)
    d = len(cnts)
    prob = [0] * d
    for idx, cnt in enumerate(cnts):
        prob[idx] = (cnt + alpha) / (total_cnt + alpha*d)
    return prob

def uniform_smoothing(cnts, alpha):
    return (np.array(cnts) != 0).astype(float)

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        # in most case, cuda.manual_seed is not required because randomness op is done by cpu.
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(path):

    parser = init_arg_parser()
    args = parser.parse_args()

    # random seed
    set_seed(args.random_seed)

    # config = str(args.kcluster)+'label-'+args.tree+'-'+str(args.alpha)+str(args.smooth)+'-ncrfae-init'+str(args.init_epoch)+'-wordembed'+str(args.word_embed)+'-randomseed'+str(args.random_seed)

    utils.set_logger(f'output/{datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")}')
    utils.writelog(str(args))
    # load pretrained embedding
    embeddings = pickle.load(open(path, 'rb'))
    # ugly_code
    args.encode_method = path.split('embedding_')[-1].split('.')[0]
    alpha = args.alpha

    # data loading
    train, dev, test = treebank_loader.load_scidtb_dataset()

    USE_GPU = False
    device = torch.device('cpu')
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
        USE_GPU = True

    model = KMeansNCRFAE(args)
    if USE_GPU:
        model.cuda()

    # load embedding
    for ins in train:
        setattr(ins, 'r_embed', torch.from_numpy(embeddings[ins.raw.name]))

    for ins in dev:
        setattr(ins, 'r_embed', torch.from_numpy(embeddings[ins.raw.name]))

    for ins in test:
        setattr(ins, 'r_embed', torch.from_numpy(embeddings[ins.raw.name]))

    if args.norm:
        model.prepare_norm(train)
    model.calculate_kmeans(train)
    model.kmeans_label_predict(train)
    model.kmeans_label_predict(dev)
    model.kmeans_label_predict(test)
    model.init_embedding()

    treesampler = TreeSampler(['RB', 'X', 'RB'])

    # get matrix
    all_rule = []
    matrix = [[[] for _ in range(args.kcluster)] for _ in range(args.kcluster)]
    for sentence in train:
        # get label
        labels = np.array(sentence.kmeans_labels).astype(int)

        # labels=model.kmeans.predict(sentence.r_embed.cpu().detach().numpy())
        if args.tree == 'gold':
            arcs = sentence.raw.arcs
        else:
            arcs = treesampler.sample(sentence.raw.edu_ids, sentence.raw.edus, sentence.raw.edus_head, sentence.raw.sbnds, None, has_root=True)

        for arc in arcs:
            id1 = labels[arc[0]]
            id2 = labels[arc[1]]
            matrix[id1][id2].append(arc[-1])
            all_rule.append(arc[-1])

    all_rule = Counter(all_rule)
    counter_matrix = [[None for _ in range(args.kcluster)] for _ in range(args.kcluster)]
    for i in range(args.kcluster):
        for j in range(args.kcluster):
            counter_matrix[i][j] = Counter(matrix[i][j])

    prob_mat = [[0]*args.kcluster for _ in range(args.kcluster)]
    for i in range(args.kcluster):
        for j in range(args.kcluster):
            prob_mat[i][j] = sum(counter_matrix[i][j].values())
    # to prob
    for i in range(args.kcluster):
        i_all = float(sum(prob_mat[i]))
        if i_all == 0:
            # prob_mat[i] = additive_smoothing(prob_mat[i], alpha)
            continue
        else:
            if args.smooth == 'additive':
                prob_mat[i] = additive_smoothing(prob_mat[i], alpha)
            else:
                prob_mat[i] = uniform_smoothing(prob_mat[i], alpha)
            # for j in range(args.kcluster):
            #     prob_mat[i][j] = prob_mat[i][j] / i_all
    # for p in prob_mat:
    #     new_p = [str(i) for i in p]
    #     print('\t'.join(new_p))

    # # vector distance:
    # cluster_centers = model.kmeans.cluster_centers_
    #
    # mutual_distance = np.matmul(cluster_centers, np.transpose(cluster_centers))
    # prob_distance = softmax(mutual_distance, axis=1)
    # print('='*20)
    # for p in prob_distance:
    #     new_p = [str(i) for i in p]
    #     print('\t'.join(new_p))

    # train part
    model.set_prob(np.array(prob_mat))
    gd_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2reg)

    final = (0, 0.0, 0.0)
    max_dev = 0.0
    for epoch in range(args.epoch):
        # print("EPOCH-{}".format(epoch))
        model.train()
        random.shuffle(train)
        batches = utility.construct_batches(train, args.batch_size)
        random.shuffle(batches)

        for i, batch in enumerate(batches):
            sub_batches = utility.construct_batches_by_length(batch, args.batch_size)
            loss = 0
            for sub_batch in sub_batches:
                if epoch < args.init_epoch:
                    loss += len(sub_batch) * model.init_forward(sub_batch)
                else:
                    loss += len(sub_batch) * model(sub_batch)
            loss = loss / len(batch)
            # loss = loss
            loss.backward()
            gd_optimizer.step()
            gd_optimizer.zero_grad()

        # print("\tAfter GD: {}".format(evaluate(model, train)))
        # print('\tTRAIN-{}-UAS: {}'.format(K, evaluate_uas(model, train_K)))
        current_dev = evaluate_uas(model, dev)
        # print('\tDEV-UAS: {}'.format(current_dev))
        test_uas = evaluate_uas(model, test)
        # print('\tTEST-UAS: {}'.format(test_uas))
        if current_dev > final[1]:
            final = (epoch, current_dev, test_uas)
        # if current_dev > max_dev:
        #     test_uas = evaluate_uas(model, test)
        #     max_dev = current_dev
        #     print('\tTEST-UAS: {}'.format(test_uas))
        #     final = (epoch, max_dev, test_uas)
        # print('TEST-all-UAS: {}'.format(evaluate_uas(model, test)))
    print('FINAL-BEST-EPOCH-{}-DEV: {} TEST: {}'.format(*final))

    # direct eval
    # eisner = IncrementalEisnerDecoder()
    # predict = []
    # golden = []
    #
    # # log
    # prob = np.log(prob_mat)
    # for i in range(args.kcluster):
    #     for j in range(args.kcluster):
    #         if np.abs(prob[i][j]) == np.inf or math.isnan(prob[i][j]):
    #             prob[i][j] = -1e5
    #
    # for sentence in test:
    #     l = len(sentence.raw.edu_ids)
    #     score_mat_np = np.zeros((l, l))
    #     for i in range(l):
    #         for j in range(l):
    #             i_label = sentence.kmeans_labels[i]
    #             j_label = sentence.kmeans_labels[j]
    #             score_mat_np[i][j] = prob[i_label][j_label]
    #     unlabeled_arcs = eisner.global_decode(arc_scores=score_mat_np, edu_ids=sentence.raw.edu_ids,
    #                                           sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
    #                                           use_sbnds=True, use_pbnds=False)
    #     pred_arcs = [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]
    #     predict.append(np.array([0] + pred_arcs))
    #     golden.append(np.array(sentence.heads))
    # num_correct = 0
    # num_tokens = 0
    # for idx, (p1, g1) in enumerate(zip(predict, golden)):
    #     assert g1.size == p1.size
    #     num_correct += np.sum(g1[1:] == p1[1:])
    #     num_tokens += (g1[1:]).size
    #
    # if num_tokens == 0:
    #     uas = 0
    # else:
    #     uas = num_correct / num_tokens
    # print(uas)


path_list = ['data/embedding_edu.pkl',
             'data/embedding_minus.pkl',
             'data/embedding_avg_pooling.pkl',
             'data/embedding_diffsum.pkl',
             'data/embedding_coherent.pkl',
             'data/embedding_endpoint.pkl',
             'data/embedding_max_pooling.pkl']
if __name__ == '__main__':
    for path in path_list:
        print(path)
        main(path)
    # main('data/embedding_endpoint.pkl')