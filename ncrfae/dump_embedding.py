import argparse
import random

import pickle
import numpy as np
import torch

from model import treebank_loader
from model.unsup_model import KMeansNCRFAE


def init_arg_parser():
    parser = argparse.ArgumentParser(description='Learning with NCRFAE')

    discourse_corpus = ('scidtb', 'rstdt')
    parser.add_argument('--corpus', type=str, default='scidtb', choices=discourse_corpus + ('wsj',))
    parser.add_argument('--gpu', type=int, default=0, help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size (10)')
    parser.add_argument('--hidden', type=int, default=50, help='hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout ratio')
    parser.add_argument('--epoch', type=int, default=100, help='maximum epoch number')
    parser.add_argument('--layers', type=int, default=1, help='number of LSTM layers')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.0, help='decay ratio of learning rate')
    parser.add_argument('--update', choices=['sgd', 'adadelta', 'adam'], default='adadelta', help='optimization method')
    parser.add_argument('--l2reg', type=float, default=0.001, help='L2 weight decay')
    parser.add_argument('--activation_function', choices=['sigmoid', 'tanh', 'relu', 'gelu'], default='relu')
    parser.add_argument('--encode_method', type=str, choices=['edu', 'minus', 'avg_pooling', 'max_pooling',
                                                              'mean', 'endpoint', 'diffsum', 'coherent', 'attention'],
                        default='coherent')
    parser.add_argument('--finetune', type=bool, default=False)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--norm', type=bool, default=True)
    parser.add_argument('--init_epoch', type=int, default=5)
    parser.add_argument('--kcluster', type=int, default=50)
    parser.add_argument('--word_embed', type=bool, default=False)
    parser.add_argument('--pretrained_model', type=str, default='bart-base')
    parser.add_argument('--decoder_learn', type=bool, default=True)
    return parser


def filter_shorter_than(data, k):
    return list(filter(lambda x: len(x) <= k, data))


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


def get_embed(dataset, d):
    for instance in dataset:
        name = instance.raw.name
        tensor = instance.r_embed.detach().cpu().numpy()
        assert name not in d
        d[name] = tensor


def main(method):
    parser = init_arg_parser()
    args = parser.parse_args()

    # reload_method
    args.encode_method = method

    # random seed
    set_seed(args.random_seed)

    # args information
    corpus = args.corpus

    # data loading
    # train, dev, test = treebank_loader.load_scidtb_dataset()
    train, dev, test = treebank_loader.load_rstdt_dataset()

    # print(args)
    print('================')
    print('setting:')
    print('================')
    print('\tepoch={}'.format(args.epoch))
    print('\tbatch_size={}'.format(args.batch_size))
    print('\tlr={}'.format(args.lr))
    print('\tlr_decay={}'.format(args.lr_decay))
    print('\tdropout={}'.format(args.dropout))
    print('\tl2reg={}'.format(args.l2reg))
    print('\thidden={}'.format(args.hidden))
    print('\tactivation_function={}'.format(args.activation_function))
    print('\tencode_method={}'.format(args.encode_method))
    print('\trandom_seed={}'.format(args.random_seed))
    print('\tnorm={}'.format(args.norm))
    print('\tinit_epoch={}'.format(args.init_epoch))
    print('\tkcluster={}'.format(args.kcluster))
    print('\tword_embed={}'.format(args.word_embed))


    USE_GPU = False
    device = torch.device('cpu')
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
        USE_GPU = True



    model = KMeansNCRFAE(args)
    if USE_GPU:
        model.cuda()

    print("Before training:")
    model.prepare_embed(train)
    # model.prepare_embed(dev)
    model.prepare_embed(test)
    # model.calculate_kmeans(train_K)
    # model.kmeans_label_predict(train_K)
    # model.kmeans_label_predict(dev_K)
    # model.kmeans_label_predict(test_K)
    # model.init_embedding()
    to_dump = dict()
    get_embed(train, to_dump)
    # get_embed(dev, to_dump)
    get_embed(test, to_dump)

    # save
    path = './data/' + args.pretrained_model + '_rstdt_' + args.encode_method + '.pkl'
    pickle.dump(to_dump, open(path, 'wb'))

    # test load
    tmp = pickle.load(open(path, 'rb'))
    print(len(tmp))


if __name__ == '__main__':
    methods = ['edu', 'minus', 'avg_pooling', 'max_pooling', 'mean', 'endpoint', 'diffsum', 'coherent']
    for method in methods:
        main(method)
