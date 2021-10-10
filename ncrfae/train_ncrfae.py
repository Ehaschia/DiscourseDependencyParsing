import argparse
import random

from torch import optim

from model import treebank_loader
from model.cs_ncrfae import *
from model.dep_dataset import DiscoursePreprocessor
from model.dep_evalutator import evaluate_uas, evaluate


def init_arg_parser():
    parser = argparse.ArgumentParser(description='Learning with NCRFAE')

    # pascal_corpus = ("arabic", "basque", "childes", "czech", "danish",
    #                  "dutch", "english", "portuguese", "slovene", "swedish")
    discourse_corpus = ('scidtb', 'rstdt')
    parser.add_argument('--corpus', type=str, default='scidtb', choices=discourse_corpus + ('wsj',))
    parser.add_argument('--gpu', type=int, default=0, help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size (10)')
    parser.add_argument('--hidden', type=int, default=100, help='hidden dimension')
    parser.add_argument('--recons_dim', type=int, default=10, help='reconstruction hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout ratio')
    parser.add_argument('--epoch', type=int, default=100, help='maximum epoch number')
    parser.add_argument('--case_insensitive', action='store_true', default=False, help='case-insensitive or not')
    parser.add_argument('--embedding_dim', type=int, default=768, help='dimension for word embedding')
    parser.add_argument('--layers', type=int, default=1, help='number of LSTM layers')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.0, help='decay ratio of learning rate')
    parser.add_argument('--update', choices=['sgd', 'adadelta'], default='adadelta', help='optimization method')
    parser.add_argument('--l2reg', type=float, default=0.0, help='L2 weight decay')
    parser.add_argument('--prior', type=float, default=1.0, help='')
    parser.add_argument('--activation_function', choices=['sigmoid', 'tanh', 'relu', 'gelu'], default='gelu')
    parser.add_argument('--multi_root', action='store_true', default=False,
                        help='multiple roots in dependency tree or not')
    parser.add_argument('--max_dependency_len', type=int, default=100, help='max dependency length in dependency tree')
    parser.add_argument('--root_length_constraint', action='store_true', default=False, help='')
    parser.add_argument('--mini_count', type=float, default=5, help='thresholds to replace rare words with <unk>')
    parser.add_argument('--patience', type=int, default=15, help='patience for early stop')
    parser.add_argument('--least_iters', type=int, default=50, help='at least train how many epochs before stop')

    # for initialize
    parser.add_argument('--rand_embedding', action='store_true', help='random initialize word embedding')

    # for saving intermediate results
    parser.add_argument('--checkpoint', default='./checkpoint/', help='path to checkpoint prefix')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch idx')
    return parser


def filter_shorter_than(data, k):
    return list(filter(lambda x: len(x) <= k, data))


def main():
    random.seed(42)
    torch.manual_seed(42)

    parser = init_arg_parser()
    args = parser.parse_args()

    # args information
    corpus = args.corpus

    # data loading
    if corpus != "wsj":
        train, dev, test = treebank_loader.load_scidtb_dataset()
    else:
        train, dev, test = treebank_loader.load_wsj_dataset()

    # print(args)
    print('================')
    print('setting:')
    print('================')
    print('\tis_multi_root={}'.format(args.multi_root))
    print('\tmax_dependency_len={}'.format(args.max_dependency_len))
    print('\tbatch_size={}'.format(args.batch_size))
    print('\tlr={}'.format(args.lr))
    print('\tlr_decay={}'.format(args.lr_decay))
    print('\tdropout={}'.format(args.dropout))
    print('\tl2reg={}'.format(args.l2reg))
    print('\tmini_count={}'.format(args.mini_count))
    print('\tprior={}'.format(args.prior))

    USE_GPU = False
    device = torch.device('cpu')
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
        USE_GPU = True

    print("")

    print("Pre-processing")

    token_extractor = lambda x: getattr(x, 'words')
    preprocessor = DiscoursePreprocessor(token_extractor, 'bert', train, dev, test, use_gpu=USE_GPU)

    K = 100

    train_K = filter_shorter_than(train, K + 1)
    dev_K = filter_shorter_than(dev, K + 1)
    test_K = filter_shorter_than(test, K + 1)

    # debug = [train_K[0]]
    # debug[0].ftags = debug[0].ftags[:3]
    # debug[0].heads = debug[0].heads[:3]
    # debug[0].length = 3
    # debug[0].sbnds = [(0, 1)]
    # debug[0].words = debug[0].words[:3]

    print("corpus={}".format(corpus))
    print("\tlength(train-{})={}".format(K, len(train_K)))
    print("\tlength(dev-{})={}".format(K, len(dev_K)))
    print("\tlength(test-{})={}".format(K, len(test_K)))

    print("")

    print('Building model')
    token_set_size = preprocessor.token_num
    # #
    model = CSNCRFAE_lstm_m(preprocessor=preprocessor,
                          tagset_size=token_set_size,
                          embedding_dim=args.embedding_dim,
                          hidden_dim=args.hidden,
                          rnn_layers=1,
                          dropout_ratio=args.dropout,
                          prior_weight=args.prior,
                          recons_dim=args.recons_dim,
                          act_func=args.activation_function,
                          use_gpu=USE_GPU,
                          is_multi_root=args.multi_root,
                          max_dependency_len=args.max_dependency_len,
                          length_constraint_on_root=args.root_length_constraint)
    # model = JAP(preprocessor, args.embedding_dim, 10, 10, args.hidden, args.hidden, device)
    if USE_GPU:
        model.cuda()

    # print('Initialization')
    # model.init()
    # kmem = KmEM(model, preprocessor, train_K, use_gpu=USE_GPU)
    # kmem.step()

    print("")

    gd_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2reg)

    # Train
    print("Before training:")
    print("\tObj: {}".format(evaluate(model, train_K)))
    # print('\tTRAIN-{}-UAS: {}'.format(K, evaluate_uas(model, train_K)))
    max_dev = evaluate_uas(model, dev_K)
    print('\tDEV-{}-UAS: {}'.format(K, max_dev))
    final = (0, max_dev, 0.0)
    for epoch in range(args.epoch):
        print("EPOCH-{}".format(epoch))
        model.train()
        random.shuffle(train_K)
        batches = utility.construct_batches(train_K, args.batch_size)
        random.shuffle(batches)

        for i, batch in enumerate(batches):
            sub_batches = utility.construct_batches_by_length(batch, args.batch_size)
            loss = 0
            for sub_batch in sub_batches:
                # loss += model(sub_batch)
                loss += len(sub_batch) * model(sub_batch)
            loss = loss / len(batch)
            # loss = loss
            loss.backward()
            gd_optimizer.step()
            gd_optimizer.zero_grad()

        print("\tAfter GD: {}".format(evaluate(model, train_K)))
        # print('\tTRAIN-{}-UAS: {}'.format(K, evaluate_uas(model, train_K)))
        current_dev = evaluate_uas(model, dev_K)
        print('\tDEV-{}-UAS: {}'.format(K, current_dev))

        # for _ in range(1):
        #     # EM
        #     em_optimizer = HardEM(model, preprocessor, train_K, use_gpu=USE_GPU)
        #     em_optimizer.step()
        #
        # print("\tAfter EM: {}".format(evaluate(model, train_K)))
        # print('\tTRAIN-{}-UAS: {}'.format(K, evaluate_uas(model, train_K)))
        # print('\tDEV-{}-UAS: {}'.format(K, evaluate_uas(model, dev_K)))
        #
        # # Test
        if current_dev > max_dev:
            test_uas = evaluate_uas(model, test_K)
            max_dev = current_dev
            print('\tTEST-{}-UAS: {}'.format(K, test_uas))
            final = (epoch, max_dev, test_uas)
        # print('TEST-all-UAS: {}'.format(evaluate_uas(model, test)))
    print('FINAL-BEST-EPOCH-{}-DEV: {} TEST: {}'.format(*final))

if __name__ == '__main__':
    main()
