import argparse
import random
from typing import List, Tuple
import os
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR

from discourse_loader.alphabet import Alphabet
from discourse_loader.embed import load_embedding_dict
from model import treebank_loader, utility
# from model.sup_model import *
from model.unsup_model import NCRFAE
from model.lstm_biaffine_model import *
from datetime import datetime
import numpy as np

import utils
from collections import defaultdict, OrderedDict

@torch.no_grad()
def evaluate_acc(model: NCRFAE, dataset: List[utils.SCIDTBSentence]) -> Tuple[float, float]:
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
def evaluate_eisner_acc(model: NCRFAE, dataset: List[utils.SCIDTBSentence]) -> Tuple[float, float]:
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

def create_alphabets(alphabet_directory, train, data=None, max_vocabulary_size=100000,
                     embedd_dict=None, min_occurrence=1):

    def expand_vocab():
        vocab_set = set(vocab_list)
        for adata in data:
            # logger.info("Processing data: %s" % data_path)
            for instance in adata:
                for edu in instance.raw.edus:
                    for word in edu:
                        if word not in vocab_set and (word in embedd_dict or word.lower() in embedd_dict):
                            vocab_set.add(word)
                            vocab_list.append(word)

    word_alphabet = Alphabet('word', defualt_value=False, singleton=True)
    vocab = defaultdict(int)
    vocab['<_PAD>'] = 10000
    vocab['<_UNK>'] = 10000
    if not os.path.isdir(alphabet_directory):
        for instance in train:
            for edus in instance.raw.edus:
                for word in edus:
                    vocab[word] += 1

        # collect singletons
        singletons = set([word for word, count in vocab.items() if count <= min_occurrence])

        # if a singleton is in pretrained embedding dict, set the count to min_occur + c
        if embedd_dict is not None:
            assert isinstance(embedd_dict, utils.OrderedDict)
            for word in vocab.keys():
                if word in embedd_dict or word.lower() in embedd_dict:
                    vocab[word] += min_occurrence

        vocab_list = sorted(vocab, key=vocab.get, reverse=True)

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

        if data is not None and embedd_dict is not None:
            expand_vocab()

        for word in vocab_list:
            word_alphabet.add(word)
            if word in singletons:
                word_alphabet.add_singleton(word_alphabet.get_index(word))

        word_alphabet.save(alphabet_directory)

    else:
        word_alphabet.load(alphabet_directory)

    word_alphabet.close()
    # logger.info("Word Alphabet Size (Singleton): %d (%d)" % (word_alphabet.size(), word_alphabet.singleton_size()))

    return word_alphabet

def dataset_to_idx(dataset, alphabet):
    for instance in dataset:
        # word_list = [word for edus in instance.raw.edus for word in edus]
        idx_list = [[alphabet.get_index(word) for word in edu] for edu in instance.raw.edus]
        instance.idx = idx_list
        size_cnt = 0
        begin_idx = []
        end_idx = []
        for idx in idx_list:
            begin_idx.append(size_cnt)
            size_cnt += len(idx)
            end_idx.append(size_cnt-1)
        instance.begin_pos = begin_idx
        instance.end_pos = end_idx

def construct_word_embedding_table(embedd_dict, embedd_dim, word_alphabet):
    scale = np.sqrt(3.0 / embedd_dim)
    table = np.empty([word_alphabet.size(), embedd_dim], dtype=np.float32)
    oov = 0
    for word, index in word_alphabet.items():
        if word in embedd_dict:
            embedding = embedd_dict[word]
        elif word.lower() in embedd_dict:
            embedding = embedd_dict[word.lower()]
        else:
            embedding = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
            oov += 1
        table[index, :] = embedding
    print('oov: %d' % oov)
    return torch.from_numpy(table)

@torch.no_grad()
def evaluate(model, dataset):
    model.eval()
    loss = model(dataset)
    return loss

def init_parameter():
    parser = argparse.ArgumentParser(description='Learning with NCRFAE')

    discourse_corpus = ('scidtb', 'rstdt')
    parser.add_argument('--corpus', type=str, default='scidtb', choices=discourse_corpus)
    parser.add_argument('--gpu', type=int, default=0, help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--conf', '-c', default='configs/lstm_config.ini', help='path to config file')
    parser.add_argument('--log_name', default=f'output/lstm_{utils.get_current_time()}')

    args = parser.parse_args()
    args = utils.Config(args.conf).update(vars(args))
    return args

def main():
    args = init_parameter()

    # set log
    utils.set_logger(args.log_name)
    utils.writelog(str(args))

    # random seed
    utils.set_seed(args.random_seed)

    # args information
    corpus = args.corpus
    utils.writelog("Pre-processing")

    # data loading
    if corpus == 'scidtb':
        utils.writelog('\t Load {}'.format(corpus))
        # update n_rels in loader
        train, dev, test = treebank_loader.load_scidtb_dataset(args=args)

    elif corpus == 'rstdt':
        utils.writelog('\t Load {}'.format(corpus))
        # update n_rels in loader
        train, _, test = treebank_loader.load_rstdt_dataset(args=args)
        dev = []
    else:
        raise NotImplementedError("Error Corpus!")

    embedd_dict, embedd_dim = load_embedding_dict(args.embedding, args.embedding_path)
    args.embedd_dim = embedd_dim
    word_alphabet = create_alphabets('./data/' + corpus + '/alphabet/', train, [dev, test], embedd_dict=embedd_dict)
    embedding = construct_word_embedding_table(embedd_dict, embedd_dim, word_alphabet)
    dataset_to_idx(train, word_alphabet)
    dataset_to_idx(dev, word_alphabet)
    dataset_to_idx(test, word_alphabet)

    # construct

    USE_GPU = False
    device = torch.device('cpu')
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
        USE_GPU = True

    utils.writelog('Building model')
    # model = NCRF(args)
    if args.model == 'biaffine':
        model = DocLSTMBiaffine(args, embedding)
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
