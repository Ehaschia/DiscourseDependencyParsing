import os.path

from discourse_loader.scidtb import read_scidtb
from discourse_loader.rstdt import read_rstdt
from model.definition import PROJECT_ROOT
from model.dep_dataset import PascalSentence, WSJSentence, SCIDTBSentence
from utils import utils, DataInstance

def _process_MST_file(filename):
    # map the corpus file to instances
    with open(filename, 'r') as f:
        lines = list(map(lambda x: x.strip(), f.readlines()))

        sents = []

        for i in range(0, len(lines), 5):
            words = lines[i].split('\t')
            fpos = lines[i + 1].split('\t')
            heads = list(map(int, lines[i + 3].split('\t')))

            inst = (words, fpos, heads)
            sents.append(inst)

        return sents


def load_pascal_dataset(corpus_name, path_prefix='data/pascal'):
    """

    :param corpus_name
    :param path_prefix: the relative path to the PROJECT_ROOT where the corpus files store
    :return: (training_set, dev_set, test_set), each one is a list of sentences
    """
    train_file, dev_file, test_file = _pascal_files_path(corpus_name, path_prefix)
    train_set = list(map(lambda x: PascalSentence(*x), _process_CONLL_file(train_file)))
    dev_set = list(map(lambda x: PascalSentence(*x), _process_CONLL_file(dev_file)))
    test_set = list(map(lambda x: PascalSentence(*x), _process_CONLL_file(test_file)))
    return train_set, dev_set, test_set


def load_wsj_dataset(path_prefix='data/wsj'):
    train_file, dev_file, test_file = list(
        map(lambda x: os.path.join(PROJECT_ROOT, path_prefix, x), ('wsj10-train.txt', 'wsj10-dev.txt', 'wsj10-test.txt'))
    )

    train_set = list(map(lambda x: WSJSentence(*x), _process_MST_file(train_file)))
    dev_set = list(map(lambda x: WSJSentence(*x), _process_MST_file(dev_file)))
    test_set = list(map(lambda x: WSJSentence(*x), _process_MST_file(test_file)))

    return train_set, dev_set, test_set

def load_scidtb_dataset(path_prefix='data', args=None):
    # vocab_word = utils.read_vocab(os.path.join(path_prefix, "scidtb-vocab", "words.vocab.txt"))
    # vocab_postag = utils.read_vocab(os.path.join(path_prefix, "scidtb-vocab", "postags.vocab.txt"))
    # vocab_deprel = utils.read_vocab(os.path.join(path_prefix, "scidtb-vocab", "deprels.vocab.txt"))
    vocab_relation = utils.read_vocab(os.path.join(path_prefix, "scidtb-vocab", "relations.coarse.vocab.txt"))

    if args is not None:
        args.update({'n_rels': len(vocab_relation)})
    train_dataset = read_scidtb("train", "", relation_level="coarse-grained")
    dev_dataset = read_scidtb("dev", "gold", relation_level="coarse-grained")
    test_dataset = read_scidtb("test", "gold", relation_level="coarse-grained")

    def get_scidtbsentence(sample: DataInstance):
        # head_word = [ins[0] for ins in sample.edus_head][1:]
        head_edu = sample.edus[1:]
        head_pos = [ins[1] for ins in sample.edus_head][1:]
        heads = [ins[0] for ins in sample.arcs]
        rels = [vocab_relation[ins[2]] for ins in sample.arcs]
        return SCIDTBSentence(head_edu, head_pos, heads=heads, sbnds=sample.sbnds, raw=sample, rels=rels)

    train_set = list(map(get_scidtbsentence, train_dataset))
    dev_set = list(map(get_scidtbsentence, dev_dataset))
    test_set = list(map(get_scidtbsentence, test_dataset))
    return train_set, dev_set, test_set

def load_rstdt_dataset(path_prefix='data', args=None):
    vocab_relation = utils.read_vocab(os.path.join(path_prefix, "rstdt-vocab", "relations.coarse.vocab.txt"))

    if args is not None:
        args.update({'n_rels': len(vocab_relation)})

    train_dataset = read_rstdt("train", relation_level="coarse-grained", with_root=True)

    test_dataset = read_rstdt("test", relation_level="coarse-grained", with_root=True)

    def get_scidtbsentence(sample: DataInstance):
        # head_word = [ins[0] for ins in sample.edus_head][1:]
        head_edu = sample.edus[1:]
        head_pos = [ins[1] for ins in sample.edus_head][1:]
        heads = [ins[0] for ins in sample.arcs]
        rels = [vocab_relation[ins[2]] for ins in sample.arcs]
        return SCIDTBSentence(head_edu, head_pos, heads=heads, sbnds=sample.sbnds, raw=sample, rels=rels)

    train_set = list(map(get_scidtbsentence, train_dataset))
    dev_set = None
    test_set = list(map(get_scidtbsentence, test_dataset))
    return train_set, dev_set, test_set