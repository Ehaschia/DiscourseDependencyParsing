import pickle
import re
import os
import random
from collections import Counter, namedtuple, OrderedDict
from itertools import groupby

import utils
from models.context_sensitive_encoder import CSEncoder
from utils.common import *
from utils.utils import make_sure_dir_exists, DataInstance
from typing import Iterable, Dict, List, Optional, Union, Iterator, Callable, Tuple, Generator, Type, Any
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence

from models.functions import make_same_sent_map

from sklearn.cluster import KMeans

class Vocab:
    def __init__(self, stoi: Dict[str, int], itos: List[str], freeze: bool, unk: str, pad: str):
        self.freeze = False
        self.stoi = stoi or dict()
        self.itos = itos or list()
        self.unk = self.add_one(unk) if unk else None
        self.pad = self.add_one(pad) if pad else None

        self.freeze = freeze

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.itos[item]
        else:
            v = self.stoi.get(item, self.unk)
            if v == self.unk:
                if not self.freeze:
                    i = self.add_one(item)
                    return self.itos[i]
                if self.unk is None:
                    raise KeyError(f"bad key {str(item)}")
            return v

    def __len__(self):
        return len(self.itos)

    def add_one(self, x: str) -> int:
        if x not in self.stoi:
            self.stoi[x] = len(self.itos)
            self.itos.append(x)
        return self.stoi[x]

    def get_size_without_special(self):
        num_special = int(self.unk is not None) + int(self.pad is not None)
        return len(self.itos) - num_special

    @staticmethod
    def filte_counter(c: Counter, threshold: int):
        return [x for x, num in c.items() if num >= threshold]

    @staticmethod
    def from_dict(d: Dict[str, int], freeze: bool = True, unk: Optional[str] = None, pad: Optional[str] = None):
        itos = []
        for x, idx in d.items():
            if idx > len(itos):
                itos.extend([None] * (idx - len(itos)))
            itos.append(x)
        vocab = Vocab(d, itos, freeze, unk, pad)
        return vocab

    @staticmethod
    def from_list(a: List[str], freeze: bool = True, unk: Optional[str] = None, pad: Optional[str] = None):
        stoi = {v: k for k, v in enumerate(a)}
        vocab = Vocab(stoi, a, freeze, unk, pad)
        return vocab

    @staticmethod
    def empty_vocab(freeze: bool = False, unk: Optional[str] = None, pad: Optional[str] = None):
        vocab = Vocab(dict(), list(), freeze, unk, pad)
        return vocab


UD_POS = Vocab.from_list('ADV NOUN ADP NUM SCONJ PROPN DET SYM INTJ PART PRON VERB X AUX CONJ ADJ'.split())
WSJ_POS = Vocab.from_list('RB NNP NN WRB NNS VBN UH JJ VB FW CD NNPS PRP VBD IN DT VBZ VBP '
                          'VBG RP $ WP RBR PRP$ CC JJS MD JJR POS EX TO WDT PDT RBS'.split())  # diff bllip: no WP$
BLLIP_POS = Vocab.from_list('RB NNP NN WRB NNS VBN UH JJ VB FW CD NNPS PRP VBD IN DT VBZ VBP ' 'VBG RP $ WP RBR PRP$ CC JJS MD JJR POS EX TO WDT PDT RBS WP$'.split())
FLOW_PREDICTED_POS = Vocab.from_list([
    '20', '23', '27', '40', '31', '36', '2', '6', '30', '43', '0', '10', '42', '3', '32', '1', '44', '16', '19', '22', '5', '21', '33', '38', '11', '13', '8', '7', '26', '28',
    '25', '9', '37', '29', '17', '18', '41', '14', '34'
])


class ConllEntry:
    __slots__ = ("id", "form", "lemma", "pos", "cpos", "feats", "parent_id", "relation", "deps", "misc", "norm", "pred_parent_id", "pred_relation")

    def __init__(self,
                 id: Union[int, str],
                 form: str,
                 lemma: str,
                 pos: str,
                 cpos: str,
                 feats: str,
                 parent_id: Union[int, str],
                 relation: str = '-',
                 deps: str = '-',
                 misc: str = '-'):
        self.id = int(id)
        self.form = form
        self.lemma = lemma
        self.pos = pos
        self.cpos = cpos
        self.feats = feats
        self.parent_id = int(parent_id) if str.isdecimal(parent_id) else -1
        self.relation = relation
        self.deps = deps
        self.misc = misc

        self.norm = self.normalize()
        self.pred_parent_id = None
        self.pred_relation = None

        self.pos = self.pos.upper()
        self.cpos = self.cpos.upper()

    def normalize(self):
        if self.form == 'NUM':
            return 'NUM'
        if all(map(lambda x: str.isdigit(x) or x in '+-.,*/:', self.form)):
            return 'NUM'
        return re.sub(r'\\/', '/', self.form.lower())

    def __repr__(self):
        return f'{self.norm}'

    def __str__(self):
        return f'{self.id}\t{self.form}\t{self.lemma}\t{self.pos}\t{self.cpos}\t{self.feats}\t{self.parent_id}'


class ConllInstance:
    def __init__(self, id: int, entries: List[ConllEntry], dataset: Optional['ConllDataset'] = None):
        self.entries = entries
        self.id = id
        self.ds = dataset

        self._pos_np = None
        self._norm_np = None

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __repr__(self):
        return f'ConllInstance(id={self.id}, len={len(self)})'

    def __str__(self):
        return f'ConllInstance(str={" ".join(self.__getattr__("norm"))})'

    def __hash__(self):
        return hash('\t'.join(self.form))

    def __getattr__(self, item):
        return [getattr(e, item) for e in self.entries]

    @property
    def pos_np(self) -> Optional[np.ndarray]:
        if self.ds.pos_vocab is None:
            return None
        if self._pos_np is None:
            self._pos_np = npasarray(list(map(self.ds.pos_vocab.stoi.__getitem__, self.pos)))
        return self._pos_np

    @property
    def norm_np(self) -> Optional[np.ndarray]:
        if self.ds.word_vocab is None:
            return None
        if self._norm_np is None:
            self._norm_np = npasarray(list(map(self.ds.word_vocab.__getitem__, self.norm)))
        return self._norm_np

    def get_raw(self) -> str:
        raw_entries = []
        if self.ds is not None and self.ds.compact_mode:
            for e in self.entries:
                raw_entries.append(f'{e.id}\t{e.form}\t{e.pos}\t{e.parent_id}')
        else:
            for e in self.entries:
                raw_entries.append(str(e))
        return '\n'.join(raw_entries)

    def remove_entry(self, id: int, reset_parent: bool = False):
        """

        :param id: entry.id, start from 1
        :param reset_parent: set parent id to -1 when find bad arc
        :return:
        """
        assert (0 < id <= len(self)), "out of bound"

        for e in self.entries:
            if e.id == id:
                continue
            if e.parent_id == id:
                if reset_parent:
                    e.parent_id = -1
                else:
                    raise ValueError(f"remove {id} will make bad arc")
            if e.parent_id > id:
                e.parent_id -= 1
            if e.id > id:
                e.id -= 1
        self.entries.pop(id - 1)

ConllDatasetBatchData = namedtuple('ConllDatasetBatchData', ('id_array', 'pos_array', 'word_array', 'len_array'))

class ConllDataset(Dataset):
    def __init__(self, path: Union[str, List[str]], word_vocab: Optional[Vocab] = None, pos_vocab: Optional[Vocab] = None, cache: Optional[str] = None):
        # cache: only cache instances

        super().__init__()
        self.path = path if isinstance(path, list) else str(path).strip().split(';')
        self.word_vocab = word_vocab
        self.pos_vocab = pos_vocab
        self.compact_mode = None
        self.is_sorted = False

        if cache is not None and os.path.exists(cache):
            self.instances = pickle.load(open(cache, 'rb'))
        else:
            self.instances = []
            for path in self.path:
                self.instances.extend(self.read_file(path, self))
            self.build_id()
        if cache is not None and not os.path.exists(cache):
            make_sure_dir_exists(cache)
            pickle.dump(self.instances, open(cache, 'wb'))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def __iter__(self):
        return iter(self.instances)

    def build_word_vocab(self, min_freq: int = 1, max_size: int = 1000000, unk='<UNK>', pad='<PAD>') -> Vocab:
        c = Counter()
        for i in self.instances:
            c.update(i.norm)
        c = Vocab.filte_counter(c, min_freq)[:max_size]
        self.word_vocab = Vocab.from_list(c, unk=unk, pad=pad)
        return self.word_vocab

    def build_pos_vocab(self) -> Vocab:
        s = set()
        for i in self.instances:
            s.update(i.pos)
        self.pos_vocab = Vocab.from_list(list(s))
        return self.pos_vocab

    def build_id(self):
        for id, instance in enumerate(self.instances):
            instance.id = id

    def sort(self, key: Optional[Callable] = None):
        sorted(self.instances, key=key or (lambda x: len(x)))
        self.is_sorted = True
        self.build_id()

    def get_dataloader(self, same_len: bool, batch_size: int, drop_last: bool, shuffle: int, num_workers: int, min_len: int = 1, max_len: int = 10000):
        sampler = LengthBucketSampler if same_len else BasicSampler

        return DataLoader(dataset=self,
                          num_workers=num_workers,
                          pin_memory=True,
                          batch_sampler=sampler(self, batch_size, drop_last, shuffle, min_len, max_len),
                          collate_fn=self.collect_fn)

    def get_all_len(self):
        return torch.tensor([len(i) for i in self.instances])

    def read_file(self, path: str, dataset: Optional['ConllDataset'] = None) -> Iterator[ConllInstance]:
        tokens = []
        open_func = open
        with open_func(path) as f:
            for line in f:
                line = line.strip()

                if line == '':
                    if tokens:
                        yield ConllInstance(0, tokens, dataset)
                    tokens = []
                elif line[0] == '#':
                    continue
                else:
                    token = line.split('\t')
                    if len(token) == 4:
                        if self.compact_mode is None:
                            self.compact_mode = True
                            print('compact mode')
                        elif self.compact_mode is False:
                            raise ValueError("comflicted format style")
                        tokens.append(ConllEntry(token[0], token[1], '-', token[2], '-', '-', token[3]))
                        continue

                    if len(token) <= 10:
                        if self.compact_mode is None:
                            self.compact_mode = False
                        elif self.compact_mode is True:
                            raise ValueError("comflicted format style")
                        tokens.append(ConllEntry(*token))
                    else:
                        print(f'skip line: "{line}"')
            if tokens:
                yield ConllInstance(0, tokens, dataset)

    @staticmethod
    def collect_fn(batch_raw_data: List[ConllInstance]):
        id_array = torch.tensor([d.id for d in batch_raw_data])
        len_array = torch.tensor([len(d) for d in batch_raw_data])

        pos_np = [d.pos_np for d in batch_raw_data]
        if pos_np[0] is None:
            pos_array = None
        else:
            pos_array = pad_sequence(list(map(torch.tensor, pos_np)), True, 0)

        norm_np = [d.norm_np for d in batch_raw_data]
        if norm_np[0] is None:
            word_array = None
        else:
            pad = batch_raw_data[0].ds.word_vocab.pad
            word_array = pad_sequence(list(map(torch.tensor, norm_np)), True, pad)

        return ConllDatasetBatchData(id_array, pos_array, word_array, len_array)

    def unk_count(self) -> Tuple[int, int, float]:
        if self.word_vocab is None:
            return -1, -1, 0.
        unk, total = 0, 0
        unk_words = set()
        for instance in self.instances:
            unk += np.sum(instance.norm_np == self.word_vocab.unk)
            unk_words.update(filter(lambda x: x[1] == self.word_vocab.unk, zip(instance.norm, instance.norm_np)))
            total += len(instance)

        return unk, len(unk_words), unk / total

    def report(self):

        if self.pos_vocab:
            pos_str = f"pos vocab has {len(self.pos_vocab)} tag."
        else:
            pos_str = "pos vocab is not defined."
        if self.word_vocab:
            all_size, word_size = len(self.word_vocab), self.word_vocab.get_size_without_special()
            word_str = f"word vocab has {word_size} words and {all_size - word_size} special token."
        else:
            word_str = "word vocab is not defined."
        unk, unk_word, unk_rate = self.unk_count()
        lens = self.get_all_len().cpu().tolist()
        len_counter = Counter(lens)
        utils.ex.logger.info(f"\n"
                             f"path: {self.path}\n"
                             f"there are {len(self)} instances.\n"
                             f"len: {len_counter}\n"
                             f"{pos_str}\n"
                             f"{word_str}\n"
                             f'there are {unk_word} oov words and show {unk}({unk_rate * 100:.2f}%) times')

    def save(self, path: str):
        with open(path, 'w') as f:
            for instance in self.instances:
                f.write(instance.get_raw())
                f.write('\n\n')


ConllDatasetWithEmbDataType = namedtuple('ConllDatasetWithEmbDataType', ('id_array', 'pos_array', 'word_array', 'len_array', 'context_emb', 'noncontext_emb'))


class ConllDatasetWithEmb(ConllDataset):
    def __init__(self, emb_path: Union[List[str], str], layers: Tuple[int], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.emb_path = emb_path if isinstance(emb_path, list) else emb_path.strip().split(';')
        self.emb = []
        for emb_path in self.emb_path:
            with open(emb_path, 'rb') as f:
                one_part = pickle.load(f)
            assert isinstance(one_part, Iterable) and isinstance(one_part[0], np.ndarray) and one_part[0].ndim == 3
            self.emb.extend(one_part)
        self.emb = [torch.tensor(e, dtype=torch.float) for e in self.emb]
        self.emb_dim = self.emb[0].shape[-1]
        self.layers = layers

    def sort(self, key: Optional[Callable] = None):
        raise NotImplementedError

    def get_dataloader(self, same_len: bool, batch_size: int, drop_last: bool, shuffle: int, num_workers: int, min_len=1, max_len=10000):
        emb = self.emb
        layers = self.layers

        def collect_fn(batch_raw_data: List[ConllInstance]):
            intermediate_data = ConllDataset.collect_fn(batch_raw_data)
            batch_size = len(intermediate_data[0])

            context_emb_np = [emb[intermediate_data.id_array[i].item()][layers[0]] for i in range(batch_size)]
            context_emb_array = pad_sequence(context_emb_np, True, 0)
            if len(self.layers) > 1:
                noncontext_emb_np = [emb[intermediate_data.id_array[i].item()][layers[1]] for i in range(batch_size)]
                noncontext_emb_array = pad_sequence(noncontext_emb_np, True, 0)
            else:
                noncontext_emb_array = None

            return ConllDatasetWithEmbDataType(*intermediate_data, context_emb_array, noncontext_emb_array)

        sampler = LengthBucketSampler if same_len else BasicSampler
        return DataLoader(dataset=self,
                          num_workers=num_workers,
                          pin_memory=True,
                          batch_sampler=sampler(self, batch_size, drop_last, shuffle, min_len, max_len),
                          collate_fn=collect_fn)

    @staticmethod
    def collect_fn(batch_raw_data: List[ConllInstance]):
        raise NotImplementedError('You should not use this func')


class LengthBucketSampler(Sampler):
    NOSHUFFLE = 0
    IN_BUCKET = 1
    COMPLETE = 2

    def __init__(self, data_source: ConllDataset, batch_size: int, drop_last: bool, shuffle: int, min_len: int, max_len: int):
        super().__init__(data_source)

        sorted_idx = list(range(len(data_source)))
        if not data_source.is_sorted:
            sorted_idx.sort(key=lambda x: len(data_source[x]))
        self.groups = [list(g) for l, g in groupby(sorted_idx, key=lambda x: len(data_source[x])) if min_len <= l <= max_len]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        self.plan = []
        if self.shuffle == self.NOSHUFFLE:
            self.build_plan()

    def build_plan(self):
        self.plan.clear()
        for group in self.groups:
            for i in range(0, len(group), self.batch_size):
                if i + self.batch_size > len(group) and self.drop_last:
                    break
                batch = group[i:i + self.batch_size]
                if self.shuffle >= self.IN_BUCKET:
                    random.shuffle(batch)
                self.plan.append(batch)

        if self.shuffle >= self.COMPLETE:
            random.shuffle(self.plan)

    def __iter__(self):
        if self.shuffle > self.NOSHUFFLE:
            self.build_plan()
        return iter(self.plan)


class BasicSampler(Sampler):
    NOSHUFFLE = 0
    COMPLETE = 1

    def __init__(self, data_source: Dataset, batch_size: int, drop_last: bool, shuffle: int, min_len: int, max_len: int):
        super().__init__(data_source)

        self.idx = list(filter(lambda x: min_len <= len(data_source[x]) <= max_len, range(len(data_source))))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        self.plan = []
        if self.shuffle == self.NOSHUFFLE:
            self.build_plan()

    def build_plan(self):
        self.plan.clear()
        if self.shuffle >= self.COMPLETE:
            random.shuffle(self.idx)
        for i in range(0, len(self.idx), self.batch_size):
            if i + self.batch_size > len(self.idx) and self.drop_last:
                if i == 0:
                    utils.ex.logger.warning(f"too few data to build any batch with drop_last=True, " f"require {self.batch_size} but only have {len(self.idx)}")
                break
            batch = self.idx[i:i + self.batch_size]
            self.plan.append(batch)

    def __iter__(self):
        if self.shuffle > self.NOSHUFFLE:
            self.build_plan()
        return iter(self.plan)


def prefetcher(loader: Union[DataLoader, Iterator], rows: Optional[List[int]] = None, keep: bool = False) -> Generator:
    """prefetch next batch and move to gpu

    if device == 'cpu', this function will have no use, just for clean code.
    """
    stream = torch.cuda.Stream()
    next_batch = None
    type_ = None
    if not isinstance(loader, Iterator):
        loader = iter(loader)

    def prefetch():
        nonlocal next_batch, type_, rows
        try:
            next_batch = next(loader)
        except StopIteration:
            next_batch = None
            return

        if type_ is None and next_batch is not None:
            if keep:
                fields = list(next_batch._fields)
                for row in rows:
                    fields.append(fields[row] + '_cuda')
                type_ = namedtuple(type(next_batch).__name__ + '_cuda', fields)
            else:
                type_ = type(next_batch)
            rows = rows or list(range(len(next_batch)))
            for row in rows:
                if next_batch[row] is not None and not next_batch[row].is_pinned():
                    utils.ex.logger.warning(f'Prefetcher is used but not working ' f'because one_batch[{row}].is_pinned()=False')

        with torch.cuda.stream(stream):
            next_batch = list(next_batch)
            for row in rows:
                if isinstance(next_batch[row], torch.Tensor):
                    if keep:
                        next_batch.append(next_batch[row].cuda(non_blocking=True))
                    else:
                        next_batch[row] = next_batch[row].cuda(non_blocking=True)
            next_batch = type_._make(next_batch)

    prefetch()
    while next_batch is not None:
        _batch = next_batch
        prefetch()
        stream.synchronize()
        yield _batch


def cuda_array_guard(loader):
    _type = None
    for one_batch in loader:
        if _type is None:
            _type = type(one_batch)
        one_batch = list(one_batch)
        for idx in range(len(one_batch)):
            if isinstance(one_batch[idx], torch.Tensor):
                one_batch[idx] = one_batch[idx].to('cuda')
        yield _type(*one_batch)

def array_guard(loader):
    _type = None
    for one_batch in loader:
        if _type is None:
            _type = type(one_batch)
        one_batch = list(one_batch)
        yield _type(*one_batch)

def minimal_file_reader(path: str,
                        sep: str = '\t',
                        entry_type: type = list,
                        entry_type_kwargs: Optional[Dict] = None,
                        instance_type: type = list,
                        instance_type_kwargs: Optional[Dict] = None):
    """
    Expected type signature:
    entry_type: Type(*fields, **kwargs)
    instance_type: Type(list_of_entry, **kwargs)
    """
    entry_type_kwargs = entry_type_kwargs or {}
    instance_type_kwargs = instance_type_kwargs or {}
    tokens = []
    with open(path) as f:
        for line in f:
            line = line.strip()

            if line == '':
                if tokens:
                    yield instance_type(tokens, **instance_type_kwargs)
                tokens = []
            elif line[0] == '#':
                continue
            else:
                token = line.split(sep)
                tokens.append(entry_type(*token, **entry_type_kwargs))
        if tokens:
            yield instance_type(tokens, **instance_type_kwargs)

ScidtbDatasetBatchData = namedtuple("ScidtbDatasetBatchData", ('id_array', 'pos_array', 'word_array', 'len_array',
                                                               'first_word_array', 'end_word_array', 'head_word_array',
                                                               'first_pos_array', 'end_pos_array', 'head_pos_array',
                                                               'head_deprel_array', 'edus_array', 'edus_len_array',
                                                               'sent_map_array'))

# convert dataInstance to tensor friendly in this class
class ScidtbInstance:
    def __init__(self, id: int, entry: DataInstance, dataset: Optional['ScidtbDataset'] = None):
        self.entry = entry
        self.id = id
        self.ds = dataset

        self._pos_np = None
        self._norm_np = None
        self._deprel_np = None
        self._first_word_np = None
        self._end_word_np = None
        self._head_word_np = None
        self._first_pos_np = None
        self._end_pos_np = None
        self._head_pos_np = None
        self._head_deprel_np = None

        self._edus_list = None
        self._edus_len_np = None
        self._sent_map_np = None

        # kmeans parameter
        self.cluster_label = None

    def __len__(self):
        return len(self.entry.arcs)

    # def __iter__(self):
    #     return iter(self.entry)

    def __repr__(self):
        return f'ScidtbInstance(id={self.id}, name={self.entry.name}, len={len(self)})'

    def __str__(self):
        return f'ScidtbInstance(str={" ".join(self._edus_list)})'

    def __hash__(self):
        return hash(self.entry.name)

    def __getattr__(self, item):
        return getattr(self.entry, item)

    @property
    def pos_np(self) -> Optional[List[np.ndarray]]:
        if self.ds.pos_vocab is None:
            return None
        if self._pos_np is None:
            self._pos_np = []
            for edu_postag in self.edus_postag:
                self._pos_np.append(npasarray(list(map(self.ds.pos_vocab.__getitem__, edu_postag))))
        return self._pos_np

    @property
    def first_word_np(self) -> Optional[np.ndarray]:
        if self.ds.word_vocab is None:
            return None
        if self._first_word_np is None:
            first_words = []
            for edu in self.edus:
                first_words.append(edu[0])
            self._first_word_np = npasarray(list(map(self.ds.word_vocab.__getitem__, first_words)))
            del first_words
        return self._first_word_np

    @property
    def end_word_np(self) -> Optional[np.ndarray]:
        if self.ds.word_vocab is None:
            return None
        if self._end_word_np is None:
            end_words = []
            for edu in self.edus:
                end_words.append(edu[-1])
            self._end_word_np = npasarray(list(map(self.ds.word_vocab.__getitem__, end_words)))
            del end_words
        return self._end_word_np

    @property
    def head_word_np(self) -> Optional[np.ndarray]:
        if self.ds.word_vocab is None:
            return None
        if self._head_word_np is None:
            head_words = []
            for edu in self.edus_head:
                head_words.append(edu[0])
            self._head_word_np = npasarray(list(map(self.ds.word_vocab.__getitem__, head_words)))
            del head_words
        return self._head_word_np


    @property
    def first_pos_np(self) -> Optional[np.ndarray]:
        if self.ds.pos_vocab is None:
            return None
        if self._first_pos_np is None:
            first_pos = []
            for edu in self.edus_postag:
                first_pos.append(edu[0])
            self._first_pos_np = npasarray(list(map(self.ds.pos_vocab.__getitem__, first_pos)))
            del first_pos
        return self._first_pos_np

    @property
    def end_pos_np(self) -> Optional[np.ndarray]:
        if self.ds.pos_vocab is None:
            return None
        if self._end_pos_np is None:
            end_pos = []
            for edu in self.edus_postag:
                end_pos.append(edu[-1])
            self._end_pos_np = npasarray(list(map(self.ds.pos_vocab.__getitem__, end_pos)))
            del end_pos
        return self._end_pos_np

    @property
    def head_pos_np(self) -> Optional[np.ndarray]:
        if self.ds.pos_vocab is None:
            return None
        if self._head_pos_np is None:
            head_pos = []
            for edu in self.edus_head:
                head_pos.append(edu[1])
            self._head_pos_np = npasarray(list(map(self.ds.pos_vocab.__getitem__, head_pos)))
            del head_pos
        return self._head_pos_np

    @property
    def norm_np(self) -> Optional[List[np.ndarray]]:
        if self.ds.word_vocab is None:
            return None
        if self._norm_np is None:
            self._norm_np = []
            for edu in self.edus:
                self._norm_np.append(npasarray(list(map(self.ds.word_vocab.__getitem__, edu))))
        return self._norm_np

    @property
    def head_deprel_np(self) -> Optional[np.ndarray]:
        if self.ds.deprel_vocab is None:
            return None
        if self._head_deprel_np is None:
            head_deprel = []
            for edu in self.edus_head:
                head_deprel.append(edu[-1])
            self._head_deprel_np = npasarray(list(map(self.ds.deprel_vocab.__getitem__, head_deprel)))
            del head_deprel
        return self._head_deprel_np

    @property
    def edus_list(self) -> Optional[List]:
        if self.ds.word_vocab is None:
            return None
        if self._edus_list is None:
            edus_list = []
            max_len = 0
            for edu in self.edus:
                max_len = len(edu) if len(edu) > max_len else max_len
                word_list = []
                for word in edu:
                    word_list.append(self.ds.word_vocab[word])
                edus_list.append(word_list)
            self._edus_list = edus_list
            del edus_list
        return self._edus_list

    @property
    def edus_len_np(self) -> Optional[np.ndarray]:
        if self._edus_len_np is None:
            self._edus_len_np = npasarray(list(map(len, self.edus)))
        return self._edus_len_np

    # differren version of sbnds
    # @property
    # def sent_map_np(self) -> Optional[np.ndarray]:
    #     if self._sent_map_np is None:
    #         self._sent_map_np = make_same_sent_map(self.edus + ['root'], self.sbnds)
    #         self._sent_map_np = self._sent_map_np[1:, 1:]
    #     return self._sent_map_np

    @property
    def sent_map_np(self) -> Optional[np.ndarray]:
        if self._sent_map_np is None:
            self._sent_map_np = np.array(self.sbnds)
        return self._sent_map_np

class ScidtbDataset(Dataset):
    def __init__(self, instance_list: List[DataInstance], vocab_word: OrderedDict= None, vocab_postag: OrderedDict= None,
                 vocab_deprel: OrderedDict= None, vocab_relation: OrderedDict= None):
        # cache: only cache instances
        self.instances = []
        for idx, instance in enumerate(instance_list):
            self.instances.append(ScidtbInstance(idx, instance, self))
        self.word_vocab = vocab_word
        self.pos_vocab = vocab_postag
        self.deprel_vocab = vocab_deprel
        self.relation_vocab = vocab_relation

    def get_dataloader(self, same_len: bool, batch_size: int, drop_last: bool, shuffle: int,
                       num_workers: int, min_len: int = 1, max_len: int = 10000):
        sampler = LengthBucketSampler if same_len else BasicSampler

        return DataLoader(dataset=self,
                          num_workers=num_workers,
                          pin_memory=True,
                          batch_sampler=sampler(self, batch_size, drop_last, shuffle, min_len, max_len),
                          collate_fn=self.collect_fn)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def __iter__(self):
        return iter(self.instances)

    def get_all_len(self):
        return torch.tensor([len(i) for i in self.instances])

    @staticmethod
    def collect_fn(batch_raw_data: List[ScidtbInstance]):

        id_array = torch.tensor([d.id for d in batch_raw_data])
        len_array = torch.tensor([len(d) for d in batch_raw_data])

        first_word_np = [ins.first_word_np for ins in batch_raw_data]
        if first_word_np[0] is None:
            first_word_array = None
        else:
            pad = batch_raw_data[0].ds.word_vocab["<unk>"]
            first_word_array = pad_sequence(list(map(torch.tensor, first_word_np)), True, pad)

        end_word_np = [ins.end_word_np for ins in batch_raw_data]
        if end_word_np[0] is None:
            end_word_array = None
        else:
            pad = batch_raw_data[0].ds.word_vocab["<unk>"]
            end_word_array = pad_sequence(list(map(torch.tensor, end_word_np)), True, pad)

        head_word_np = [ins.head_pos_np for ins in batch_raw_data]
        if head_word_np[0] is None:
            head_word_array = None
        else:
            pad = batch_raw_data[0].ds.word_vocab["<unk>"]
            head_word_array = pad_sequence(list(map(torch.tensor, head_word_np)), True, pad)

        first_pos_np = [ins.first_pos_np for ins in batch_raw_data]
        if first_pos_np[0] is None:
            first_pos_array = None
        else:
            pad = batch_raw_data[0].ds.pos_vocab["<unk>"]
            first_pos_array = pad_sequence(list(map(torch.tensor, first_pos_np)), True, pad)

        end_pos_np = [ins.end_pos_np for ins in batch_raw_data]
        if end_word_np[0] is None:
            end_pos_array = None
        else:
            pad = batch_raw_data[0].ds.pos_vocab["<unk>"]
            end_pos_array = pad_sequence(list(map(torch.tensor, end_pos_np)), True, pad)

        head_pos_np = [ins.head_pos_np for ins in batch_raw_data]
        if head_pos_np[0] is None:
            head_pos_array = None
        else:
            pad = batch_raw_data[0].ds.pos_vocab["<unk>"]
            head_pos_array = pad_sequence(list(map(torch.tensor, head_pos_np)), True, pad)

        head_deprel_np = [ins.head_deprel_np for ins in batch_raw_data]
        if head_deprel_np[0] is None:
            head_deprel_array = None
        else:
            pad = batch_raw_data[0].ds.deprel_vocab["<unk>"]
            head_deprel_array = pad_sequence(list(map(torch.tensor, head_deprel_np)), True, pad)

        edus_list = [ins.edus_list for ins in batch_raw_data]
        if edus_list[0] is None:
            edus_array = None
        else:
            pad = batch_raw_data[0].ds.word_vocab["<unk>"]
            max_len = max(list(map(max, [list(map(len, edus_ins)) for edus_ins in edus_list])))
            # padding to same size [batch, max_edu, max_len]
            for edus_ins in edus_list:
                for idx, edu in enumerate(edus_ins):
                    edus_ins[idx] = edu + [pad] * (max_len - len(edu))

            edus_array = pad_sequence(list(map(torch.tensor, edus_list)), batch_first=True, padding_value=pad).long()

        edus_len_np = [ins.edus_len_np for ins in batch_raw_data]
        if edus_len_np[0] is None:
            edus_len_array = None
        else:
            pad = 0
            edus_len_array = pad_sequence(list(map(torch.tensor, edus_len_np)), batch_first=True, padding_value=pad).long()

        sent_map_np = [ins.sent_map_np for ins in batch_raw_data]

        # if sent_map_np[0] is None:
        #     sent_map_array = None
        # else:
        #     pad = 0
        #     sent_len = max(list(map(max, [list(map(len, sent_map_ins)) for sent_map_ins in sent_map_np])))
        #     for idx, sent_map_ins in enumerate(sent_map_np):
        #         x, y = sent_map_ins.shape
        #         zeros = np.zeros((x, sent_len-y))
        #         sent_map_np[idx] = np.concatenate((sent_map_ins, zeros), axis=1)
        #     sent_map_array = pad_sequence(list(map(torch.tensor, sent_map_np)), batch_first=True, padding_value=pad).long()

        if sent_map_np[0] is None:
            sent_map_array = None
        else:
            pad = -100
            sent_map_array = pad_sequence(list(map(torch.tensor, sent_map_np)), batch_first=True, padding_value=pad).long()
        return ScidtbDatasetBatchData(id_array=id_array, pos_array=None, word_array=None, len_array=len_array,
                                      first_word_array=first_word_array, end_word_array=end_word_array,
                                      head_word_array=head_word_array, first_pos_array=first_pos_array,
                                      end_pos_array=end_pos_array, head_pos_array=head_pos_array,
                                      head_deprel_array=head_deprel_array, edus_len_array=edus_len_array,
                                      edus_array=edus_array, sent_map_array=sent_map_array)

ScidtbDatasetWithEmbBatchData = namedtuple("ScidtbDatasetWithEmbBatchData", ('id_array', 'pos_array', 'word_array', 'len_array',
                                                               'first_word_array', 'end_word_array', 'head_word_array',
                                                               'first_pos_array', 'end_pos_array', 'head_pos_array',
                                                               'head_deprel_array', 'edus_array', 'edus_len_array',
                                                               'sent_map_array', 'context_embed_array', 'cluster_label_array'))


class ScidtbInstanceWithEmb(ScidtbInstance):
    def __init__(self, id: int, entry: DataInstance, dataset: 'ScidtbDatasetWithEmb' = None):
        super().__init__(id, entry, dataset)
        self.context_embed_np = None

    def __getattr__(self, item):
        return getattr(self.entry, item)

    # here property implement is bad.
    # https://python3-cookbook.readthedocs.io/zh_CN/latest/c08/p08_extending_property_in_subclass.html
    # @property
    def build_endpoint_embed_np(self) -> Optional[np.ndarray]:
        if self.ds.cs_encoder is None:
            return None
        edus_representation = self.ds.cs_encoder.sent_sensitive_encode(self.entry)
        final = []
        for edu_representation in edus_representation:
            edu_representation = torch.cat([edu_representation[0], edu_representation[-1]], dim=-1)
            final.append(edu_representation)
        final = torch.stack(final, dim=0).detach().cpu().numpy()
        self.context_embed_np = final
        return self.context_embed_np


class ScidtbDatasetWithEmb(ScidtbDataset):
    def __init__(self, instance_list: List[DataInstance], vocab_word: OrderedDict= None, vocab_postag: OrderedDict=None,
                 vocab_deprel: OrderedDict=None, vocab_relation: OrderedDict=None, encoder: str='bert', pretrained=None):
        super().__init__(instance_list, vocab_word, vocab_postag, vocab_deprel, vocab_relation)
        self.instances = []
        for idx, instance in enumerate(instance_list):
            self.instances.append(ScidtbInstanceWithEmb(idx, instance, self))
        if encoder == 'bert':
            self.cs_encoder = CSEncoder(encoder, gpu=True)
        elif encoder == 'None':
            self.cs_encoder = None
        else:
            raise NotImplementedError

        # load pretrained
        if pretrained is not None:
            embeddings = utils.load_embedding(pretrained)
            for ins in self.instances:
                # Alert here remove embedding of <root>
                ins.context_embed_np = embeddings[ins.name][1:]

    def norm_embed(self, std):
        embed_np = []
        for instance in self.instances:
            embed_np.append(instance.context_embed_np)
        embed_np = np.concatenate(embed_np, axis=0)
        if std is None:
            std = np.std(embed_np)
        for instance in self.instances:
            instance.context_embed_np = instance.context_embed_np / std
        return std

    def kmeans(self, kcluster: int, random_seed: int) -> KMeans:
        embed_np = []
        for instance in self.instances:
            embed_np.append(instance.build_context_embed_np())
        embed_np = np.concatenate(embed_np, axis=0)
        kmeans = KMeans(n_clusters=kcluster, random_state=random_seed).fit(embed_np)
        return kmeans

    def kmeans_label(self, kmeans: KMeans):
        for instance in self.instances:
            labels = kmeans.predict(instance.build_context_embed_np())
            instance.cluster_label = labels

    def clean_encoder(self):
        if self.cs_encoder is not None:
            self.cs_encoder.clean()

    @staticmethod
    def collect_fn(batch_raw_data: List[ScidtbInstanceWithEmb]):

        id_array = torch.tensor([d.id for d in batch_raw_data])
        len_array = torch.tensor([len(d) for d in batch_raw_data])

        first_word_np = [ins.first_word_np for ins in batch_raw_data]
        if first_word_np[0] is None:
            first_word_array = None
        else:
            pad = batch_raw_data[0].ds.word_vocab["<unk>"]
            first_word_array = pad_sequence(list(map(torch.tensor, first_word_np)), True, pad)

        end_word_np = [ins.end_word_np for ins in batch_raw_data]
        if end_word_np[0] is None:
            end_word_array = None
        else:
            pad = batch_raw_data[0].ds.word_vocab["<unk>"]
            end_word_array = pad_sequence(list(map(torch.tensor, end_word_np)), True, pad)

        head_word_np = [ins.head_pos_np for ins in batch_raw_data]
        if head_word_np[0] is None:
            head_word_array = None
        else:
            pad = batch_raw_data[0].ds.word_vocab["<unk>"]
            head_word_array = pad_sequence(list(map(torch.tensor, head_word_np)), True, pad)

        first_pos_np = [ins.first_pos_np for ins in batch_raw_data]
        if first_pos_np[0] is None:
            first_pos_array = None
        else:
            pad = batch_raw_data[0].ds.pos_vocab["<unk>"]
            first_pos_array = pad_sequence(list(map(torch.tensor, first_pos_np)), True, pad)

        end_pos_np = [ins.end_pos_np for ins in batch_raw_data]
        if end_word_np[0] is None:
            end_pos_array = None
        else:
            pad = batch_raw_data[0].ds.pos_vocab["<unk>"]
            end_pos_array = pad_sequence(list(map(torch.tensor, end_pos_np)), True, pad)

        head_pos_np = [ins.head_pos_np for ins in batch_raw_data]
        if head_pos_np[0] is None:
            head_pos_array = None
        else:
            pad = batch_raw_data[0].ds.pos_vocab["<unk>"]
            head_pos_array = pad_sequence(list(map(torch.tensor, head_pos_np)), True, pad)

        head_deprel_np = [ins.head_deprel_np for ins in batch_raw_data]
        if head_deprel_np[0] is None:
            head_deprel_array = None
        else:
            pad = batch_raw_data[0].ds.deprel_vocab["<unk>"]
            head_deprel_array = pad_sequence(list(map(torch.tensor, head_deprel_np)), True, pad)

        edus_list = [ins.edus_list for ins in batch_raw_data]
        if edus_list[0] is None:
            edus_array = None
        else:
            pad = batch_raw_data[0].ds.word_vocab["<unk>"]
            max_len = max(list(map(max, [list(map(len, edus_ins)) for edus_ins in edus_list])))
            # padding to same size [batch, max_edu, max_len]
            for edus_ins in edus_list:
                for idx, edu in enumerate(edus_ins):
                    edus_ins[idx] = edu + [pad] * (max_len - len(edu))

            edus_array = pad_sequence(list(map(torch.tensor, edus_list)), batch_first=True, padding_value=pad).long()

        edus_len_np = [ins.edus_len_np for ins in batch_raw_data]
        if edus_len_np[0] is None:
            edus_len_array = None
        else:
            pad = 0
            edus_len_array = pad_sequence(list(map(torch.tensor, edus_len_np)), batch_first=True, padding_value=pad).long()

        sent_map_np = [ins.sent_map_np for ins in batch_raw_data]
        # if sent_map_np[0] is None:
        #     sent_map_array = None
        # else:
        #     pad = 0
        #     sent_len = max(list(map(max, [list(map(len, sent_map_ins)) for sent_map_ins in sent_map_np])))
        #     for idx, sent_map_ins in enumerate(sent_map_np):
        #         x, y = sent_map_ins.shape
        #         zeros = np.zeros((x, sent_len-y))
        #         sent_map_np[idx] = np.concatenate((sent_map_ins, zeros), axis=1)
        #     sent_map_array = pad_sequence(list(map(torch.tensor, sent_map_np)), batch_first=True, padding_value=pad).long()
        if sent_map_np[0] is None:
            sent_map_array = None
        else:
            pad = -100
            # maxlen = max([sent_map_ins.shape[0] for sent_map_ins in sent_map_np])
            sent_map_array = pad_sequence(list(map(torch.tensor, sent_map_np)), batch_first=True, padding_value=pad).long()

        context_embed_np = [ins.context_embed_np for ins in batch_raw_data]
        if context_embed_np is None:
            context_embed_array = None
        else:
            pad = 0.0
            context_embed_array = pad_sequence(list(map(torch.tensor, context_embed_np)), batch_first=True, padding_value=pad)

        cluster_label_np = [ins.cluster_label for ins in batch_raw_data]
        if cluster_label_np[0] is None:
            cluster_label_array = None
        else:
            pad = 0
            cluster_label_array = pad_sequence(list(map(torch.tensor, cluster_label_np)), True, pad).long()
        return ScidtbDatasetWithEmbBatchData(id_array=id_array, pos_array=None, word_array=None, len_array=len_array,
                                             first_word_array=first_word_array, end_word_array=end_word_array,
                                             head_word_array=head_word_array, first_pos_array=first_pos_array,
                                             end_pos_array=end_pos_array, head_pos_array=head_pos_array,
                                             head_deprel_array=head_deprel_array, edus_len_array=edus_len_array,
                                             edus_array=edus_array, sent_map_array=sent_map_array,
                                             context_embed_array=context_embed_array,
                                             cluster_label_array=cluster_label_array)


