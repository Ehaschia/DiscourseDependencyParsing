import torch

from model import utility
from model.dep_dataset import SentencePreprocessor
from model.ncrfae import NCRFAE


class HardEM():
    def __init__(self, model: NCRFAE,
                 preprocessor: SentencePreprocessor,
                 data, laplace_smoothing_k=1, use_gpu=False):
        self.model = model
        self.preprocessor = preprocessor
        self.data = data
        self.k = laplace_smoothing_k  # laplace-smoothing-k
        self.use_gpu = use_gpu

    def step(self):
        batches = utility.construct_batches_by_length(self.data, batch_size=200)
        count = torch.zeros_like(self.model.multinomial.data).fill_(self.k)
        for batch in batches:
            trees = self.model.decoding(batch, training=False, enable_prior=True)

            batch_size, _ = trees.size()

            batch_token_id = self.model.preprocessor.process_batch(batch)
            sent_idx = torch.arange(batch_size).contiguous().view(-1, 1).long()

            # Supervised HACK
            trees = torch.LongTensor([s.heads for s in batch])
            # HACK END

            if self.use_gpu:
                sent_idx = sent_idx.cuda()
                trees = trees.cuda()

            heads_token_id = batch_token_id.data[sent_idx, trees[:, 1:]].contiguous().view(-1)
            children_token_id = batch_token_id.data[:, 1:].contiguous().view(-1)

            m, n = count.size()
            linear_idx = heads_token_id * m + children_token_id

            one_tensor = torch.FloatTensor([1]).expand_as(linear_idx)
            if self.use_gpu:
                one_tensor = one_tensor.cuda()

            count.put_(linear_idx, one_tensor, accumulate=True)

        count /= (count.sum(dim=1).contiguous().view(-1, 1))

        self.model.multinomial.data = count
