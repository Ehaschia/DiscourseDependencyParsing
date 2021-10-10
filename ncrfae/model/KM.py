import torch

from model import utility
from model.dep_dataset import SentencePreprocessor
from model.ncrfae import NCRFAE


def Km_expected_count(sent_len):
    count = torch.zeros((sent_len, sent_len))

    for i in range(sent_len):
        if i != 0:
            for j in range(1, sent_len):
                if j != i:
                    count[i, j] += 1. / abs(j - i)

    count[0, 1:] = 1. / (sent_len - 1)  # ROOT
    if sent_len > 2:
        count[1:, 1:] = count[1:, 1:] / count[1:, 1:].sum(dim=0) * (sent_len - 2) / (sent_len - 1)
    return count


class KmEM():
    def __init__(self, model: NCRFAE, preprocessor: SentencePreprocessor,
                 data,
                 laplace_smoothing_k=1,
                 use_gpu=False):
        self.model = model
        self.preprocessor = preprocessor
        self.data = data
        self.k = laplace_smoothing_k  # laplace-smoothing-k
        self.use_gpu = use_gpu

    def step(self):
        batches = utility.construct_batches_by_length(self.data, batch_size=200)
        count = torch.zeros_like(self.model.multinomial.data).fill_(self.k)
        for batch in batches:
            batch_tensor = self.preprocessor.process_batch(batch).data

            batch_size, sent_length = batch_tensor.size()

            m, n = count.size()

            km_expected_count = (Km_expected_count(sent_length)[None, :, :]).repeat(batch_size, 1, 1)
            if self.use_gpu:
                km_expected_count = km_expected_count.cuda()
            edge_index = (batch_tensor[:, :, None] * m + batch_tensor[:, None, :]).view(-1)

            count.put_(edge_index, km_expected_count, accumulate=True)

        count /= (count.sum(dim=1).contiguous().view(-1, 1))
        self.model.multinomial.data = count


class KmInit(torch.autograd.Function):
    pass


if __name__ == '__main__':
    print(Km_expected_count(4))
    print(Km_expected_count(3))
    print(Km_expected_count(2))
