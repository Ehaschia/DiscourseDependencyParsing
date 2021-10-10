import operator
from functools import reduce
from typing import List

import numpy as np

from model import utility
from model.dep_dataset import PascalSentence
from model.ncrfae import NCRFAE
import torch

@torch.no_grad()
def evaluate_uas(model: NCRFAE, dataset: List[PascalSentence]) -> float:
    batches = utility.construct_batches_by_length(dataset, batch_size=200)

    num_tokens = 0
    num_correct = 0
    model.eval()
    for batch in batches:
        golden = np.array(list(map(lambda x: getattr(x, 'heads'), batch)))
        trees = model.decoding(batch)
        # trees = model.decoding(batch, enable_prior=False).detach().cpu().numpy()
        trees = np.array(trees)
        num_correct += np.sum(golden[:, 1:] == trees[:, 1:])
        num_tokens += (golden[:, 1:]).size

    if num_tokens == 0:
        uas = 0
    else:
        uas = num_correct / num_tokens
    return uas


@torch.no_grad()
def evaluate(model, dataset):
    batches = utility.construct_batches_by_length(dataset, batch_size=200)
    data_size = reduce(operator.add, map(operator.length_hint, batches), 0)
    epoch_loss = 0.
    model.eval()
    for i, batch in enumerate(batches):
        loss = model(batch)
        epoch_loss += utility.to_scalar(loss) * len(batch)
    return epoch_loss / data_size if data_size > 0 else 0.0
