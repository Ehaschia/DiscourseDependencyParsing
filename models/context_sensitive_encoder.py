# this file loads context sensitive encoder like bert
# and encodes **discourse** context

import torch

from transformers import BertModel, BertTokenizer
from typing import List, Tuple

# context sensitive data loader
from utils import DataInstance
import torch.nn as nn
import gc

class CSEncoder(nn.Module):

    def __init__(self, encoder: str, gpu: bool = True, finetune=False):
        super(CSEncoder, self).__init__()
        self.encoder_name = encoder
        self.gpu = gpu
        if self.encoder_name == "bert":
            self.model = BertModel.from_pretrained('./encoder/bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained("./encoder/bert-base-uncased")
            self.finetune = finetune
            if finetune:
                self.model.train()
            else:
                self.model.eval()

        else:
            raise NotImplementedError()

        if self.gpu:
            self.model.to('cuda')

    @torch.no_grad()
    def encode(self, discourse: DataInstance) -> Tuple[torch.Tensor, torch.Tensor]:
        # inputs = [self.tokenizer(str_edu) for str_edu in str_edus]
        edus = list(map(' '.join, discourse.edus))
        with torch.no_grad():
            inputs = self.tokenizer(edus, return_tensors='pt', padding=True)

            inputs.data['input_ids'] = inputs.data['input_ids'].to(self.model.device)
            inputs.data['token_type_ids'] = inputs.data['token_type_ids'].to(self.model.device)
            inputs.data['attention_mask'] = inputs.data['attention_mask'].to(self.model.device)
            res = self.model(**inputs)
        # here we use the predict
        res[0].require_grad = False
        res[1].require_grad = False
        return res[0][:, 0].detach(), res[1].detach()

    # encode whole sentence
    @torch.no_grad()
    def encode_sentence(self, discourse: DataInstance):
        edus = list(map(' '.join, discourse.edus))
        to_merge = edus[1:]
        sents = [edus[0]]

        for idx, (begin, end) in enumerate(discourse.sbnds):
            sents.append(' '.join(to_merge[begin:end + 1]))
        inputs = self.tokenizer(sents, return_tensors='pt', padding=True)
        inputs.data['input_ids'] = inputs.data['input_ids'].to(self.model.device)
        inputs.data['token_type_ids'] = inputs.data['token_type_ids'].to(self.model.device)
        inputs.data['attention_mask'] = inputs.data['attention_mask'].to(self.model.device)
        res = self.model(**inputs)

        res[0].require_grad = False
        return res[0].detach()

    # encode sentence and get edu representation
    def sent_sensitive_encode(self, discourse: DataInstance) -> List[torch.Tensor]:
        sent_representation = self.grad_encode_sentence(discourse) if self.finetune else self.encode_sentence(discourse)

        tokens = self.tokenizer(list(map(' '.join, discourse.edus)), padding=False)['input_ids']
        edu_representation = [sent_representation[0][1:len(tokens[0])-1]]
        for sent, span in zip(sent_representation[1:], discourse.sbnds):
            current_edus = tokens[span[0]+1: span[1]+2]
            idx = 0
            sent = sent[1:]
            for current_edu in current_edus:
                current_edu = current_edu[1:-1]
                edu_representation.append(sent[idx: idx+len(current_edu)])
                idx += len(current_edu)

        assert len(edu_representation) == len(discourse.edus)
        return edu_representation

    def sent_minus_encode(self, discourse: DataInstance):
        edus_representation = self.sent_sensitive_encode(discourse)
        final = []
        for edu_representation in edus_representation:
            final.append(edu_representation[-1] - edu_representation[0])
        return torch.stack(final, dim=0)

    def sent_avg_pooling_encode(self, discourse: DataInstance):
        edus_representation = self.sent_sensitive_encode(discourse)
        final = []
        for edu_representation in edus_representation:
            # current_size = len(edu_representation)
            edu_size = edu_representation.shape[0]
            # edu_representation = torch.stack(edu_representation, dim=0)
            edu_representation = torch.avg_pool1d(edu_representation.unsqueeze(0).permute(0, 2, 1),
                                                  kernel_size=edu_size, stride=edu_size).squeeze()
            final.append(edu_representation)
        return torch.stack(final, dim=0)

    def sent_max_pooling_encode(self, discourse: DataInstance):
        edus_representation = self.sent_sensitive_encode(discourse)
        final = []
        for edu_representation in edus_representation:
            # current_size = len(edu_representation)
            edu_size = edu_representation.shape[0]
            # edu_representation = torch.stack(edu_representation, dim=0)
            edu_representation = torch.nn.functional.max_pool1d(edu_representation.unsqueeze(0).permute(0, 2, 1),
                                                  kernel_size=edu_size, stride=edu_size).squeeze()
            final.append(edu_representation)
        return torch.stack(final, dim=0)

    def sent_mean_encode(self, discourse: DataInstance):
        edus_representation = self.sent_sensitive_encode(discourse)
        final = []
        for edu_representation in edus_representation:
            # current_size = len(edu_representation)
            edu_size = edu_representation.shape[0]
            # edu_representation = torch.stack(edu_representation, dim=0)
            edu_representation = edu_representation.mean(dim=0)
            final.append(edu_representation)
        final = torch.stack(final, dim=0)
        return final # / torch.std(final)

    def grad_encode(self, discourse: DataInstance):
        # inputs = [self.tokenizer(str_edu) for str_edu in str_edus]
        edus = list(map(' '.join, discourse.edus))

        inputs = self.tokenizer(edus, return_tensors='pt', padding=True)

        inputs.data['input_ids'] = inputs.data['input_ids'].to(self.model.device)
        inputs.data['token_type_ids'] = inputs.data['token_type_ids'].to(self.model.device)
        inputs.data['attention_mask'] = inputs.data['attention_mask'].to(self.model.device)

        res = self.model(**inputs)
        # here we use the predict
        # assert res[0].requires_grad == True
        # assert res[1].require_grad == True
        return res[0][:, 0], res[1]

    def grad_encode_sentence(self, discourse: DataInstance):
        edus = list(map(' '.join, discourse.edus))
        to_merge = edus[1:]
        sents = [edus[0]]

        for idx, (begin, end) in enumerate(discourse.sbnds):
            sents.append(' '.join(to_merge[begin:end + 1]))
        inputs = self.tokenizer(sents, return_tensors='pt', padding=True)
        inputs.data['input_ids'] = inputs.data['input_ids'].to(self.model.device)
        inputs.data['token_type_ids'] = inputs.data['token_type_ids'].to(self.model.device)
        inputs.data['attention_mask'] = inputs.data['attention_mask'].to(self.model.device)
        res = self.model(**inputs)

        return res[0]

    def clean(self):
        del self.model
        gc.collect()
