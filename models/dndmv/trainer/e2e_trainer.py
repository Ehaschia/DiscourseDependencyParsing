from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, List, Union

import torch
from torch import Tensor

from easydict import EasyDict
import utils
from models.dndmv.dmv import DMV
from models.dndmv.dndmv import DiscriminativeNeuralDMV
from utils.data import ConllDataset, prefetcher, ScidtbDataset, ScidtbDatasetWithEmb
from utils.utils import calculate_uas, make_sure_dir_exists, namedtuple2dict
from torch.utils.data import Dataset

class OnlineEMTrainer(object):
    """Online EM algorithm

    Sample one batch rather than all instance to run E step.
    This trainer WILL NOT handle oom of cuda caused by big e_batch_size_train.
    This trainer WILL automatically transfer data to cuda if device = GPU.
    """

    # noinspection PyUnusedLocal
    @staticmethod
    def config_template():
        end2end = True
        e_batch_size_train = 1000  # batch size to run e step
        m_batch_size_train = 100  # batch size (for instance) of rule samples to run m step

        epoch = 100  # epoch for online train
        epoch_init = 3  # epoch for init nn using dmv samples
        epoch_nn = 10  # epoch for nn in non-end2end train
        neural_stop_criteria = 1e-3

        same_len = False  # whethar instances in one batch have the same len
        shuffle = 0  # shuffle level, see data/LengthBucketSampler or data/BasicSampler
        drop_last = False  # drop batch which len(batch) < e_batch_size
        max_len_train = 10  # max len of instance used on train dataset
        max_len_eval = 10  # max len of instance used on dev/test dataset

        num_workers = 4  # num_workers of DataLoader
        device = 'cuda'

    def __init__(self, cfg: Dict, dmv: DMV, nn: DiscriminativeNeuralDMV, converter: Callable,
                 train_ds: Union[ConllDataset, ScidtbDataset],
                 test_ds: Union[ConllDataset, ScidtbDataset],
                 dev_ds: Union[ConllDataset, ScidtbDataset, None] = None):
        """
        :param cfg: config dict
        :param dmv: dmv model
        :param nn: nn model, see module/neural_m/_interface.py
        :param converter: (word_array, pos_array) -> tag_array
        :param train_ds: train dataset
        :param test_ds: test dataset
        :param dev_ds: dev dataset. use test_ds if not given
        """
        print('not update dmv')
        self.cfg = EasyDict(cfg)
        self.workspace = Path(self.cfg.workspace)

        self.dmv = dmv
        self.nn = nn
        self.converter = converter

        self.train_ds = train_ds
        self.dev_ds = dev_ds or test_ds
        self.test_ds = test_ds

        self.train_iter = self.train_ds.get_dataloader(
            same_len=self.cfg.same_len, num_workers=self.cfg.num_worker,
            min_len=self.cfg.min_len_train, max_len=self.cfg.max_len_train,
            batch_size=self.cfg.e_batch_size, shuffle=self.cfg.shuffle,
            drop_last=self.cfg.drop_last)

        make_sure_dir_exists(self.workspace)
        make_sure_dir_exists(self.workspace / 'best_uas')
        make_sure_dir_exists(self.workspace / 'best_ll')
        make_sure_dir_exists(self.workspace / 'all')

    def train(self, epoch: Optional[int] = None, report: bool = True, stop_hook: Optional[Callable] = None, subepoch: Optional[int] = None) -> None:
        """ train the model
        :param epoch: max train epoch
        :param report: do eval after each epoch
        :param stop_hook:
            (epoch_id, ll, best_ll, best_ll_epoch, uas, best_uas, best_uas_epoch) -> bool
        """
        epoch = epoch or self.cfg.epoch
        subepoch = subepoch or self.cfg.epoch_nn
        best_ll, best_uas = -1e30, -1.
        best_ll_epoch, best_uas_epoch = -1, -1

        for epoch_id in range(epoch):
            it = self.train_iter if self.cfg.device == 'cpu' else prefetcher(self.train_iter)
            model_ll, nn_loss, n_instance, nn_total_epoch, n_batch = 0., 0., 0, 0, 0

            for one_batch in it:
                tag_array = self.converter(None, one_batch.cluster_label_array)
                arrays = namedtuple2dict(one_batch)

                if self.cfg.end2end:
                    self.nn.train()
                    tdr_param = self.nn.predict_pipeline(arrays, tag_array, mode='tdr')
                else:
                    self.nn.eval()
                    with torch.no_grad():
                        tdr_param = self.nn.predict_pipeline(arrays, tag_array, mode='tdr')
                    tdr_param = [p.requires_grad_() if p is not None else None for p in tdr_param]

                # use dmv to get missing params
                needed_param = [(i, m) for i, (p, m) in enumerate(zip(tdr_param, 'tdr')) if p is None]
                dmv_mode = ''.join([p[1] for p in needed_param])
                if dmv_mode != '':
                    dmv_param = self.dmv.build_scores(tag_array, mode=dmv_mode, using_fake=False)
                    for idx, (i, _) in enumerate(needed_param):
                        tdr_param[i] = dmv_param[idx]

                t = self.dmv.build_final_trans_scores(tdr_param[0], tdr_param[2], False)
                d = self.dmv.build_final_dec_scores(tdr_param[1], False)
                ll = self.dmv(t, d, one_batch.len_array)
                ll = torch.sum(ll)
                self.dmv.zero_grad()

                if self.cfg.end2end:
                    self.nn.zero_grad()
                    loss = -ll / torch.sum(one_batch.len_array)
                    loss.backward()
                    self.nn.optimize()
                    loss = loss.item()
                else:
                    ll.backward()
                    counts = {'t': tdr_param[0].grad.detach() if tdr_param[0].grad is not None else None,
                              'd': tdr_param[1].grad.detach() if tdr_param[1].grad is not None else None,
                              'r': tdr_param[2].grad.detach() if tdr_param[2].grad is not None else None}
                    loss, nn_epoch = self.non_end2end_neural_train(tag_array, arrays, counts,
                        mode=''.join(set('tdr') - set(dmv_mode)), epoch=subepoch)
                    # self.dmv.update_param_with_count(mode=dmv_mode)
                    # self.dmv.normalize_param()

                    nn_total_epoch += nn_epoch
                nn_loss += loss
                model_ll += ll.item()
                n_instance += len(tag_array)
                n_batch += 1

            # if epoch_id == 9:
            #     self.nn.vib.vib_beta = 1e-8
            #     print(self.nn.vib.vib_beta)

            nn_loss = nn_loss / n_instance
            model_ll = model_ll / n_instance
            # utils.ex.logger.info(
            #     f'epoch {epoch_id}, train.loss={nn_loss}, train.likelihood={model_ll}, '
            #     f'{"using end2end" if self.cfg.end2end else f"run {nn_total_epoch / n_batch:.2f} epochs"}')
            # utils.ex.log_scalar('train.loss', nn_loss, epoch_id)
            # utils.ex.log_scalar('train.likelihood', model_ll, epoch_id)
            utils.writelog(f'epoch {epoch_id}, train.loss={nn_loss}, train.likelihood={model_ll}, run {nn_total_epoch / n_batch:.2f} epochs')

            torch.save(self.dmv.state_dict(), self.workspace / 'all' / f'dmv_{epoch_id}')
            torch.save(self.nn.state_dict(), self.workspace / 'all' / f'nn_{epoch_id}')
            torch.save(torch.cuda.memory_snapshot(), self.workspace / 'all' / f'ss_{epoch_id}')

            if model_ll > best_ll:
                # utils.ex.logger.info('get new best ll')
                torch.save(self.dmv.state_dict(), self.workspace / 'best_ll' / 'dmv')
                torch.save(self.nn.state_dict(), self.workspace / 'best_ll' / 'nn')
                best_ll = model_ll
                best_ll_epoch = epoch_id

            if report:
                uas_train, ll_train = self.evaluate(self.train_ds)
                uas_dev, ll_dev = 0.0, 0.0
                uas_test, ll_test = self.evaluate(self.test_ds)
                # utils.ex.logger.info(f'epoch {epoch_id}, dev.uas={uas_dev}, test.uas={uas_test}')
                # utils.ex.log_scalar('dev.uas', uas_dev, epoch_id)
                # utils.ex.log_scalar('test.uas', uas_test, epoch_id)

                utils.writelog(f'epoch {epoch_id}, train.uas= {uas_train}, dev.uas={uas_dev}, test.uas={uas_test}')
                # nni.report_intermediate_result(uas_test)

                if uas_test > best_uas:
                    # utils.ex.logger.info('get new best test.uas')
                    torch.save(self.dmv.state_dict(), self.workspace / 'best_uas' / 'dmv')
                    torch.save(self.nn.state_dict(), self.workspace / 'best_uas' / 'nn')
                    best_uas = uas_test
                    best_uas_epoch = epoch_id

            if stop_hook and stop_hook(epoch_id, model_ll, best_ll, best_ll_epoch, uas_test, best_uas, best_uas_epoch):
                # utils.ex.logger.info('early stop triggered')
                utils.writelog('early stop triggered')
                break

    def init_train(self, epoch: Optional[int] = None, report: bool = True, subepoch: Optional[int] = None) -> None:
        """ warm up nn, if fix, will not update counts.

        diff with train:
            prefer to use dmv`s parameters
            will not update dmv`s parameters
            ignore cfg.end2end (always be False)

        :param epoch: epochs
        :param report: do eval after each epoch
        """
        epoch = epoch or self.cfg.epoch_init
        subepoch = subepoch or self.cfg.epoch_nn

        for epoch_id in range(epoch):
            it = self.train_iter if self.cfg.device == 'cpu' else prefetcher(self.train_iter)
            nn_loss, n_instance, nn_epoch_total, n_batch = 0., 0, 0, 0
            for one_batch in it:
                tag_array = self.converter(None, one_batch.cluster_label_array)
                # assert (tag_array < 40).all()
                self.nn.eval()
                with torch.no_grad():
                    tdr_param = self.dmv.build_scores(tag_array, mode='tdr', using_fake=False)

                    # use nn to predict missing params
                    needed_param = [(i, m) for i, (p, m) in enumerate(zip(tdr_param, 'tdr')) if p is None]
                    nn_mode = ''.join([p[1] for p in needed_param])
                    if nn_mode != '':
                        arrays = namedtuple2dict(one_batch)
                        nn_param = self.nn.predict_pipeline(arrays, tag_array, mode=nn_mode)
                        for idx, (i, _) in enumerate(needed_param):
                            tdr_param[i] = nn_param[idx]
                tdr_param = [p.requires_grad_() for p in tdr_param]

                t = self.dmv.build_final_trans_scores(tdr_param[0], tdr_param[2], False)
                d = self.dmv.build_final_dec_scores(tdr_param[1], False)
                ll = self.dmv(t, d, one_batch.len_array)

                self.dmv.zero_grad()
                torch.sum(ll).backward()

                counts = {'t': tdr_param[0].grad.detach(),
                          'd': tdr_param[1].grad.detach(),
                          'r': tdr_param[2].grad.detach()}
                arrays = namedtuple2dict(one_batch)
                loss, nn_epoch = self.non_end2end_neural_train(tag_array, arrays, counts, epoch=subepoch)  # noqa
                nn_loss += loss
                nn_epoch_total += nn_epoch
                n_instance += len(tag_array)
                n_batch += 1

            nn_loss = nn_loss / n_instance
            utils.ex.logger.info(f'init epoch {epoch_id}, init.train.loss={nn_loss}, '
                                 f'run {nn_epoch_total / n_batch:.2f} epochs')
            utils.ex.log_scalar('init.train.loss', nn_loss, epoch_id)

            if report:
                uas_dev, ll_dev = 0.0, 0.0
                uas_test, ll_test = self.evaluate(self.test_ds)
                utils.ex.logger.info(f'init epoch {epoch_id}, init.dev.uas={uas_dev}, init.test.uas={uas_test}')
                utils.ex.log_scalar('init.dev.uas', uas_dev, epoch_id)
                utils.ex.log_scalar('init.test.uas', uas_test, epoch_id)

    def init_train_v2(self, dataset: Dataset, epoch: Optional[int] = None, report: bool = True, subepoch: Optional[int] = None):
        """
        unlike init_train, here directly use pretrained counts to train nn.
        in init_train, pretrained counts first init dmv, then run dmv get counts, then train nn.
        """
        heads, valences, head_valences = self.dmv.batch_recovery(dataset)
        len_array = dataset.get_all_len().to(self.cfg.device)
        _loader = iter(dataset.get_dataloader(False, len(dataset), False, 0, 0))
        _loader = prefetcher(_loader) if self.cfg.device != 'cpu' else _loader
        _batch = next(_loader)
        with torch.no_grad():
            # tag_array = self.converter(_batch[2], _batch[1])
            r, t, d = self.dmv.calcu_viterbi_count(heads, head_valences, valences, len_array)
            d = torch.sum(d, dim=2)
            t = self.nn.transition_param_helper_2(t)
        epoch = epoch or self.cfg.epoch_init
        subepoch = subepoch or self.cfg.epoch_nn

        for epoch_id in range(epoch):
            loader = dataset.get_dataloader(
                same_len=self.cfg.same_len, num_workers=self.cfg.num_worker, min_len=self.cfg.min_len_train,
                max_len=self.cfg.max_len_train, batch_size=self.cfg.e_batch_size, shuffle=self.cfg.shuffle, drop_last=self.cfg.drop_last)
            it = loader if self.cfg.device == 'cpu' else prefetcher(loader)
            nn_loss, n_instance = 0., 0
            for one_batch in it:
                tag_array = self.converter(None, one_batch.cluster_label_array)
                arrays = namedtuple2dict(one_batch)
                counts = {'r': r[one_batch.id_array], 't': t[one_batch.id_array], 'd': d[one_batch.id_array]}
                loss, nn_epoch = self.non_end2end_neural_train(tag_array, arrays, counts, epoch=subepoch)  # noqa
                nn_loss += loss
                n_instance += len(one_batch.id_array)
            if report:
                self.nn.eval()
                uas_train, ll_train = self.evaluate(self.train_ds)
                uas_dev, ll_dev = self.evaluate(self.dev_ds)
                uas_test, ll_test = self.evaluate(self.test_ds)
                # utils.ex.logger.info(f'init epoch {epoch_id}, init.dev.uas={uas_dev}, init.test.uas={uas_test}, loss={nn_loss}')
                utils.logger.info(f'init epoch {epoch_id}, init.train.uas= {uas_train}, init.dev.uas={uas_dev}, init.test.uas={uas_test}, loss={nn_loss}')

                # # TODO ugly adapt
                # if uas_test > 0.527:
                #     break
                # utils.ex.log_scalar('init.dev.uas', uas_dev, epoch_id)
                # utils.ex.log_scalar('init.test.uas', uas_test, epoch_id)

    def semi_supervised_init(self, dataset: Dataset, epoch: Optional[int] = None, report: bool = True,
                             subepoch: Optional[int] = None):
        supervised_ratio = self.cfg.supervised_ratio
        assert 0.0 <= supervised_ratio <= 1.0
        dataset_size = len(dataset)
        supervised_part = dataset[:int(dataset_size*supervised_ratio)]
        for inst in supervised_part:
            inst.supervised = True
        for inst in dataset:
            if not inst.supervised:
                inst.instance_weight = self.cfg.unsupervised_weight

        heads, valences, head_valences = self.dmv.batch_recovery(dataset)
        len_array = dataset.get_all_len().to(self.cfg.device)
        _loader = iter(dataset.get_dataloader(False, len(dataset), False, 0, 0))
        _loader = prefetcher(_loader) if self.cfg.device != 'cpu' else _loader
        _batch = next(_loader)
        with torch.no_grad():
            # tag_array = self.converter(_batch[2], _batch[1])
            r, t, d = self.dmv.calcu_viterbi_count(heads, head_valences, valences, len_array)
            d = torch.sum(d, dim=2)
            t = self.nn.transition_param_helper_2(t)
        epoch = epoch or self.cfg.epoch_init
        subepoch = subepoch or self.cfg.epoch_nn

        for epoch_id in range(epoch):
            loader = dataset.get_dataloader(
                same_len=self.cfg.same_len, num_workers=self.cfg.num_worker, min_len=self.cfg.min_len_train,
                max_len=self.cfg.max_len_train, batch_size=self.cfg.e_batch_size, shuffle=self.cfg.shuffle, drop_last=self.cfg.drop_last)
            it = loader if self.cfg.device == 'cpu' else prefetcher(loader)
            nn_loss, n_instance = 0., 0
            for one_batch in it:
                tag_array = self.converter(None, one_batch.cluster_label_array)
                arrays = namedtuple2dict(one_batch)
                counts = {'r': r[one_batch.id_array], 't': t[one_batch.id_array], 'd': d[one_batch.id_array]}
                loss, nn_epoch = self.non_end2end_neural_train(tag_array, arrays, counts, epoch=subepoch)  # noqa
                nn_loss += loss
                n_instance += len(one_batch.id_array)
            if report:
                self.nn.eval()
                uas_train, ll_train = self.evaluate(self.train_ds)
                uas_dev, ll_dev = self.evaluate(self.dev_ds)
                uas_test, ll_test = self.evaluate(self.test_ds)
                # utils.ex.logger.info(f'init epoch {epoch_id}, init.dev.uas={uas_dev}, init.test.uas={uas_test}, loss={nn_loss}')
                utils.logger.info(f'init epoch {epoch_id}, init.train.uas= {uas_train}, init.dev.uas={uas_dev}, init.test.uas={uas_test}, loss={nn_loss}')


    def semi_train(self, epoch: Optional[int] = None, report: bool = True, stop_hook: Optional[Callable] = None,
                   subepoch: Optional[int] = None) -> None:
        """ semisupervised train the model
        :param epoch: max train epoch
        :param report: do eval after each epoch
        :param stop_hook:
            (epoch_id, ll, best_ll, best_ll_epoch, uas, best_uas, best_uas_epoch) -> bool
        """

        # cannot end2end train this work
        assert self.cfg.end2end is False

        supervised_ratio = self.cfg.supervised_ratio
        assert 0.0 <= supervised_ratio <= 1.0
        dataset_size = len(self.train_ds)
        supervised_part = self.train_ds[:int(dataset_size*supervised_ratio)]
        for inst in supervised_part:
            inst.supervised = True
        for inst in self.train_ds:
            if not inst.supervised:
                inst.instance_weight = self.cfg.unsupervised_weight

        # supervised count
        epoch = epoch or self.cfg.epoch
        subepoch = subepoch or self.cfg.epoch_nn
        best_ll, best_uas = -1e30, -1.
        best_ll_epoch, best_uas_epoch = -1, -1

        heads, valences, head_valences = self.dmv.batch_recovery(self.train_ds)
        len_array = self.train_ds.get_all_len().to(self.cfg.device)
        supervised_mask = torch.tensor([inst.supervised for inst in self.train_ds], device=self.cfg.device)
        with torch.no_grad():
            # tag_array = self.converter(_batch[2], _batch[1])
            supervised_r, supervised_t, supervised_d = self.dmv.calcu_viterbi_count(heads, head_valences, valences, len_array)
            supervised_d = torch.sum(supervised_d, dim=2)
            supervised_t = self.nn.transition_param_helper_2(supervised_t)

        for epoch_id in range(epoch):
            it = self.train_iter if self.cfg.device == 'cpu' else prefetcher(self.train_iter)
            model_ll, nn_loss, n_instance, nn_total_epoch, n_batch = 0., 0., 0, 0, 0

            for one_batch in it:
                tag_array = self.converter(None, one_batch.cluster_label_array)
                arrays = namedtuple2dict(one_batch)

                self.nn.eval()
                with torch.no_grad():
                    tdr_param = self.nn.predict_pipeline(arrays, tag_array, mode='tdr')

                tdr_param = [p.requires_grad_() if p is not None else None for p in tdr_param]

                # use dmv to get missing params
                needed_param = [(i, m) for i, (p, m) in enumerate(zip(tdr_param, 'tdr')) if p is None]
                dmv_mode = ''.join([p[1] for p in needed_param])
                if dmv_mode != '':
                    dmv_param = self.dmv.build_scores(tag_array, mode=dmv_mode, using_fake=False)
                    for idx, (i, _) in enumerate(needed_param):
                        tdr_param[i] = dmv_param[idx]

                t = self.dmv.build_final_trans_scores(tdr_param[0], tdr_param[2], False)
                d = self.dmv.build_final_dec_scores(tdr_param[1], False)
                ll = self.dmv(t, d, one_batch.len_array)
                ll = torch.sum(ll)
                self.dmv.zero_grad()

                ll.backward()

                counts = {'t': tdr_param[0].grad.detach() if tdr_param[0].grad is not None else None,
                          'd': tdr_param[1].grad.detach() if tdr_param[1].grad is not None else None,
                          'r': tdr_param[2].grad.detach() if tdr_param[2].grad is not None else None}

                # update supervised part
                max_len = torch.max(one_batch.len_array)
                supervised_count = {'t': supervised_t[one_batch.id_array][:, :max_len, :max_len],
                                    'd': supervised_d[one_batch.id_array][:, :max_len],
                                    'r': supervised_r[one_batch.id_array][:, :max_len]}

                batch_supervised_mask = supervised_mask[one_batch.id_array]
                for key in counts.keys():
                    rank = len(counts[key].size())
                    m = batch_supervised_mask.reshape((-1, ) + (1, )*(rank-1))
                    counts[key] = (~m) * counts[key] + m * supervised_count[key]
                loss, nn_epoch = self.non_end2end_neural_train(tag_array, arrays, counts,
                    mode=''.join(set('tdr') - set(dmv_mode)), epoch=subepoch)
                # self.dmv.update_param_with_count(mode=dmv_mode)
                # self.dmv.normalize_param()

                nn_total_epoch += nn_epoch
                nn_loss += loss
                model_ll += ll.item()
                n_instance += len(tag_array)
                n_batch += 1

            # if epoch_id == 9:
            #     self.nn.vib.vib_beta = 1e-8
            #     print(self.nn.vib.vib_beta)

            nn_loss = nn_loss / n_instance
            model_ll = model_ll / n_instance
            # utils.ex.logger.info(
            #     f'epoch {epoch_id}, train.loss={nn_loss}, train.likelihood={model_ll}, '
            #     f'{"using end2end" if self.cfg.end2end else f"run {nn_total_epoch / n_batch:.2f} epochs"}')
            # utils.ex.log_scalar('train.loss', nn_loss, epoch_id)
            # utils.ex.log_scalar('train.likelihood', model_ll, epoch_id)
            utils.writelog(f'epoch {epoch_id}, train.loss={nn_loss}, train.likelihood={model_ll}, run {nn_total_epoch / n_batch:.2f} epochs')

            torch.save(self.dmv.state_dict(), self.workspace / 'all' / f'dmv_{epoch_id}')
            torch.save(self.nn.state_dict(), self.workspace / 'all' / f'nn_{epoch_id}')
            torch.save(torch.cuda.memory_snapshot(), self.workspace / 'all' / f'ss_{epoch_id}')

            if model_ll > best_ll:
                # utils.ex.logger.info('get new best ll')
                torch.save(self.dmv.state_dict(), self.workspace / 'best_ll' / 'dmv')
                torch.save(self.nn.state_dict(), self.workspace / 'best_ll' / 'nn')
                best_ll = model_ll
                best_ll_epoch = epoch_id

            if report:
                uas_train, ll_train = self.evaluate(self.train_ds)
                uas_dev, ll_dev = 0.0, 0.0
                uas_test, ll_test = self.evaluate(self.test_ds)
                # utils.ex.logger.info(f'epoch {epoch_id}, dev.uas={uas_dev}, test.uas={uas_test}')
                # utils.ex.log_scalar('dev.uas', uas_dev, epoch_id)
                # utils.ex.log_scalar('test.uas', uas_test, epoch_id)

                utils.writelog(f'epoch {epoch_id}, train.uas= {uas_train}, dev.uas={uas_dev}, test.uas={uas_test}')
                # nni.report_intermediate_result(uas_test)

                if uas_test > best_uas:
                    # utils.ex.logger.info('get new best test.uas')
                    torch.save(self.dmv.state_dict(), self.workspace / 'best_uas' / 'dmv')
                    torch.save(self.nn.state_dict(), self.workspace / 'best_uas' / 'nn')
                    best_uas = uas_test
                    best_uas_epoch = epoch_id

            if stop_hook and stop_hook(epoch_id, model_ll, best_ll, best_ll_epoch, uas_test, best_uas, best_uas_epoch):
                # utils.ex.logger.info('early stop triggered')
                utils.writelog('early stop triggered')
                break

    def non_end2end_neural_train(self, tag_array: Tensor, arrays: Dict[str, Tensor], counts: Dict[str, Tensor],
            mode: str = 'tdr', epoch: Optional[int] = None, batch_size: Optional[int] = None) -> Tuple[float, int]:
        self.nn.train()
        epoch = epoch or self.cfg.epoch_nn
        batch_size = batch_size or self.cfg.m_batch_size

        nn_loss_previous = 0.
        nn_loss_current = 0.
        for epoch_id in range(epoch):
            nn_loss_current = 0.
            idx = torch.randperm(len(tag_array))
            for batch_start in range(0, len(idx), batch_size):
                sub_idx = idx[batch_start: batch_start + batch_size]
                sub_tag_array = tag_array[sub_idx]
                sub_arrays = {k: (v[sub_idx] if v is not None else v) for k, v in arrays.items()}
                sub_counts = [counts[m][sub_idx] for m in mode]
                loss = self.nn.train_pipeline(sub_arrays, sub_tag_array, sub_counts, mode)
                self.nn.zero_grad()
                loss.backward()
                self.nn.optimizer.step()
                nn_loss_current += loss.item()

            if nn_loss_previous > 0.:
                diff_rate = abs(nn_loss_previous - nn_loss_current) / nn_loss_previous
                if diff_rate < self.cfg.neural_stop_criteria:
                    break

            nn_loss_previous = nn_loss_current

        return nn_loss_current, epoch_id  # noqa

    def evaluate(self, dataset: Union[ConllDataset, ScidtbDataset], prefer_nn: bool = True) -> Tuple[float, float]:
        self.nn.eval()

        pred, ll_sum = [], 0.
        loader = dataset.get_dataloader(False, len(dataset), False, 0, 0, 1, self.cfg.max_len_eval)
        it = loader if self.cfg.device == 'cpu' else prefetcher(loader)
        for one_batch in it:
            tag_array = self.converter(None, one_batch.cluster_label_array)
            arrays = namedtuple2dict(one_batch)
            with torch.no_grad():
                nn_t, nn_d, nn_r = self.nn.predict_pipeline(arrays, tag_array, mode='tdr')
                t, d, r = self.dmv.build_scores(tag_array, 'tdr', using_fake=False)
                if prefer_nn:
                    t, d, r = self.dmv.merge_scores((nn_t, nn_d, nn_r), (t, d, r))
                else:
                    t, d, r = self.dmv.merge_scores((t, d, r), (nn_t, nn_d, nn_r))
                d = self.dmv.build_final_dec_scores(d, using_fake=False)
                t = self.dmv.build_final_trans_scores(t, r, using_fake=False)

                out, ll = self.dmv.parse(t, d, one_batch.len_array, one_batch.sent_map_array, one_batch.paragraph_map_array)
                pred.extend(out)
                ll_sum += ll

        gold = [[arc[0] for arc in d.arcs] for d in dataset if len(d) <= self.cfg.max_len_eval]
        uas, root_uas = calculate_uas(pred, gold)

        return uas, ll_sum / len(dataset)

    @staticmethod
    def default_stop_hook(epoch_id: int, ll: float, best_ll: float, best_ll_epoch: int, uas: float,
            best_uas: float, best_uas_epoch: int) -> bool:
        return epoch_id > best_uas_epoch + 10

    @staticmethod
    def uas_stop_hook(epoch_id: int, ll: float, best_ll: float, best_ll_epoch: int, uas: float,
            best_uas: float, best_uas_epoch: int) -> bool:
        return uas < 0.48
