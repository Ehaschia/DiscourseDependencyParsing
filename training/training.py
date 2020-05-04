import jsonlines
import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda, optimizers, serializers

import utils
import treetk

import models
import parsing
import metrics

def train(model,
          decoder,
          max_epoch,
          batch_size,
          weight_decay,
          gradient_clipping,
          optimizer_name,
          train_databatch,
          dev_databatch,
          path_train,
          path_valid,
          path_snapshot,
          path_pred,
          path_gold):
    """
    :type model: Model
    :type decoder: IncrementalEisnerDecoder
    :type max_epoch: int
    :type batch_size: int
    :type weight_decay: float
    :type gradient_clipping: float
    :type optimizer_name: str
    :type train_databatch: DataBatch
    :type dev_databatch: DataBatch
    :type path_train: str
    :type path_valid: str
    :type path_snapshot: str
    :type path_pred: str
    :type path_gold: str
    :rtype: None
    """
    writer_train = jsonlines.Writer(open(path_train, "w"), flush=True)
    if dev_databatch is not None:
        writer_valid = jsonlines.Writer(open(path_valid, "w"), flush=True)

    # Optimizer preparation
    if optimizer_name == "adam":
        opt = optimizers.Adam()
    else:
        raise ValueError("Invalid optimizer_name=%s" % optimizer_name)
    opt.setup(model)
    if weight_decay > 0.0:
        opt.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    if gradient_clipping > 0.0:
        opt.add_hook(chainer.optimizer.GradientClipping(gradient_clipping))

    n_train = len(train_databatch)
    it = 0
    bestscore_holder = utils.BestScoreHolder(scale=1.0)
    bestscore_holder.init()

    if dev_databatch is not None:
        # Initial validation
        with chainer.using_config("train", False), chainer.no_backprop_mode():
            parsing.parse(model=model,
                          decoder=decoder,
                          databatch=dev_databatch,
                          path_pred=path_pred)
            scores = metrics.attachment_scores(
                        pred_path=path_pred,
                        gold_path=path_gold)
            scores["LAS"] *= 100.0
            scores["UAS"] *= 100.0
            scores["epoch"] = 0
            writer_valid.write(scores)
            utils.writelog(utils.pretty_format_dict(scores))
        # Saving
        bestscore_holder.compare_scores(scores["LAS"], 0)
        serializers.save_npz(path_snapshot, model)
        utils.writelog("Saved the model to %s" % path_snapshot)
    else:
        # Saving
        serializers.save_npz(path_snapshot, model)
        utils.writelog("Saved the model to %s" % path_snapshot)

    for epoch in range(1, max_epoch+1):
        perm = np.random.permutation(n_train)
        for inst_i in range(0, n_train, batch_size):
            # Processing one mini-batch

            # Init
            loss_attachment, acc_attachment = 0.0, 0.0
            loss_relation, acc_relation = 0.0, 0.0
            actual_batchsize = 0
            actual_total_arcs = 0

            # Mini-batch preparation
            batch_edu_ids = train_databatch.batch_edu_ids[perm[inst_i:inst_i+batch_size]]
            batch_edus = train_databatch.batch_edus[perm[inst_i:inst_i+batch_size]]
            batch_edus_postag = train_databatch.batch_edus_postag[perm[inst_i:inst_i+batch_size]]
            batch_edus_head = train_databatch.batch_edus_head[perm[inst_i:inst_i+batch_size]]
            batch_sbnds = train_databatch.batch_sbnds[perm[inst_i:inst_i+batch_size]]
            batch_pbnds = train_databatch.batch_pbnds[perm[inst_i:inst_i+batch_size]]
            batch_arcs = train_databatch.batch_arcs[perm[inst_i:inst_i+batch_size]]

            for edu_ids, edus, edus_postag, edus_head, sbnds, pbnds, gold_arcs \
                    in zip(batch_edu_ids,
                           batch_edus,
                           batch_edus_postag,
                           batch_edus_head,
                           batch_sbnds,
                           batch_pbnds,
                           batch_arcs):
                # Processing one instance

                # CHECK: Single ROOT?
                assert sum([1 for h,d,l in gold_arcs if h == 0]) == 1

                # Feature extraction
                edu_vectors = model.forward_edus(edus, edus_postag, edus_head) # (n_edus, bilstm_dim + tempfeat1_dim)
                same_sent_map = models.make_same_sent_map(edus=edus, sbnds=sbnds) # (n_edus, n_edus)

                # Positive and negative trees
                gold_heads = -np.ones((len(edus),), dtype=np.int32)
                for h,d,l in gold_arcs:
                    gold_heads[d] = h
                with chainer.using_config("train", False), chainer.no_backprop_mode():
                    # Positive
                    pos_arcs = [(h,d) for h,d,l in gold_arcs] # list of (int, int)
                    # Negative
                    neg_arcs = decoder.decode(
                                        model=model,
                                        edu_ids=edu_ids,
                                        edu_vectors=edu_vectors,
                                        same_sent_map=same_sent_map,
                                        sbnds=sbnds,
                                        pbnds=pbnds,
                                        use_sbnds=True,
                                        use_pbnds=False,
                                        gold_heads=gold_heads) # list of (int, int)
                    margin = compute_tree_distance(pos_arcs, neg_arcs, coef=1.0)

                # Scoring
                pred_scores = model.forward_arcs_for_attachment(
                                        edu_vectors=edu_vectors,
                                        same_sent_map=same_sent_map,
                                        batch_arcs=[pos_arcs, neg_arcs],
                                        aggregate=True) # (1+1, 1)

                # Labeling
                pred_relations = model.forward_arcs_for_labeling(
                                        edu_vectors=edu_vectors,
                                        same_sent_map=same_sent_map,
                                        batch_arcs=[pos_arcs]) # (1, n_arcs, n_relations)
                pred_relations = pred_relations[0] # (n_arcs, n_relations)

                # Attachment Loss
                loss_attachment += F.clip(pred_scores[1] + margin - pred_scores[0], 0.0, 10000000.0)

                # Ranked Accuracy
                pred_scores = F.reshape(pred_scores, (1, 1+1)) # (1, 1+1)
                gold_scores = np.zeros((1,), dtype=np.int32) # (1,)
                gold_scores = utils.convert_ndarray_to_variable(gold_scores, seq=False) # (1,)
                acc_attachment += F.accuracy(pred_scores, gold_scores)

                # Relation Loss/Accuracy
                gold_relations = [l for h,d,l in gold_arcs] # list of str
                gold_relations = [model.vocab_relation[r] for r in gold_relations] # list of int
                gold_relations = np.asarray(gold_relations, dtype=np.int32) # (n_arcs,)
                gold_relations = utils.convert_ndarray_to_variable(gold_relations, seq=False) # (n_arcs,)
                loss_relation += F.softmax_cross_entropy(pred_relations, gold_relations) * float(len(gold_relations))
                acc_relation += F.accuracy(pred_relations, gold_relations) * float(len(gold_relations))

                actual_batchsize += 1
                actual_total_arcs += len(gold_relations)

            # Backward & Update
            actual_batchsize = float(actual_batchsize)
            actual_total_arcs = float(actual_total_arcs)
            loss_attachment = loss_attachment / actual_batchsize
            acc_attachment = acc_attachment / actual_batchsize
            loss_relation = loss_relation / actual_total_arcs
            acc_relation = acc_relation / actual_total_arcs
            loss = loss_attachment + loss_relation
            model.zerograds()
            loss.backward()
            opt.update()
            it += 1

            # Write log
            loss_attachment_data = float(cuda.to_cpu(loss_attachment.data))
            acc_attachment_data = float(cuda.to_cpu(acc_attachment.data))
            loss_relation_data = float(cuda.to_cpu(loss_relation.data))
            acc_relation_data = float(cuda.to_cpu(acc_relation.data))
            out = {"iter": it,
                   "epoch": epoch,
                   "progress": "%d/%d" % (inst_i+actual_batchsize, n_train),
                   "progress_ratio": float(inst_i+actual_batchsize)/n_train*100.0,
                   "Attachment Loss": loss_attachment_data,
                   "Ranked Accuracy": acc_attachment_data * 100.0,
                   "Relation Loss": loss_relation_data,
                   "Relation Accuracy": acc_relation_data * 100.0}
            writer_train.write(out)
            utils.writelog(utils.pretty_format_dict(out))

        if dev_databatch is not None:
           # Validation
            with chainer.using_config("train", False), chainer.no_backprop_mode():
                parsing.parse(model=model,
                              decoder=decoder,
                              databatch=dev_databatch,
                              path_pred=path_pred)
                scores = metrics.attachment_scores(
                            pred_path=path_pred,
                            gold_path=path_gold)
                scores["LAS"] *= 100.0
                scores["UAS"] *= 100.0
                scores["epoch"] = epoch
                writer_valid.write(scores)
                utils.writelog(utils.pretty_format_dict(scores))
            # Saving
            did_update = bestscore_holder.compare_scores(scores["LAS"], epoch)
            if did_update:
                serializers.save_npz(path_snapshot, model)
                utils.writelog("Saved the model to %s" % path_snapshot)
            # Finished?
            if bestscore_holder.ask_finishing(max_patience=10):
                utils.writelog("Patience %d is over. Training finished successfully." % bestscore_holder.patience)
                writer_train.close()
                if dev_databatch is not None:
                    writer_valid.close()
                return
        else:
            # No validation
            # Saving
            serializers.save_npz(path_snapshot, model)
            # We continue training until it reaches the maximum number of epochs.

def compute_tree_distance(arcs1, arcs2, coef):
    """
    :type arcs1: list of (int, int)
    :type arcs2: list of (int, int)
    :type coef: float
    :rtype: float
    """
    assert len(arcs1) == len(arcs2)

    dtree1 = treetk.arcs2dtree(arcs1)
    dtree2 = treetk.arcs2dtree(arcs2)

    dist = 0.0
    for d in range(len(dtree1.tokens)):
        if d == 0:
            continue
        h1, _ = dtree1.get_head(d)
        h2, _ = dtree2.get_head(d)
        if h1 != h2:
            dist += 1.0
    dist = coef * dist
    return dist

