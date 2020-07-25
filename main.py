import argparse
import os
import time

import jsonlines
import numpy as np
# import chainer
# import chainer.functions as F
# from chainer import cuda, optimizers, serializers
import pyprind
import torch
import torch.optim as optimizers
import torch.nn.functional as F

import treesamplers
import utils
import treetk

import dataloader
import models
import decoders
import metrics


def main(args):
    ##################
    # Arguments
    gpu = args.gpu
    model_name = args.model
    path_config = args.config
    trial_name = args.name
    actiontype = args.actiontype
    max_epoch = args.max_epoch
    dev_size = args.dev_size
    initial_tree_sampling = args.initial_tree_sampling

    # Check
    assert actiontype in ["train", "evaluate", "unsup_train"]
    if actiontype == "train":
        assert max_epoch > 0

    if trial_name is None or trial_name == "None":
        trial_name = utils.get_current_time()

    ##################
    # Path setting
    config = utils.Config(path_config)

    basename = "%s.%s.%s" \
            % (model_name,
               utils.get_basename_without_ext(path_config),
               trial_name)

    if actiontype == "train":
        path_log = os.path.join(config.getpath("results"), basename + ".training.log")
    else:
        path_log = os.path.join(config.getpath("results"), basename + ".evaluation.log")
    path_train = os.path.join(config.getpath("results"), basename + ".training.jsonl")
    path_valid = os.path.join(config.getpath("results"), basename + ".validation.jsonl")
    path_snapshot = os.path.join(config.getpath("results"), basename + ".model")
    path_pred = os.path.join(config.getpath("results"), basename + ".evaluation.arcs")
    path_eval = os.path.join(config.getpath("results"), basename + ".evaluation.json")

    utils.set_logger(path_log)

    ##################
    # Random seed
    random_seed = trial_name
    random_seed = utils.hash_string(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # cuda.cupy.random.seed(random_seed)

    ##################
    # Log so far
    utils.writelog("gpu=%d" % gpu)
    utils.writelog("model_name=%s" % model_name)
    utils.writelog("path_config=%s" % path_config)
    utils.writelog("trial_name=%s" % trial_name)
    utils.writelog("actiontype=%s" % actiontype)
    utils.writelog("max_epoch=%s" % max_epoch)
    utils.writelog("dev_size=%s" % dev_size)

    utils.writelog("path_log=%s" % path_log)
    utils.writelog("path_train=%s" % path_train)
    utils.writelog("path_valid=%s" % path_valid)
    utils.writelog("path_snapshot=%s" % path_snapshot)
    utils.writelog("path_pred=%s" % path_pred)
    utils.writelog("path_eval=%s" % path_eval)

    utils.writelog("random_seed=%d" % random_seed)

    ##################
    # Data preparation
    begin_time = time.time()

    train_dataset = dataloader.read_scidtb("train", "", relation_level="coarse-grained")
    test_dataset = dataloader.read_scidtb("test", "gold", relation_level="coarse-grained")
    dev_dataset = dataloader.read_scidtb("dev", "gold", relation_level="coarse-grained")
    vocab_word = utils.read_vocab(os.path.join(config.getpath("data"), "scidtb-vocab", "words.vocab.txt"))
    vocab_postag = utils.read_vocab(os.path.join(config.getpath("data"), "scidtb-vocab", "postags.vocab.txt"))
    vocab_deprel = utils.read_vocab(os.path.join(config.getpath("data"), "scidtb-vocab", "deprels.vocab.txt"))
    vocab_relation = utils.read_vocab(os.path.join(config.getpath("data"), "scidtb-vocab", "relations.coarse.vocab.txt"))

    end_time = time.time()
    utils.writelog("Loaded the corpus. %f [sec.]" % (end_time - begin_time))

    ##################
    # Hyper parameters
    word_dim = config.getint("word_dim")
    postag_dim = config.getint("postag_dim")
    deprel_dim = config.getint("deprel_dim")
    lstm_dim = config.getint("lstm_dim")
    mlp_dim = config.getint("mlp_dim")
    batch_size = config.getint("batch_size")
    weight_decay = config.getfloat("weight_decay")
    gradient_clipping = config.getfloat("gradient_clipping")
    optimizer_name = config.getstr("optimizer_name")
    n_init_epochs = config.getint("n_init_epochs")
    negative_size = config.getint("negative_size")

    utils.writelog("word_dim=%d" % word_dim)
    utils.writelog("postag_dim=%d" % postag_dim)
    utils.writelog("depre_dim=%d" % deprel_dim)
    utils.writelog("lstm_dim=%d" % lstm_dim)
    utils.writelog("mlp_dim=%d" % mlp_dim)
    utils.writelog("batch_size=%d" % batch_size)
    utils.writelog("weight_decay=%f" % weight_decay)
    utils.writelog("gradient_clipping=%f" % gradient_clipping)
    utils.writelog("optimizer_name=%s" % optimizer_name)

    ##################
    # Model preparation
    # cuda.get_device(gpu).use()
    device = torch.device('cuda')

    # Initialize a model
    initialW = utils.read_word_embedding_matrix(
                        path=config.getpath("pretrained_word_embeddings"),
                        dim=word_dim,
                        vocab=vocab_word,
                        scale=0.0)
    if model_name == "arcfactoredmodel":
        template_feature_extractor1 = models.TemplateFeatureExtractor1(
                                                dataset=train_dataset,
                                                vocab_relation=vocab_relation)
        template_feature_extractor2 = models.TemplateFeatureExtractor2()
        utils.writelog("Template feature (1) size=%d" % template_feature_extractor1.feature_size)
        utils.writelog("Template feature (2) size=%d" % template_feature_extractor2.feature_size)
        if actiontype == "train":
            for template in template_feature_extractor1.templates:
                dim = template_feature_extractor1.template2dim[template]
                utils.writelog("Template feature (1) #%s %s" % (dim, template))
            for template in template_feature_extractor2.templates:
                dim = template_feature_extractor2.template2dim[template]
                utils.writelog("Template feature (2) #%s %s" % (dim, template))
        model = models.ArcFactoredModel(
                        vocab_word=vocab_word,
                        vocab_postag=vocab_postag,
                        vocab_deprel=vocab_deprel,
                        vocab_relation=vocab_relation,
                        word_dim=word_dim,
                        postag_dim=postag_dim,
                        deprel_dim=deprel_dim,
                        lstm_dim=lstm_dim,
                        mlp_dim=mlp_dim,
                        initialW=initialW,
                        template_feature_extractor1=template_feature_extractor1,
                        template_feature_extractor2=template_feature_extractor2,
                        device=device)
        model.to(device)
    elif model_name == "emtrymodel":
        template_feature_extractor1 = models.TemplateFeatureExtractor1(
                                                dataset=train_dataset,
                                                vocab_relation=vocab_relation)
        template_feature_extractor2 = models.TemplateFeatureExtractor2()
        utils.writelog("Template feature (1) size=%d" % template_feature_extractor1.feature_size)
        utils.writelog("Template feature (2) size=%d" % template_feature_extractor2.feature_size)
        if actiontype == "train":
            for template in template_feature_extractor1.templates:
                dim = template_feature_extractor1.template2dim[template]
                utils.writelog("Template feature (1) #%s %s" % (dim, template))
            for template in template_feature_extractor2.templates:
                dim = template_feature_extractor2.template2dim[template]
                utils.writelog("Template feature (2) #%s %s" % (dim, template))
        model = models.EmTryModel(
                        vocab_word=vocab_word,
                        vocab_postag=vocab_postag,
                        vocab_deprel=vocab_deprel,
                        vocab_relation=vocab_relation,
                        word_dim=word_dim,
                        postag_dim=postag_dim,
                        deprel_dim=deprel_dim,
                        lstm_dim=lstm_dim,
                        mlp_dim=mlp_dim,
                        initialW=initialW,
                        template_feature_extractor1=template_feature_extractor1,
                        template_feature_extractor2=template_feature_extractor2,
                        device=device)
        model.to(device)
    else:
        raise ValueError("Invalid model_name=%s" % model_name)
    utils.writelog("Initialized the model ``%s''" % model_name)

    # Load pre-trained parameters
    if actiontype.find("train") == -1:
        # serializers.load_npz(path_snapshot, model)
        model.load_state_dict(torch.load(path_snapshot))
        utils.writelog("Loaded trained parameters from %s" % path_snapshot)

    ##################
    # Decoder preparation
    decoder = decoders.IncrementalEisnerDecoder()

    ##################
    # Training / evaluation
    if actiontype == "train":
        model.train()
        if dev_size > 0:
            # Training with cross validation
            train_dataset, dev_dataset = utils.split_dataset(dataset=train_dataset, n_dev=dev_size, seed=None)
            with open(os.path.join(config.getpath("results"), basename + ".valid_gold.arcs"), "w") as f:
                for data in dev_dataset:
                    arcs = data.arcs
                    arcs = ["%s-%s-%s" % (h,d,r) for (h,d,r) in arcs]
                    f.write("%s\n" % " ".join(arcs))
        else:
            # Training with the full training set
            dev_dataset = None

        train(
            model=model,
            decoder=decoder,
            max_epoch=max_epoch,
            batch_size=batch_size,
            weight_decay=weight_decay,
            gradient_clipping=gradient_clipping,
            optimizer_name=optimizer_name,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            path_train=path_train,
            path_valid=path_valid,
            path_snapshot=path_snapshot,
            path_pred=os.path.join(config.getpath("results"), basename + ".valid_pred.arcs"),
            path_gold=os.path.join(config.getpath("results"), basename + ".valid_gold.arcs"))
    elif actiontype == "unsup_train":

        sampler = treesamplers.TreeSampler(initial_tree_sampling.split("_"))

        model.train()
        if dev_size > 0:
            # Training with cross validation
            train_dataset, dev_dataset = utils.split_dataset(dataset=train_dataset, n_dev=dev_size, seed=None)
            with open(os.path.join(config.getpath("results"), basename + ".valid_gold.arcs"), "w") as f:
                for data in dev_dataset:
                    arcs = data.arcs
                    arcs = ["%s-%s-%s" % (h,d,r) for (h,d,r) in arcs]
                    f.write("%s\n" % " ".join(arcs))
        else:
            # Training with the full training set
            dev_dataset = None
        unsup_train(
                    model=model,
                    decoder=decoder,
                    max_epoch=max_epoch,
                    batch_size=batch_size,
                    weight_decay=weight_decay,
                    gradient_clipping=gradient_clipping,
                    optimizer_name=optimizer_name,
                    train_dataset=train_dataset,
                    dev_dataset=dev_dataset,
                    path_train=path_train,
                    path_valid=path_valid,
                    path_snapshot=path_snapshot,
                    path_pred=os.path.join(config.getpath("results"), basename + ".valid_pred.arcs"),
                    path_gold=os.path.join(config.getpath("results"), basename + ".valid_gold.arcs"),
                    negative_size=negative_size,
                    n_init_epochs=n_init_epochs,
                    sampler=sampler)
    elif actiontype == "evaluate":
        model.eval()
        with torch.no_grad():
            # Test
            parse(
                model=model,
                decoder=decoder,
                dataset=test_dataset,
                path_pred=path_pred)
            scores = metrics.attachment_scores(
                        pred_path=path_pred,
                        gold_path=os.path.join(config.getpath("data"), "scidtb", "preprocessed", "test", "gold", "gold.arcs"))
            scores["LAS"] *= 100.0
            scores["UAS"] *= 100.0
            utils.write_json(path_eval, scores)
            utils.writelog(utils.pretty_format_dict(scores))

    utils.writelog("Done: %s" % basename)


def train(model,
          decoder,
          max_epoch,
          batch_size,
          weight_decay,
          gradient_clipping,
          optimizer_name,
          train_dataset,
          dev_dataset,
          path_train,
          path_valid,
          path_snapshot,
          path_pred,
          path_gold):
    """
    :type model: ArcFactoredModel
    :type decoder: IncrementalEisnerDecoder
    :type max_epoch: int
    :type batch_size: int
    :type weight_decay: float
    :type gradient_clipping: float
    :type optimizer_name: str
    :type train_dataset: numpy.ndarray
    :type dev_dataset: numpy.ndarray
    :type path_train: str
    :type path_valid: str
    :type path_snapshot: str
    :type path_pred: str
    :type path_gold: str
    :rtype: None
    """
    writer_train = jsonlines.Writer(open(path_train, "w"), flush=True)
    if dev_dataset is not None:
        writer_valid = jsonlines.Writer(open(path_valid, "w"), flush=True)

    # Optimizer preparation
    parameters_need_update = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer_name == "adam":
        opt = optimizers.Adam(parameters_need_update, weight_decay=weight_decay)
    else:
        raise ValueError("Invalid optimizer_name=%s" % optimizer_name)

    # if gradient_clipping > 0.0:
    #     opt.add_hook(chainer.optimizer.GradientClipping(gradient_clipping))

    n_train = len(train_dataset)
    it = 0
    bestscore_holder = utils.BestScoreHolder(scale=1.0)
    bestscore_holder.init()

    if dev_dataset is not None:
        # Initial validation
        model.eval()
        with torch.no_grad():
            parse(
                model=model,
                decoder=decoder,
                dataset=dev_dataset,
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
        torch.save(model.state_dict(), path_snapshot)
        utils.writelog("Saved the model to %s" % path_snapshot)
    else:
        # Saving
        torch.save(model.state_dict(), path_snapshot)
        utils.writelog("Saved the model to %s" % path_snapshot)

    model.train()

    for epoch in range(1, max_epoch+1):

        perm = np.random.permutation(n_train)

        for inst_i in range(0, n_train, batch_size):

            ### Mini batch

            # Init
            loss_attachment, acc_attachment = 0.0, 0.0
            loss_relation, acc_relation = 0.0, 0.0
            actual_batchsize = 0
            actual_total_arcs = 0

            for data in train_dataset[perm[inst_i:inst_i+batch_size]]:

                ### One data instance

                edu_ids = data.edu_ids
                edus = data.edus
                edus_postag = data.edus_postag
                edus_head = data.edus_head
                sbnds = data.sbnds
                pbnds = data.pbnds
                gold_arcs = data.arcs

                # CHECK: Single ROOT?
                assert sum([1 for h,d,l in gold_arcs if h == 0]) == 1

                # Feature extraction
                edu_vectors = model.forward_edus(edus, edus_postag, edus_head) # (n_edus, bilstm_dim + tempfeat1_dim)
                same_sent_map = models.make_same_sent_map(edus=edus, sbnds=sbnds) # (n_edus, n_edus)

                # Positive and negative trees
                gold_heads = -np.ones((len(edus),), dtype=np.int32)
                for h,d,l in gold_arcs:
                    gold_heads[d] = h
                with torch.no_grad():
                    # Positive
                    pos_arcs = [(h,d) for h,d,l in gold_arcs] # list of (int, int)
                    # Negative
                    arc_scores = precompute_all_arc_scores(
                                        model=model,
                                        edu_ids=edu_ids,
                                        edu_vectors=edu_vectors,
                                        same_sent_map=same_sent_map)
                    neg_arcs = decoder.decode(
                                        arc_scores=arc_scores,
                                        edu_ids=edu_ids,
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
                loss_attachment += torch.clamp(pred_scores[1] + margin - pred_scores[0], 0.0, 10000000.0)

                # Ranking Accuracy
                pred_scores = torch.reshape(pred_scores, (1, 1+1)) # (1, 1+1)
                gold_labels = np.zeros((1,), dtype=np.int32) # (1,)
                pred_labels = torch.argmax(pred_scores, dim=-1).cpu().numpy()
                acc_attachment += np.sum(np.equal(pred_labels, gold_labels))

                # Relation Loss/Accuracy
                gold_relations = [l for h,d,l in gold_arcs] # list of str
                gold_relations = [model.vocab_relation[r] for r in gold_relations] # list of int
                gold_relations = np.asarray(gold_relations, dtype=np.int32) # (n_arcs,)
                # gold_relations = utils.convert_ndarray_to_variable(gold_relations, seq=False) # (n_arcs,)
                loss_relation += F.cross_entropy(pred_relations,
                                                 torch.tensor(gold_relations).long().to(pred_relations.device),
                                                 reduction='sum')
                pred_labels = torch.argmax(pred_scores, dim=-1).cpu().numpy()
                acc_relation += np.sum(np.equal(pred_labels, gold_relations))

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
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), gradient_clipping)
            opt.step()
            opt.zero_grad()
            it += 1

            # Write log
            loss_attachment_data = loss_attachment.item()
            acc_attachment_data = acc_attachment.item()
            loss_relation_data = loss_relation.item()
            acc_relation_data = acc_relation.item()
            out = {"iter": it,
                   "epoch": epoch,
                   "progress": "%d/%d" % (inst_i+actual_batchsize, n_train),
                   "progress_ratio": float(inst_i+actual_batchsize)/n_train*100.0,
                   "Attachment Loss": loss_attachment_data,
                   "Ranking Accuracy": acc_attachment_data * 100.0,
                   "Relation Loss": loss_relation_data,
                   "Relation Accuracy": acc_relation_data * 100.0}
            writer_train.write(out)
            utils.writelog(utils.pretty_format_dict(out))

        if dev_dataset is not None:
           # Validation
            model.eval()
            with torch.no_grad():
                parse(
                    model=model,
                    decoder=decoder,
                    dataset=dev_dataset,
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
                torch.save(model.state_dict(), path_snapshot)
                utils.writelog("Saved the model to %s" % path_snapshot)
            # Finished?
            if bestscore_holder.ask_finishing(max_patience=10):
                utils.writelog("Patience %d is over. Training finished successfully." % bestscore_holder.patience)
                writer_train.close()
                if dev_dataset is not None:
                    writer_valid.close()
                return
        else:
            # No validation
            # Saving
            torch.save(model.state_dict(), path_snapshot)
            # We continue training until it reaches the maximum number of epochs.



def unsup_train(model,
                decoder,
                max_epoch,
                batch_size,
                weight_decay,
                gradient_clipping,
                optimizer_name,
                train_dataset,
                dev_dataset,
                path_train,
                path_valid,
                path_snapshot,
                path_pred,
                path_gold,
                n_init_epochs,
                negative_size,
                sampler):
    """
    :type model: ArcFactoredModel
    :type decoder: IncrementalEisnerDecoder
    :type max_epoch: int
    :type batch_size: int
    :type weight_decay: float
    :type gradient_clipping: float
    :type optimizer_name: str
    :type train_dataset: numpy.ndarray
    :type dev_dataset: numpy.ndarray
    :type path_train: str
    :type path_valid: str
    :type path_snapshot: str
    :type path_pred: str
    :type path_gold: str
    :type n_init_epochs: int
    :type negative_size: int
    :rtype: None
    """
    writer_train = jsonlines.Writer(open(path_train, "w"), flush=True)
    if dev_dataset is not None:
        writer_valid = jsonlines.Writer(open(path_valid, "w"), flush=True)

    boundary_flags = [(True,False)]
    assert negative_size >= len(boundary_flags)
    negative_tree_sampler = treesamplers.NegativeTreeSampler()

    # Optimizer preparation
    parameters_need_update = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer_name == "adam":
        opt = optimizers.Adam(parameters_need_update, weight_decay=weight_decay)
    else:
        raise ValueError("Invalid optimizer_name=%s" % optimizer_name)

    # if gradient_clipping > 0.0:
    #     opt.add_hook(chainer.optimizer.GradientClipping(gradient_clipping))

    n_train = len(train_dataset)
    it = 0
    bestscore_holder = utils.BestScoreHolder(scale=1.0)
    bestscore_holder.init()

    if dev_dataset is not None:
        # Initial validation
        model.eval()
        with torch.no_grad():
            parse(
                model=model,
                decoder=decoder,
                dataset=dev_dataset,
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
        torch.save(model.state_dict(), path_snapshot)
        utils.writelog("Saved the model to %s" % path_snapshot)
    else:
        # Saving
        torch.save(model.state_dict(), path_snapshot)
        utils.writelog("Saved the model to %s" % path_snapshot)

    model.train()

    for epoch in range(1, max_epoch+1):

        perm = np.random.permutation(n_train)

        ########## E-Step (BEGIN) ##########
        utils.writelog("E step ===>")
        prog_bar = pyprind.ProgBar(n_train)

        for inst_i in range(0, n_train, batch_size):

            ### Mini batch

            for data in train_dataset[inst_i:inst_i + batch_size]:

                ### One data instance

                edu_ids = data.edu_ids
                edus = data.edus
                edus_postag = data.edus_postag
                edus_head = data.edus_head
                sbnds = data.sbnds
                pbnds = data.pbnds
                model.eval()
                with torch.no_grad():

                    # Feature extraction
                    edu_vectors = model.forward_edus(edus, edus_postag, edus_head)  # (n_edus, bilstm_dim)
                    same_sent_map = models.make_same_sent_map(edus=edus, sbnds=sbnds)  # (n_edus, n_edus)
                    # Positive tree
                    if epoch <= n_init_epochs:
                        pos_arcs = sampler.sample(
                            inputs=edu_ids,
                            edus=edus,
                            edus_head=edus_head,
                            sbnds=sbnds,
                            pbnds=pbnds)
                    else:
                        arc_scores = precompute_all_arc_scores(
                                            model=model,
                                            edu_ids=edu_ids,
                                            edu_vectors=edu_vectors,
                                            same_sent_map=same_sent_map)

                        pos_arcs = decoder.decode(
                                        arc_scores=arc_scores,
                                        edu_ids=edu_ids,
                                        sbnds=sbnds,
                                        pbnds=pbnds,
                                        use_sbnds=True,
                                        use_pbnds=False)
                        pos_arcs = [(h, d, "*") for h,d in pos_arcs]
                    data.arcs = pos_arcs  # NOTE
                    prog_bar.update()
        ########## E-Step (END) ##########

        ########## M-Step (BEGIN) ##########
        utils.writelog("M step ===>")
        model.train()
        for inst_i in range(0, n_train, batch_size):

            ### Mini batch

            # Init
            loss_attachment, acc_attachment = 0.0, 0.0
            actual_batchsize = 0
            actual_total_arcs = 0

            for data in train_dataset[perm[inst_i:inst_i+batch_size]]:

                ### One data instance

                edu_ids = data.edu_ids
                edus = data.edus
                edus_postag = data.edus_postag
                edus_head = data.edus_head
                sbnds = data.sbnds
                pbnds = data.pbnds
                pos_arcs = data.arcs

                # CHECK: Single ROOT?
                assert sum([1 for h,d,l in pos_arcs if h == 0]) == 1

                # Feature extraction
                edu_vectors = model.forward_edus(edus, edus_postag, edus_head) # (n_edus, bilstm_dim + tempfeat1_dim)
                same_sent_map = models.make_same_sent_map(edus=edus, sbnds=sbnds) # (n_edus, n_edus)

                # negative trees
                pos_heads = -np.ones((len(edus),), dtype=np.int32)
                margins = []
                pos_neg_arcs = [pos_arcs]
                for h,d,l in pos_arcs:
                    pos_heads[d] = h
                with torch.no_grad():
                    for use_sbnds, use_pbnds in boundary_flags:
                        arc_scores = precompute_all_arc_scores(
                                            model=model,
                                            edu_ids=edu_ids,
                                            edu_vectors=edu_vectors,
                                            same_sent_map=same_sent_map)

                        # Negative
                        neg_arcs = decoder.decode(
                                            arc_scores=arc_scores,
                                            edu_ids=edu_ids,
                                            sbnds=sbnds,
                                            pbnds=pbnds,
                                            use_sbnds=use_sbnds,
                                            use_pbnds=use_pbnds,
                                            gold_heads=pos_heads) # list of (int, int)
                        margin = compute_tree_distance(pos_arcs, neg_arcs, coef=1.0)
                        margins.append(margin)
                        pos_neg_arcs.append(neg_arcs)
                    for _ in range(negative_size - len(boundary_flags)):
                        neg_arcs = negative_tree_sampler.sample(inputs=edu_ids, sbnds=sbnds, pbnds=pbnds)
                        margin = compute_tree_distance(pos_arcs, neg_arcs, coef=1.0)
                        margins.append(margin)
                        pos_neg_arcs.append(neg_arcs)

                pred_scores = model.forward_arcs_for_attachment(
                                        edu_vectors=edu_vectors,
                                        same_sent_map=same_sent_map,
                                        batch_arcs=pos_neg_arcs,
                                        aggregate=True) # (1+1, 1)

                # Attachment Loss
                for neg_i in range(negative_size):
                    loss_attachment += torch.clamp(pred_scores[1+neg_i] + margins[neg_i] - pred_scores[0], 0.0, 10000000.0)

                # Ranking Accuracy
                pred_scores = torch.reshape(pred_scores, (1, len(pos_neg_arcs))) # (1, 1+1)
                gold_labels = np.zeros((1,), dtype=np.int32) # (1,)
                pred_labels = torch.argmax(pred_scores, dim=-1).cpu().numpy()
                acc_attachment += np.sum(np.equal(pred_labels, gold_labels))

                actual_batchsize += 1


            # Backward & Update
            actual_batchsize = float(actual_batchsize)
            loss_attachment = loss_attachment / actual_batchsize
            acc_attachment = acc_attachment / actual_batchsize
            loss = loss_attachment # + loss_relation
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), gradient_clipping)
            opt.step()
            opt.zero_grad()
            it += 1

            # Write log
            loss_attachment_data = loss_attachment.item()
            acc_attachment_data = acc_attachment.item()
            out = {"iter": it,
                   "epoch": epoch,
                   "progress": "%d/%d" % (inst_i+actual_batchsize, n_train),
                   "progress_ratio": float(inst_i+actual_batchsize)/n_train*100.0,
                   "Attachment Loss": loss_attachment_data,
                   "Ranking Accuracy": acc_attachment_data * 100.0}
            writer_train.write(out)
            utils.writelog(utils.pretty_format_dict(out))

        if dev_dataset is not None:
           # Validation
            model.eval()
            with torch.no_grad():
                parse(
                    model=model,
                    decoder=decoder,
                    dataset=dev_dataset,
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
                torch.save(model.state_dict(), path_snapshot)
                utils.writelog("Saved the model to %s" % path_snapshot)
            # Finished?
            if bestscore_holder.ask_finishing(max_patience=10):
                utils.writelog("Patience %d is over. Training finished successfully." % bestscore_holder.patience)
                writer_train.close()
                if dev_dataset is not None:
                    writer_valid.close()
                return
        else:
            # No validation
            # Saving
            torch.save(model.state_dict(), path_snapshot)
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


def parse(model, decoder, dataset, path_pred):
    """
    :type model: ArcFactoredModel
    :type decoder: IncrementalEisnerDecoder
    :type dataset: numpy.ndarray
    :type path_pred: str
    :rtype: None
    """
    with open(path_pred, "w") as f:

        for data in pyprind.prog_bar(dataset):

            edu_ids = data.edu_ids
            edus = data.edus
            edus_postag = data.edus_postag
            edus_head = data.edus_head
            sbnds = data.sbnds
            pbnds = data.pbnds

            # Feature extraction
            edu_vectors = model.forward_edus(edus, edus_postag, edus_head) # (n_edus, bilstm_dim + tempfeat1_dim)
            same_sent_map = models.make_same_sent_map(edus=edus, sbnds=sbnds) # (n_edus, n_edus)

            # Parsing (attachment)
            arc_scores = precompute_all_arc_scores(
                                model=model,
                                edu_ids=edu_ids,
                                edu_vectors=edu_vectors,
                                same_sent_map=same_sent_map)
            unlabeled_arcs = decoder.decode(
                                arc_scores=arc_scores,
                                edu_ids=edu_ids,
                                sbnds=sbnds,
                                pbnds=pbnds,
                                use_sbnds=True,
                                use_pbnds=True) # list of (int, int)

            # Parsing (labeling)
            logits_rel = model.forward_arcs_for_labeling(
                                    edu_vectors=edu_vectors,
                                    same_sent_map=same_sent_map,
                                    batch_arcs=[unlabeled_arcs]) # (1, n_arcs, n_labels)
            logits_rel = logits_rel.cpu().numpy()[0] # (n_spans, n_relations)
            relations = np.argmax(logits_rel, axis=1) # (n_spans,)
            relations = [model.ivocab_relation[r] for r in relations] # list of str
            labeled_arcs = [(h,d,r) for (h,d),r in zip(unlabeled_arcs, relations)] # list of (int, int, str)

            dtree = treetk.arcs2dtree(arcs=labeled_arcs)
            labeled_arcs = ["%s-%s-%s" % (x[0],x[1],x[2]) for x in dtree.tolist()]
            f.write("%s\n" % " ".join(labeled_arcs))


def precompute_all_arc_scores(model, edu_ids, edu_vectors, same_sent_map):
    """
    :type model: ArcFactoredModel
    :type edu_ids: list of int (length=n_edus)
    :type edu_vectors: Variable(shape=(n_edus*, bilstm_dim + tempfeat1_dim), dtype=np.float32)
    :type same_sent_map: numpy.ndarray(shape=(n_edus*, n_edus*), dtype=np.int32)
    :rtype: numpy.ndarray(shape=(n_edus, n_edus), dtype="float")
    """
    n_edus = len(edu_ids)

    result = np.zeros((n_edus, n_edus), dtype="float")

    # Aggregating patterns
    arcs = []
    for h in range(0, n_edus):
        for d in range(0, n_edus):
            if h == d:
                continue
            arc = (h, d)
            arcs.append(arc)

    # Shifting: local position -> global position
    arcs = [(edu_ids[h], edu_ids[d]) for h,d in arcs]

    # Scoring
    arc_scores = model.forward_arcs_for_attachment(
                                edu_vectors=edu_vectors,
                                same_sent_map=same_sent_map,
                                batch_arcs=[arcs],
                                aggregate=False) # (1, n_arcs, 1)
    arc_scores = arc_scores.cpu().numpy()[0] # (n_arcs, 1)
    for arc_i, (h, d) in enumerate(arcs):
        result[h,d] = float(arc_scores[arc_i])
    return result

def precompute_all_arc_scores_variable(model, edu_ids, edu_vectors, same_sent_map, gold_arcs):
    """
    :type model: ArcFactoredModel
    :type edu_ids: list of int (length=n_edus)
    :type edu_vectors: Variable(shape=(n_edus*, bilstm_dim + tempfeat1_dim), dtype=np.float32)
    :type same_sent_map: numpy.ndarray(shape=(n_edus*, n_edus*), dtype=np.int32)
    :rtype: numpy.ndarray(shape=(n_edus, n_edus), dtype="float")
    """
    n_edus = len(edu_ids)

    # Aggregating patterns
    arcs = []
    for h in range(0, n_edus):
        for d in range(0, n_edus):
            if h == d:
                continue
            arc = (h, d)
            arcs.append(arc)

    # Shifting: local position -> global position
    arcs = [(edu_ids[h], edu_ids[d]) for h,d in arcs]

    # Scoring
    arc_scores = model.forward_arcs_for_attachment(
                                edu_vectors=edu_vectors,
                                same_sent_map=same_sent_map,
                                batch_arcs=[arcs],
                                aggregate=False).squeeze() # (1, n_arcs, 1)
    # result = np.zeros((n_edus, n_edus), dtype="float")
    result = arc_scores.new(n_edus, n_edus)

    for arc_i, (h, d) in enumerate(arcs):
        result[h,d] += arc_scores[arc_i]
    for i in range(n_edus):
        result[i, i] = -1e5
    gold_idx = torch.zeros(n_edus).long().to(result.device)
    # loss
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    for arc_i, (h, d) in enumerate(arcs):
        if (h, d) in gold_arcs:
            gold_idx[d] = h
    final_loss = loss(result.t(), gold_idx)[1:].mean()
    return final_loss



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--actiontype", type=str, required=True)
    parser.add_argument("--max_epoch", type=int, default=-1)
    parser.add_argument("--dev_size", type=int, default=-1)
    parser.add_argument("--initial_tree_sampling", type=str, default="RB_RB_RB")
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        utils.logger.error(e, exc_info=True)

