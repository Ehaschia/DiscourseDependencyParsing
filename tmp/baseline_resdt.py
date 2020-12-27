# load baseline file and evaluate
# import os

import dataloader
# import utils
import treetk

import metrics



def draw(dataset, predict_dtree, predict_ctree):

    relation_mapper = treetk.rstdt.RelationMapper()
    i = 0
    for data in dataset:
        if data.name.find('wsj_0614') == -1:
            continue

        edu_ids = data.edu_ids
        edus = data.edus
        edus_postag = data.edus_postag
        edus_head = data.edus_head
        sbnds = data.sbnds
        pbnds = data.pbnds
        nary_sexp = data.nary_sexp
        bin_sexp = data.bin_sexp
        arcs = data.arcs

        print("Data instance #%d" % i)
        print("\t Paragraph #0")
        print("\t\t Sentence #0")
        print("\t\t\t EDU #0")
        print("\t\t\t\t EDU ID:", edu_ids[0])
        print("\t\t\t\t EDU:", edus[0])
        print("\t\t\t\t EDU (POS):", edus_postag[0])
        print("\t\t\t\t EDU (head):", edus_head[0])
        p_i = 1
        s_i = 1
        e_i = 1
        for p_begin, p_end in pbnds:
            print("\t Paragraph #%d" % p_i)
            for s_begin, s_end in sbnds[p_begin:p_end+1]:
                print("\t\t Sentence #%d" % s_i)
                for edu_id, edu, edu_postag, edu_head in zip(edu_ids[1+s_begin:1+s_end+1],
                                                             edus[1+s_begin:1+s_end+1],
                                                             edus_postag[1+s_begin:1+s_end+1],
                                                             edus_head[1+s_begin:1+s_end+1]):
                    print("\t\t\t EDU #%d" % e_i)
                    print("\t\t\t\t EDU ID:", edu_id)
                    print("\t\t\t\t EDU:", edu)
                    print("\t\t\t\t EDU (POS):", edu_postag)
                    print("\t\t\t\t EDU (head):", edu_head)
                    e_i += 1
                s_i += 1
            p_i += 1
        nary_tree = treetk.rstdt.postprocess(treetk.sexp2tree(nary_sexp, with_nonterminal_labels=True, with_terminal_labels=False))
        nary_tree = treetk.rstdt.map_relations(nary_tree, mode="c2a")
        bin_tree = treetk.rstdt.postprocess(treetk.sexp2tree(bin_sexp, with_nonterminal_labels=True, with_terminal_labels=False))
        bin_tree = treetk.rstdt.map_relations(bin_tree, mode="c2a")
        arcs = [(h,d,relation_mapper.c2a(l)) for h,d,l in arcs]
        dtree = treetk.arcs2dtree(arcs)
        treetk.pretty_print(nary_tree)
        treetk.pretty_print(bin_tree)
        treetk.pretty_print_dtree(dtree)
        # print('-'*10 + 'Sample Tree' + '-'*10)
        # predict_dtree = [(h,d,relation_mapper.c2a(l)) for h,d,l in predict_dtree]
        # predict_dtree = treetk.arcs2dtree(predict_dtree)
        # bin_predict_tree = treetk.rstdt.postprocess(
        #     treetk.sexp2tree(predict_ctree.split(), with_nonterminal_labels=True, with_terminal_labels=False))
        # bin_predict_tree = treetk.rstdt.map_relations(bin_predict_tree, mode="c2a")
        # treetk.pretty_print(bin_predict_tree)
        # treetk.pretty_print_dtree(predict_dtree)
        i += 1


def load_baseline():

    # train_dataset = dataloader.read_scidtb("train", "", relation_level="coarse-grained")
    test_dataset = dataloader.read_rstdt("train", relation_level="coarse-grained", with_root=True)
    # dev_dataset = dataloader.read_scidtb("dev", "gold", relation_level="coarse-grained")

    # processed_train = []
    # for data in train_dataset:
    #     filename = data.name
    #
    #     hyphens = utils.read_lines(os.path.join(root, "train", filename.replace(".edus.tokens", ".baseline.arcs")),
    #                                process=lambda line: line.split())
    #     assert len(hyphens) == 1
    #     hyphens = hyphens[0] # list of str
    #     arcs = treetk.hyphens2arcs(hyphens) # list of (int, int, str)
    #     processed_train.append((data, arcs))
    #
    # processed_dev = []
    # for data in dev_dataset:
    #     filename = data.name
    #
    #     hyphens = utils.read_lines(os.path.join(root, "dev/gold", filename.replace(".edus.tokens", ".baseline.arcs")),
    #                                process=lambda line: line.split())
    #     assert len(hyphens) == 1
    #     hyphens = hyphens[0]  # list of str
    #     arcs = treetk.hyphens2arcs(hyphens)  # list of (int, int, str)
    #     processed_dev.append((data, arcs))
    #
    # processed_test = []
    # for data in test_dataset:
    #     filename = data.name
    #
    #     hyphens = utils.read_lines(os.path.join(root, "test/gold", filename.replace(".edus.tokens", ".baseline.arcs")),
    #                                process=lambda line: line.split())
    #     assert len(hyphens) == 1
    #     hyphens = hyphens[0]  # list of str
    #     arcs = treetk.hyphens2arcs(hyphens)  # list of (int, int, str)
    #     processed_test.append((data, arcs))
    processed_test = []
    for data in test_dataset:
        processed_test.append(data.arcs)
    return test_dataset, processed_test


# load sampled ctree test data
def load_sample():
    test_path = "./data/rstdt-sample/RB_RB_LB.base.evaluation.ctrees"
    test = [line.strip() for line in open(test_path, 'r').readlines()]
    return test

def ctree_to_dtree(ctrees):
    def func_label_rule(node, i, j):
        relations = node.relation_label.split("/")
        if len(relations) == 1:
            return relations[0] # Left-most node is head.
        else:
            if i > j:
                return relations[j]
            else:
                return relations[j-1]
    dtrees = []
    for ctree in ctrees:
        sexp = [ctree.split()]
        assert len(sexp) == 1
        sexp = sexp[0]

        # Constituency
        ctree = treetk.rstdt.postprocess(treetk.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=False))

        # Dependency
        # Assign heads
        ctree = treetk.rstdt.assign_heads(ctree)
        # Conversion
        dtree = treetk.ctree2dtree(ctree, func_label_rule=func_label_rule)
        arcs = dtree.tolist(labeled=True)
        dtrees.append(arcs)
    return dtrees, ctrees


import treesamplers
def depformat():
    test_dataset = dataloader.read_scidtb("test", "gold", relation_level="coarse-grained")
    sampler = treesamplers.TreeSampler("RB_TD_TD".split("_"))
    processed_test = []
    for data in test_dataset:
        edu_ids = data.edu_ids
        edus = data.edus
        edus_postag = data.edus_postag
        edus_head = data.edus_head
        sbnds = data.sbnds
        pbnds = data.pbnds
        gold_arcs = data.arcs

        # CHECK: Single ROOT?
        # assert sum([1 for h, d, l in gold_arcs if h == 0]) == 1
        arcs = sampler.sample(edu_ids, edus, edus_head, sbnds, pbnds)
        processed_test.append((data, arcs))
    return processed_test


def eval(gold, sample):
    # save to tmp to call
    i = 10
    print(gold[i])
    print(sample[i])

    res = metrics.attachment_scores_v2([gold[i]], [sample[i]])
    print(res)


def main():
    # train, dev, test = load_baseline()
    predict = load_sample()
    raw_gold, processed_gold = load_baseline()
    # predict_dtree, predict_ctree = ctree_to_dtree(predict)
    # eval(processed_gold, predict_dtree)
    # draw(raw_gold, predict_dtree[10], predict_ctree[10])
    draw(raw_gold, None, None)

if __name__ == '__main__':
    main()