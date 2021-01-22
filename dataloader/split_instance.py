from utils import DataInstance
import numpy as np

def split_instance(ins: DataInstance, inst_r_embed=None):
    pbnds = ins.pbnds

    new_spans = []
    for p_begin, p_end in pbnds:
        span = (ins.sbnds[p_begin][0], ins.sbnds[p_end][1])
        new_spans.append(span)

    new_inst_list = []
    for idx, (span_begin, span_end) in enumerate(new_spans):
        if span_begin == span_end:
            continue
        new_inst = {}
        edu_ids = ins.edu_ids[span_begin: span_end+1]
        edus = ins.edus[span_begin: span_end+1]
        edus_head = ins.edus_head[span_begin: span_end+1]
        edus_postag = ins.edus_postag[span_begin: span_end+1]
        name = ins.name + '-paragraph-' + str(idx)

        # not consider bin_sexp and nary_sexp
        new_sbnds = []
        # span_begin and end are shifted, thus shift back
        for s_begin, s_end in ins.sbnds:
            if span_begin <= s_begin <= s_end <= span_end:
                new_sbnds.append((s_begin, s_end))

        # arcs
        new_arcs = []
        for arcs in ins.arcs:
            if arcs[1] in edu_ids:
                new_arcs.append(arcs)
        # consider shift
        shift = edu_ids[0] - 1
        shifted_edu_ids = [i+1 for i in range(len(edu_ids))]
        shifted_edus = edus
        shifted_edus_postag = edus_postag
        shifted_edus_head = edus_head
        shifted_name = name
        shifted_sbnds = [(i-shift, j-shift) for i, j in new_sbnds]

        shifted_arcs = []
        for arc in new_arcs:
            if arc[0] - shift in shifted_edu_ids:
                shifted_arcs.append((arc[0] - shift, arc[1] - shift, arc[2]))
            else:
                shifted_arcs.append((0, arc[1] - shift, arc[2]))

        new_inst['edus'] = shifted_edus
        new_inst['edu_ids'] = shifted_edu_ids
        new_inst['edus_head'] = shifted_edus_head
        new_inst['edus_postag'] = shifted_edus_postag
        new_inst['name'] = shifted_name
        new_inst['sbnds'] = shifted_sbnds
        new_inst['arcs'] = shifted_arcs
        new_inst['pbnds'] = tuple([(0, len(shifted_sbnds)-1)])
        new_inst['embedding'] = inst_r_embed[span_begin: span_end+1] if inst_r_embed is not None else None
        datainstance = DataInstance(**new_inst)

        new_inst_list.append(datainstance)
    return new_inst_list
