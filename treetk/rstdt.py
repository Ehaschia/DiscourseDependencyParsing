import os
import re

from .ll import NonTerminal, Terminal

LPAREN = "("
RPAREN = ")"
PARENS = [LPAREN, RPAREN]

###########################
# Text -> S-expression

def read_sexp(path):
    """
    :type path: str
    :rtype: list of str
    """
    sexp = []
    for line in open(path):
        line = line.strip()
        tokens = line.replace("(", " ( ")\
                     .replace(")", " ) ")\
                     .replace("_!", " _! ")\
                     .replace("//TT_ERR", "")\
                     .split()
        if len(tokens) == 0:
            continue
        sexp.extend(tokens)
    return sexp

###########################
# S-expression -> Tree

def make_terminal(edu, edu_i, relation, nuclearity):
    """
    :type edu: str
    :type edu_i: int
    :type relation: str
    :type nuclearity: str
    :rtype: Terminal
    """
    # node = Terminal(token=edu, index=edu_i, label="<%s,%s>" % (relation, nuclearity))
    node = Terminal(token=edu, index=edu_i, label="*")
    node.relation = relation # Temporal
    node.nuclearity = nuclearity # Temporal
    return node

def make_nonterminal(index_span, relation, nuclearity):
    """
    :type index_span: (int, int)
    :type relation: str
    :type nuclearity: str
    :rtype: NonTerminal
    """
    # node = NonTerminal(label="<%s,%s>" % (relation, nuclearity))
    node = NonTerminal(label="*")
    node.index_span = index_span
    node.relation = relation # Temporal
    node.nuclearity = nuclearity # Temporal
    node.nuclearities_of_children = [] # Temporal: used for labeling nodes
    node.relations_of_children = [] # Temporal: used for labeling nodes
    return node

def sexp2tree(sexp):
    """
    :type sexp: list of str
    :rtype: NonTerminal

    Please note that the input ``sexp'' is assumed to be the raw version in RST-DT.
    So, if you want to convert a loaded S-expression after preprocessing, please use ``treetk.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=True)'' instead.
    """
    tmp_node = make_nonterminal(index_span=(-1,-1), relation="TMP", nuclearity="tmp")
    stack = [tmp_node]

    tokens = sexp + ["<<<FIN>>>"]
    n_tokens = len(tokens)

    i = 0
    # LPAREN
    assert tokens[i] == LPAREN
    i += 1
    # Root
    assert tokens[i] == "Root"
    i += 1
    # LPAREN
    assert tokens[i] == LPAREN
    i += 1
    # span
    assert tokens[i] == "span"
    i += 1
    # span IDs
    span_left = int(tokens[i])
    i += 1
    span_right = int(tokens[i])
    i += 1
    # RPAREN
    assert tokens[i] == RPAREN
    # Create the root node
    root_node = make_nonterminal(index_span=(span_left-1, span_right-1), relation="Root", nuclearity="Root")
    stack.append(root_node)
    i += 1

    while i < n_tokens:
        if tokens[i] == LPAREN:
            i += 1
            # Nucleus or Satellite
            assert tokens[i] in ["Nucleus", "Satellite"]
            nuclearity = "N" if tokens[i] == "Nucleus" else "S"
            i += 1
            # LPAREN
            assert tokens[i] == LPAREN
            i += 1
            # Non-terminal or Terminal
            assert tokens[i] in ["leaf", "span"]
            is_terminal = True if tokens[i] == "leaf" else False
            i += 1
            # EDU index or span
            if is_terminal:
                edu_i = int(tokens[i]) - 1 # NOTE: shifted by -1
                i += 1
            else:
                span_left = int(tokens[i])
                i += 1
                span_right = int(tokens[i])
                i += 1
                index_span = (span_left-1, span_right-1) # NOTE: shifted by -1
            # RPAREN
            assert tokens[i] == RPAREN
            i += 1
            # LPAREN
            assert tokens[i] == LPAREN
            i += 1
            # Coherence relation
            assert tokens[i] == "rel2par"
            i += 1
            relation = tokens[i]
            i += 1
            # RPAREN
            assert tokens[i] == RPAREN
            i += 1
            # If this is terminal, create a node, and then add it as a child of the top node on the stack.
            # If this is non-terminal, create a temporal node, and then push to the stack.
            if is_terminal:
                # LPAREN
                assert tokens[i] == LPAREN
                i += 1
                # "text"
                assert tokens[i] == "text"
                i += 1
                # "_!"
                assert tokens[i] == "_!"
                i += 1
                # EDU
                edu = []
                while tokens[i] != "_!":
                    edu.append(tokens[i])
                    i += 1
                edu = " ".join(edu)
                # "_!"
                assert tokens[i] == "_!"
                i += 1
                # RPAREN
                assert tokens[i] == RPAREN
                i += 1
                # RPAREN
                assert tokens[i] == RPAREN
                i += 1
                # Create a node
                node = make_terminal(edu=edu, edu_i=edu_i, relation=relation, nuclearity=nuclearity)
                stack[-1].add_child(node)
            else:
                # Create a node
                node = make_nonterminal(index_span=index_span, relation=relation, nuclearity=nuclearity)
                stack.append(node)
        elif tokens[i] == RPAREN:
            node = stack.pop()
            stack[-1].add_child(node)
            i += 1
        else:
            # Fin.
            assert tokens[i] == "<<<FIN>>>"
            break
    assert len(stack) == 1
    tmp_node = stack.pop()
    assert len(tmp_node.children) == 1
    return tmp_node.children[0]

def shift_labels(node):
    """
    :type node: NonTerminal/Terminal
    :rtype: NonTerminal/Terminal
    """
    if node.is_terminal():
        return node

    # Set properties (i.e., ``nuclearities_of_children'', ``relations_of_children'') for labeling the current node
    node.nuclearities_of_children = [c.nuclearity for c in node.children]
    node.relations_of_children = [c.relation for c in node.children if c.relation != "span"]
    if len(set(node.nuclearities_of_children)) == 1 and node.nuclearities_of_children[0] == "N":
        assert len(set(node.relations_of_children)) == 1
        node.relations_of_children = [node.relations_of_children[0]]

    # Labeling this node
    new_label_rel = "/".join(node.relations_of_children)
    new_label_nuc = "/".join(node.nuclearities_of_children)
    new_label = "<%s,%s>" % (new_label_rel, new_label_nuc)
    node.label = new_label

    # Recursive (pre-order)
    for c_i in range(len(node.children)):
        node.children[c_i] = shift_labels(node.children[c_i])

    return node

def tree2str(node, labeled=True):
    """
    :type node: NonTerminal/Terminal
    :type labeled: bool
    :rtype: str
    """
    if node.is_terminal():
        return "%s" % node.index

    inner = " ".join([tree2str(c, labeled=labeled) for c in node.children])
    if labeled:
        # label_rel = "/".join(node.relations_of_children)
        # label_nuc = "/".join(node.nuclearities_of_children)
        # label = "<%s,%s>" % (label_rel, label_nuc)
        return "( %s %s )" % (node.label, inner)
    else:
        return "( %s )" % inner

###########################
# Preprocessing (optional)

def binarize(node):
    """
    :type node: NonTerminal/Terminal
    :rtype: NonTerminal/Terminal
    """
    if node.is_terminal():
        return node

    # Right branching
    if len(node.children) > 2:
        node.children = _right_branching(node.children)

    # Recursive
    for c_i in range(len(node.children)):
        node.children[c_i] = binarize(node.children[c_i])

    return node

def _right_branching(nodes):
    """
    :type nodes: list of NonTerminal/Terminal
    :rtype: [NonTerminal/Terminal, NonTerminal/Terminal]
    """
    if len(nodes) == 2:
        return nodes

    lhs = nodes[0] # The left-most child node is head
    index_span = (nodes[1].index_span[0], nodes[-1].index_span[1])
    relation = nodes[1].relation
    nuclearity = nodes[1].nuclearity
    rhs = make_nonterminal(index_span=index_span, relation=relation, nuclearity=nuclearity)
    rhs.children = _right_branching(nodes[1:])
    return [lhs, rhs]

###########################
# Postprocessing (necessary)

def postprocess(root):
    """
    :type root: NonTerminal/Terminal
    :rtype: NonTerminal/Terminal
    """
    if root.is_terminal():
        return root

    root = _assign_relation_and_nuclearity_labels(root)
    return root

def _assign_relation_and_nuclearity_labels(node):
    """
    :type node: NonTerminal/Terminal
    :rtype: NonTerminal/Terminal
    """
    if node.is_terminal():
        return node

    # Extract relation and nuclearity labels from the node label
    relation_label, nuclearity_label = extract_relation_and_nuclearity_labels(node.label) # str, str
    node.relation_label = relation_label
    node.nuclearity_label = nuclearity_label

    # Recursive
    for c_i in range(len(node.children)):
        node.children[c_i] = _assign_relation_and_nuclearity_labels(node.children[c_i])

    return node

def extract_relation_and_nuclearity_labels(label):
    """
    :type label: str
    :rtype: str, str
    """
    re_comp = re.compile("<(.+),(.+)>")
    match = re_comp.findall(label)
    assert len(match) == 1
    relation_label = match[0][0]
    nuclearity_label = match[0][1]
    return relation_label, nuclearity_label

def assign_heads(root):
    """
    :type root: NonTerminal/Terminal
    :rtype: NonTerminal/Terminal
    """
    if root.is_terminal():
        return root
    root.calc_heads(func_head_child_rule=lambda node: node.nuclearity_label.split("/").index("N"))
    return root

###########################
# Renaming of relation labels

class RelationMapper(object):
    """
    A class for mapping between fine-grained relations and coarse-grained classes.
    Mapping is defined in ./rstdt_relation_mapping.txt
    We also make mapping from coarse-grained relations to the corresponding abbreviations (defined in ./rstdt_relation_abbreviations.txt).
    """

    def __init__(self, corpus_name="rstdt"):

        if corpus_name == "rstdt":
            mapping_file = "rstdt_relation_mapping.txt"
            abb_file = "rstdt_relation_abbreviations.txt"
        elif corpus_name == "scidtb":
            mapping_file = "scidtb_relation_mapping.txt"
            abb_file = "scidtb_relation_abbreviations.txt"
        else:
            raise ValueError("Invalid corpus_name=%s" % corpus_name)

        self.fine2coarse = {} # {str: str}
        self.coarse2fine = {} # {str: list of str}
        for line in open(os.path.join(os.path.dirname(__file__), mapping_file)):
            items = line.strip().split()
            assert len(items) > 1
            crel = items[0]
            frels = items[1:]
            self.coarse2fine[crel] = frels
            for frel in frels:
                assert not frel in self.fine2coarse
                self.fine2coarse[frel] = crel

        self.coarse2abb = {} # {str: str}
        self.abb2coarse = {} # {str: str}
        for line in open(os.path.join(os.path.dirname(__file__), abb_file)):
            items = line.strip().split()
            assert len(items) > 1
            crel = items[0]
            abb = items[1]
            self.coarse2abb[crel] = abb
            self.abb2coarse[abb] = crel

        if corpus_name == "rstdt":
            self.coarse2yung = {} # {str: str}
            self.yung2coarse = {} # {str: list of str}
            for line in open(os.path.join(os.path.dirname(__file__), "rstdt_relation_mapping_yung17.txt")):
                items = line.strip().split()
                assert len(items) > 1
                yrel = items[0]
                crels = items[1:]
                self.yung2coarse[yrel] = crels
                for crel in crels:
                    assert not crel in self.coarse2yung
                    self.coarse2yung[crel] = yrel

            self.yung2abb = {} # {str: str}
            self.abb2yung = {} # {str: str}
            for line in open(os.path.join(os.path.dirname(__file__), "rstdt_relation_abbreviations_yung17.txt")):
                items = line.strip().split()
                assert len(items) > 1
                yrel = items[0]
                abb = items[1]
                self.yung2abb[yrel] = abb
                self.abb2yung[abb] = yrel

    # def c2f(self, crel):
    #     """
    #     :type crel: str
    #     :rtype: list of str
    #     """
    #     return self.coarse2fine[crel]

    ###

    def f2c(self, frel):
        """
        :type frel: str
        :rtype: str
        """
        return self.fine2coarse[frel]

    def c2a(self, crel):
        """
        :type crel: str
        :rtype: str
        """
        return self.coarse2abb[crel]

    def a2c(self, abb):
        """
        :type abb: str
        :rtype: str
        """
        return self.abb2coarse[abb]

    ###

    def c2y(self, crel):
        """
        :type crel: str
        :rtype: str
        """
        return self.coarse2yung[crel]

    def y2a(self, yrel):
        """
        :type yrel: str
        :rtype: str
        """
        return self.yung2abb[yrel]

    def a2y(self, abb):
        """
        :type abb: str
        :rtype: str
        """
        return self.abb2yung[abb]

    ###

    def get_relation_lists(self):
        """
        :rtype: list of str, list of str
        """
        crels = list(self.coarse2fine.keys())
        frels = list(self.fine2coarse.keys())
        return crels, frels

def map_relations(root, mode):
    """
    :type root: NonTerminal/Terminal
    :type mode: str
    :rtype: NonTermina/Terminal
    """
    relation_mapper = RelationMapper()
    map_func = None
    if mode == "f2c":
        map_func = relation_mapper.f2c
    elif mode == "c2a":
        map_func = relation_mapper.c2a
    elif mode == "a2c":
        map_func = relation_mapper.a2c
    elif mode == "c2y":
        map_func = relation_mapper.c2y
    elif mode == "y2a":
        map_func = relation_mapper.y2a
    elif mode == "a2y":
        map_func = relation_mapper.a2y
    else:
        raise ValueError("Invalid mode=%s" % mode)
    root = _map_relations(root, map_func=map_func)
    return root

def _map_relations(node, map_func):
    """
    :type node: NonTerminal/Terminal
    :type map_func: function
    :rtype: NonTerminal/Terminal
    """
    if node.is_terminal():
        return node

    # Map relations of this node
    relations = node.relation_label.split("/")
    relations = [map_func(r) for r in relations]
    relation_label = "/".join(relations)
    node.relation_label = relation_label
    node.label = "<%s,%s>" % (relation_label, node.nuclearity_label)

    # Recursion
    for c_i in range(len(node.children)):
        node.children[c_i] = _map_relations(node.children[c_i], map_func=map_func)

    return node

