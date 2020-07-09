from collections import defaultdict

import numpy as np

from . import treetk

class DependencyTree(object):

    def __init__(self, arcs, tokens):
        """
        :type arcs: list of (int, int, str)
        :type tokens: list of str
        """
        # NOTE that integers in arcs are not word IDs but indices in the sentence.
        self.arcs = arcs
        self.tokens = tokens

        self.head2dependents = defaultdict(list) # {int: list of (int, str)}
        self.dependent2head = {} # {int: (int/None, str/None)}

        # Create a mapping of head -> dependents
        for head, dependent, label in self.arcs:
            self.head2dependents[head].append((dependent, label))

        # Create a mapping of dependent -> head
        for dependent in range(len(self.tokens)):
            self.dependent2head[dependent] = (None, None)
        for head, dependent, label in self.arcs:
            # Tokens should not have multiple heads.
            if self.dependent2head[dependent] != (None, None):
                raise ValueError("The %d-th token has multiple heads! Arcs=%s" % (dependent, self.arcs))
            self.dependent2head[dependent] = (head, label)
        assert self.dependent2head[0] == (None, None) # The 0-th token is a root symbol.

    def __str__(self):
        """
        :rtype: str
        """
        arcs = self.tolist()
        arcs = ["%d-%d-%s" % (h,d,l) for h,d,l in arcs]
        return " ".join(arcs)

    def tolist(self, labeled=True, replace_with_tokens=False):
        """
        :type labeled: bool
        :type replace_with_tokens: bool
        :rtype: list of (T, T, str), or list of (T, T) where T \in {int, str}
        """
        result = self.arcs
        if replace_with_tokens:
            result = [(self.tokens[h], self.tokens[d], l) for h,d,l in result]
        if not labeled:
            result = [(h,d) for h,d,l in result]
        return result

    def get_dependents(self, index):
        """
        :type index: int
        :rtype: list of (int, str)
        """
        return self.head2dependents.get(index, [])

    def get_head(self, index):
        """
        :rtype index: int
        :rtype: (int, str)
        """
        return self.dependent2head[index]

def arcs2dtree(arcs, tokens=None):
    """
    :type arcs: list of (int, int, str), or list of (int, int)
    :type tokens: list of str, or None
    :rtype DependencyTree
    """
    arcs = sort_arcs(arcs)
    arcs_checked = [x if len(x) == 3 else (x[0],x[1],"*") for x in arcs]
    if tokens is None:
        # tokens = ["<root>"] + ["x%s" % (tok_i+1) for tok_i in range(len(arcs_checked))]
        tokens = [str(i) for i in range(len(arcs_checked)+1)]
    dtree = DependencyTree(arcs=arcs_checked, tokens=tokens)
    return dtree

def hyphens2arcs(hyphens):
    """
    :type hyphens: list of str
    :rtype: list of (int, int, str)
    """
    arcs = [x.split("-") for x in hyphens]
    arcs = [(int(arc[0]), int(arc[1]), str("-".join(arc[2:]))) if len(arc) >= 3
             else (int(arc[0]), int(arc[1]), "*")
             for arc in arcs]
    return arcs

def sort_arcs(arcs):
    """
    :type arcs: list of (int, int)/(int, int, str)
    :rtype: list of (int, int)/(int, int, str)
    """
    return sorted(arcs, key=lambda x: x[1])

#####################################
# Aggregation of arcs

def traverse_dtree(dtree, head_i, order="pre-order", acc=None):
    """
    :type dtree: DependencyTree
    :type head_i: int
    :type order: str
    :type acc: list of int
    :rtype: list of int
    """
    if acc is None:
        acc = []

    if len(dtree.get_dependents(head_i)) == 0:
        acc.append(head_i)
        return acc

    if order == "pre-order":
        # Process the current head
        acc.append(head_i)
        # Process the dependents
        for dep_i, _ in dtree.get_dependents(head_i):
            acc = traverse_dtree(dtree, dep_i, order=order, acc=acc)
    elif order == "post-order":
        # Process the dependents
        for dep_i, _ in dtree.get_dependents(head_i):
            acc = traverse_dtree(dtree, dep_i, order=order, acc=acc)
        # Process the current head
        acc.append(head_i)
    else:
        raise ValueError("Invalid order=%s" % order)

    return acc

#####################################

LEAF_WINDOW = 8
SPACE_SIZE = 1
SPACE = " " * SPACE_SIZE

EMPTY = 0
ARROW = 1
VERTICAL = 2
HORIZONTAL = 3
# LABEL_BEGIN = 4
# LABEL_END = 5

def pretty_print_dtree(dtree, return_str=False):
    """
    :type dtree: DependencyTree
    :type return_str: bool
    :rtype: None or str
    """
    arcs_labeled = dtree.tolist(labeled=True)
    arcs_unlabeled = {(b,e) for b,e,_ in arcs_labeled}
    arc2label = {(b,e): l for b,e,l in arcs_labeled}

    # Tokens with padding
    tokens = dtree.tokens
    tokens_padded = [_pad_token(token) for token in tokens]
    # Compute heights of the arcs.
    arc2height = _get_arc2height(arcs_unlabeled)
    # Create a textmap.
    textmap = _init_textmap(tokens_padded, arc2height)
    # Edit the textmap.
    textmap = _edit_textmap(textmap, tokens_padded, arc2height, arc2label)
    # Create a text based on the textmap.
    text = _generate_text(textmap, tokens_padded)
    if return_str:
        return text
    else:
        print(text)

def _pad_token(token):
    """
    :type token: str
    :rtype: str
    """
    token = " " + token + " "
    while len(token) <= LEAF_WINDOW:
        token = " " + token + " "
    token = "[" + token[1:-1] + "]"
    return token

def _get_arc2height(arcs):
    """
    :type arcs: list of (int, int)
    :rtype: dictionary of {(int, int): int}
    """
    # arc2height = {(b,e): np.abs(b - e) for b, e in arcs}

    n_arcs = len(arcs)
    arcs_sorted = sorted(arcs, key=lambda x: np.abs(x[0] - x[1]))
    arc2height = {arc: 1 for arc in arcs}
    for arc_i in range(n_arcs):
        bi, ei = sorted(arcs_sorted[arc_i])
        for arc_j in range(n_arcs):
            if arc_i == arc_j:
                continue
            bj, ej = sorted(arcs_sorted[arc_j])
            if bi <= bj <= ej <= ei:
                arc2height[arcs_sorted[arc_i]] = max(arc2height[arcs_sorted[arc_j]] + 1, arc2height[arcs_sorted[arc_i]])
    return arc2height

def _init_textmap(tokens_padded, arc2height):
    """
    :type tokens_padded: list of str
    :type arc2height: dictionary of {(int, int): int}
    :rtype: numpy.ndarray(shape=(R,C), dtype="O")
    """
    max_height = -1
    for arc in arc2height.keys():
        height = arc2height[arc]
        if height > max_height:
            max_height = height
    textmap = np.zeros((1 + max_height * 2,
                        sum([len(token) for token in tokens_padded]) + (len(tokens_padded)-1) * SPACE_SIZE),
                       dtype="O")
    return textmap

def _edit_textmap(textmap, tokens_padded, arc2height, arc2label):
    """
    :type textmap: numpy.ndarray(shape=(R,C), dtype="O")
    :type tokens_padded: list of str
    :type arc2height: dictionary of {(int, int): int}
    :type arc2label: dictionary of {(int, int): str}
    :rtype: numpy.ndarray(shape=(R,C), dtype="O")
    """
    # Token index -> center position (i.e., column index in textmap)
    index2position = {} # {int: int}
    for token_i in range(len(tokens_padded)):
        center = int(len(tokens_padded[token_i]) / 2) \
                    + sum([len(token) for token in tokens_padded[:token_i]]) \
                    + SPACE_SIZE * token_i
        index2position[token_i] = center

    arcs_sorted = sorted(arc2height.keys(), key=lambda x: arc2height[x])
    for arc in arcs_sorted:
        b, e = arc
        b_pos = index2position[b]
        e_pos = index2position[e]
        height = arc2height[arc]
        label = arc2label[arc]
        # End point
        textmap[-1, e_pos] = ARROW
        textmap[-2:-1-height*2:-1, e_pos] = VERTICAL
        # Beginning point
        if b < e:
            textmap[-1, b_pos+2] = VERTICAL
        else:
            textmap[-1, b_pos-2] = VERTICAL
        # Horizontal lines
        if b < e:
            textmap[-1-height*2, b_pos+2:e_pos+1] = HORIZONTAL
        else:
            textmap[-1-height*2, e_pos:b_pos-2+1] = HORIZONTAL
        # Vertical lines
        if b < e:
            textmap[-2:-1-height*2:-1, b_pos+2] = VERTICAL
        else:
            textmap[-2:-1-height*2:-1, b_pos-2] = VERTICAL
        # Label
        if b < e:
            textmap[-1-height*2+1, e_pos-len(label):e_pos] = list(label)
        else:
            textmap[-1-height*2+1, e_pos+1:e_pos+1+len(label)] = list(label)

    return textmap

def _generate_text(textmap, tokens_padded):
    """
    :type textmap: numpy.ndarray(shape=(R,C), dtype="O")
    :type tokens_padded: list of str
    """
    text = ""
    for row_i in range(textmap.shape[0]):
        row_text = ""
        for col_i in range(textmap.shape[1]):
            if textmap[row_i, col_i] == EMPTY:
                row_text = row_text + " "
            elif textmap[row_i, col_i] == ARROW:
                row_text = row_text + "V"
            elif textmap[row_i, col_i] == VERTICAL:
                row_text = row_text + "|"
            elif textmap[row_i, col_i] == HORIZONTAL:
                row_text = row_text + "_"
            else:
                row_text = row_text + str(textmap[row_i, col_i])
        row_text = row_text.rstrip() + "\n"
        text = text + row_text
    for token in tokens_padded:
        text = text + token
        text = text + SPACE
    text = text[:-SPACE_SIZE]
    return text

#####################################

def ctree2dtree(tree, func_label_rule=None):
    """
    :type NonTerminal or Terminal
    :type func_label_rule: function of (NonTerminal, int, int) -> str
    :rtype: DependencyTree
    """
    if (tree.head_token_index is None) or (tree.head_child_index is None):
        raise ValueError("Please call ``tree.calc_heads(func_head_child_rule)'' before conversion.")
    assert not tree.is_terminal()

    nodes = treetk.traverse(tree, order="post-order", include_terminal=False)

    arcs = []
    for node in nodes:
        head_token_index = node.head_token_index
        for c_i in range(len(node.children)):
            dep_token_index = node.children[c_i].head_token_index
            if head_token_index == dep_token_index:
                continue
            if func_label_rule is None:
                label = "*"
            else:
                label = func_label_rule(node, node.head_child_index, c_i)
            arcs.append((head_token_index, dep_token_index, label))
    tokens = tree.leaves()

    # Add a ROOT symbol to the dependency tree
    arcs = [(h+1, d+1, l) for h,d,l in arcs]
    arcs.append((0, tree.head_token_index+1, "<root>"))
    tokens = ["<root>"] + tokens

    dtree = arcs2dtree(arcs=arcs, tokens=tokens)
    return dtree

#####################################

def dtree2ctree(dtree):
    """
    :type dtree: DependencyTree
    :rtype: NonTerminal
    """
    from . import lu

    # We exclude the root symbol in the resulting constituent tree
    root_deps = dtree.get_dependents(0)
    assert len(root_deps) == 1
    root_dep_index, _  = root_deps[0]

    token_indices = traverse_dtree(dtree, head_i=root_dep_index, order="post-order", acc=None)

    # Create terminal nodes in post-order
    memo_nodes = defaultdict(list)
    for token_index in token_indices:
        terminal = lu.Terminal(token=dtree.tokens[token_index], index=token_index-1) # NOTE: index is shifted by -1
        memo_nodes[token_index].append(terminal)

    # Create non-terminal nodes
    for token_index in token_indices:
        if len(dtree.get_dependents(token_index)) == 0:
            continue
        # Required properties
        label_list = []
        children = []
        index_span = (None, None)
        head_token_index = token_index - 1 # NOTE: index is shifted by -1
        head_child_index = None
        # Left-side child nodes
        for dep_index, label in dtree.get_dependents(token_index):
            if dep_index >= token_index:
                continue
            child_node = memo_nodes[dep_index][-1]
            label_list.append(label)
            children.append(child_node)
        # Center/head child node
        child_node = memo_nodes[token_index][-1]
        children.append(child_node)
        head_child_index = len(children) - 1
        # Right-side child nodes
        for dep_index, label in dtree.get_dependents(token_index):
            if dep_index <= token_index:
                continue
            child_node = memo_nodes[dep_index][-1]
            label_list.append(label)
            children.append(child_node)
        label = "/".join(label_list)
        min_index = min([c.index_span[0] for c in children])
        max_index = max([c.index_span[1] for c in children])
        index_span = (min_index, max_index) # NOTE: indices are automatically shifted by -1
        # Create a non-terminal node
        nonterminal = lu.NonTerminal(label=label)
        nonterminal.children = children
        nonterminal.index_span = index_span
        nonterminal.head_token_index = head_token_index
        nonterminal.head_child_index = head_child_index
        memo_nodes[token_index].append(nonterminal)

    return memo_nodes[root_dep_index][-1]


