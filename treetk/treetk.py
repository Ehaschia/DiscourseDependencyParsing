import sys

import numpy as np

################
# Conversion (sexp -> tree, or tree -> sexp)

def sexp2tree(sexp, with_nonterminal_labels, with_terminal_labels, LPAREN="(", RPAREN=")"):
    """
    :type sexp: list of str
    :type with_nonterminal_labels: bool
    :type with_terminal_labels: bool
    :type LPAREN: str
    :type RPAREN: str
    :rtype: NonTerminal
    """
    if with_nonterminal_labels and with_terminal_labels:
        from . import ll
        tree = ll.sexp2tree(sexp, LPAREN=LPAREN, RPAREN=RPAREN)
    elif with_nonterminal_labels and not with_terminal_labels:
        from . import lu
        tree = lu.sexp2tree(sexp, LPAREN=LPAREN, RPAREN=RPAREN)
    elif not with_nonterminal_labels and with_terminal_labels:
        from . import ul
        tree = ul.sexp2tree(sexp, LPAREN=LPAREN, RPAREN=RPAREN)
    elif not with_nonterminal_labels and not with_terminal_labels:
        from . import uu
        tree = uu.sexp2tree(sexp, LPAREN=LPAREN, RPAREN=RPAREN)
    else:
        print("Unsupported argument pairs: with_nonterminal_labels=%s, with_terminal_labels=%s" % \
                (with_nonterminal_labels, with_terminal_labels))
        sys.exit(-1)
    return tree

def tree2sexp(tree):
    """
    :type tree: NonTerminal or Terminal
    :rtype: list of str
    """
    # sexp = tree.__str__()
    sexp = _tree2sexp(tree)
    sexp = preprocess(sexp)
    return sexp

def _tree2sexp(node):
    """
    :type node: NonTerminal or Terminal
    :rtype: str
    """
    if node.is_terminal():
        if hasattr(node, "label"):
            return "( %s %s )" % (node.label, node.token)
        else:
            return "%s" % node.token
    else:
        inner = " ".join([_tree2sexp(c) for c in node.children])
        if hasattr(node, "label"):
            return "( %s %s )" % (node.label, inner)
        else:
            return "( %s )" % inner

def preprocess(x, LPAREN="(", RPAREN=")"):
    """
    :type x: str or list of str
    :rtype: list of str
    """
    if isinstance(x, list):
        x = " ".join(x)
    sexp = x.replace(LPAREN, " %s " % LPAREN).replace(RPAREN, " %s " % RPAREN).split()
    return sexp

def filter_parens(sexp, LPAREN="(", RPAREN=")"):
    """
    :type sexp: list of str
    :type LPAREN: str
    :type RPAREN: str
    :rtype: list of str
    """
    return [x for x in sexp if not x in [LPAREN, RPAREN]]

################
# Aggregation of nodes

def traverse(node, order="pre-order", include_terminal=True, acc=None):
    """
    :type node: NonTerminal or Terminal
    :type order: str
    :type include_terminal: bool
    :type acc: list of NonTerminal/Terminal
    :rtype list of NonTerminal/Terminal
    """
    if acc is None:
        acc = []

    if node.is_terminal():
        if include_terminal:
            acc.append(node)
        return acc

    if order == "pre-order":
        # Process the current node
        acc.append(node)
        # Process the child nodes
        for c in node.children:
            acc = traverse(c, order=order, include_terminal=include_terminal, acc=acc)
    elif order == "post-order":
        # Process the child nodes
        for c in node.children:
            acc = traverse(c, order=order, include_terminal=include_terminal, acc=acc)
        # Process the current node
        acc.append(node)
    else:
        raise ValueError("Invalid order=%s" % order)

    return acc

################
# Aggregation of production rules

def aggregate_production_rules(root, order="pre-order", include_terminal=True):
    """
    :type root: NonTerminal
    :type order: str
    :type include_terminal: bool
    :rtype: list of tuple of str
    """
    # NOTE: only for trees with nonterminal labels
    assert root.with_nonterminal_labels
    if include_terminal:
        assert root.with_terminal_labels

    nodes = traverse(root, order=order, include_terminal=include_terminal, acc=None)

    rules = []
    for node in nodes:
        if node.is_terminal():
            # Terminal node
            if node.with_terminal_labels:
                rules.append((node.label, node.token))
        else:
            # Non-Terminal node
            if node.with_terminal_labels:
                # e.g., NP -> DT NN
                rhs = [c.label for c in node.children]
            else:
                # e.g., NP -> a mouse
                rhs = [c.token if c.is_terminal() else c.label for c in node.children]
            rule = [node.label] + list(rhs)
            rules.append(tuple(rule))
    return rules

################
# Aggregation of spans

def aggregate_spans(root, include_terminal=False, order="pre-order"):
    """
    :type root: NonTerminal or Terminal
    :type order: str
    :rtype: list of (int,int,str)/(int,int)
    """
    nodes = traverse(root, order=order, include_terminal=include_terminal, acc=None)

    spans = []
    for node in nodes:
        if node.is_terminal():
            if node.with_terminal_labels:
                # e.g., (2, 2, "NN")
                spans.append(tuple(list(node.index_span) + [node.label]))
            else:
                # e.g., (2, 2)
                spans.append(node.index_span)
        else:
            if node.with_nonterminal_labels:
                # e.g., (2, 4, "NP")
                spans.append(tuple(list(node.index_span) + [node.label]))
            else:
                # e.g., (2, 4)
                spans.append(node.index_span)

    return spans

def aggregate_composition_spans(root, order="pre-order", binary=True):
    """
    :type root: NonTerminal/Terminal
    :type order: str
    :type binary: bool
    :rtype: list of [(int,int), (int,int), str]/[(int,int), (int,int)]
    """
    nodes = traverse(root, order=order, include_terminal=False, acc=None)

    # Check
    if binary:
        for node in nodes:
            assert len(node.children) == 2

    comp_spans = []
    for node in nodes:
        if node.with_nonterminal_labels:
            # e.g., [(0,1), (2,4), "NP"]
            comp_spans.append([c.index_span for c in node.children] + [node.label])
        else:
            # e.g., [(0,1), (2,4)]
            comp_spans.append([c.index_span for c in node.children])

    return comp_spans

################
# Aggregation of constituents

def aggregate_constituents(root, order="pre-order"):
    """
    :type root: NonTerminal/Terminal
    :rtype: list of list of str
    """
    nodes = traverse(root, order=order, include_terminal=False, acc=None)

    constituents = []
    for node in nodes:
        constituents.append(node.leaves())

    return constituents

################
# Tree shifting

def left_shift(node):
    """
    :type node: NonTerminal
    :rtype: NonTerminal

    e.g., (A (B C)) -> ((A B) C)
    """
    assert not node.is_terminal()
    assert len(node.children) == 2
    assert not node.children[1].is_terminal()
    right = node.children[1]
    node.children[1] = None
    tmp = right.children[0]
    right.children[0] = None
    node.children[1] = tmp
    right.children[0] = node
    return right

def right_shift(node):
    """
    :type node: NonTerminal
    :rtype: NonTerminal

    e.g., ((A B) C) -> (A (B C))
    """
    assert not node.is_terminal()
    assert len(node.children) == 2
    assert not node.children[0].is_terminal()
    left = node.children[0]
    node.children[0] = None
    tmp = left.children[1]
    left.children[1] = None
    node.children[0] = tmp
    left.children[1] = node
    return left

################
# Label assignment

def assign_labels(node, span2label, with_terminal_labels):
    """
    :type node: NonTerminal/Terminal
    :type spans: {(int, int): str}
    :type with_terminal_labels: bool
    :rtype: NonTerminal/Terminal
    """
    if node.is_terminal():
        if with_terminal_labels:
            # Terminal
            assert node.index_span in span2label
            node.label = span2label[node.index_span]
            node.with_terminal_labels = True
        else:
            pass
    else:
        # NonTerminal
        assert node.index_span in span2label
        node.label = span2label[node.index_span]
        node.with_nonterminal_labels = True
    if not node.is_terminal():
        for c_i in range(len(node.children)):
            node.children[c_i] = assign_labels(node.children[c_i], span2label, with_terminal_labels=with_terminal_labels)
    return node

################
# Checking

def is_completely_binary(node):
    """
    :type node: NonTerminal or Terminal
    :rtype: bool
    """
    if node.is_terminal():
        return True
    if len(node.children) != 2:
        return False
    acc = True
    for c in node.children:
        acc *= is_completely_binary(c)
    return bool(acc)

################
# Visualization

LEAF_WINDOW = 8
SPACE_SIZE = 1
SPACE = " " * SPACE_SIZE

EMPTY = 0
VERTICAL = 1
HORIZONTAL = 2

def pretty_print(tree, return_str=False, LPAREN="(", RPAREN=")"):
    """
    :type tree: NonTerminal or Terminal
    :type LPAREN: str
    :type RPAREN: str
    :rtype: None or str
    """
    # Tokens with padding
    tokens = tree.leaves()
    tokens_padded = [_pad_token(token) for token in tokens]
    # Create a textmap.
    textmap = _init_textmap(tokens_padded, tree)
    # Edit the textmap.
    textmap = _edit_textmap(textmap, tokens_padded, tree)
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

def _init_textmap(tokens_padded, tree):
    """
    :type tokens_padded: list of str
    :type tree: NonTerminal or Terminal
    :rtype: numpy.ndarray(shape=(R,C), dtype="O")
    """
    max_height = tree.set_height()
    max_height += 1 # include POS nodes
    textmap = np.zeros((max_height * 3,
                        sum([len(token) for token in tokens_padded]) + (len(tokens_padded)-1) * SPACE_SIZE),
                       dtype="O")
    return textmap

def _edit_textmap(textmap, tokens_padded, tree):
    """
    :type textmap: numpy.ndarray(shape=(R,C), dtype="O")
    :type tokens_padded: list of str
    :type tree: NonTerminal or Terminal
    :rtype: numpy.ndarray(shape=(R,C), dtype="O")
    """
    # Token index -> center position (i.e., column index in textmap)
    index2position = {} # {int: int}
    for token_i in range(len(tokens_padded)):
        center = int(len(tokens_padded[token_i]) / 2) \
                    + sum([len(token) for token in tokens_padded[:token_i]]) \
                    + SPACE_SIZE * token_i
        index2position[token_i] = center

    # Edit
    tree = _set_position_for_each_node(tree, index2position)
    textmap = _edit_horizontal_lines(tree, textmap)
    textmap = _edit_vertical_lines(tree, textmap)

    # Reverse and post-processing
    textmap = textmap[::-1, :]
    textmap = textmap[1:, :]
    return textmap

def _set_position_for_each_node(node, index2position):
    """
    :type node: NonTerminal or Terminal
    :type index2position: {int: int}
    :rtype: numpy.ndarray(shape=(R,C), dtype=int)
    """
    if node.is_terminal():
        position = index2position[node.index]
        node.position = position
        return node

    min_position = np.inf
    max_position = -np.inf
    for c in node.children:
        c = _set_position_for_each_node(c, index2position)
        if c.position < min_position:
            min_position = c.position
        if c.position > max_position:
            max_position = c.position

    position = (min_position + max_position) // 2
    node.position = position

    return node

def _edit_vertical_lines(node, textmap):
    """
    :type node: NonTerminal or Terminal
    :type textmap: numpy.ndarray(shape=(R,C), dtype=int)
    :rtype: numpy.ndarray(shape=(R,C), dtype=int)
    """
    row_i = node.height * 3 + 1
    col_i = node.position

    textmap[row_i-1, col_i] = VERTICAL
    textmap[row_i+1, col_i] = VERTICAL

    if node.with_nonterminal_labels and not node.is_terminal():
        label = list(node.label)
    elif node.with_terminal_labels and node.is_terminal():
        label = list(node.label)
    else:
        label = "*"

    if len(label) % 2 == 0:
        half = len(label) // 2 - 1
    else:
        half = len(label) // 2
    former_label = label[0:half]
    latter_label = label[half:]

    if (col_i - len(former_label) < 0) or (textmap.shape[1] < col_i + len(latter_label)):
        raise Exception("Node label='%s' is too long. Please use treetk.nltk_pretty_print() instead." % node.label)

    textmap[row_i, col_i-len(former_label):col_i] = former_label
    textmap[row_i, col_i:col_i+len(latter_label)] = latter_label

    if node.is_terminal():
        return textmap

    max_height = -1
    for c in node.children:
        if c.height > max_height:
            max_height = c.height
    for c in node.children:
        textmap[(c.height * 3 + 1) + 2: (max_height * 3 + 1) + 2, c.position] = VERTICAL

    for c in node.children:
        textmap = _edit_vertical_lines(c, textmap)

    return textmap

def _edit_horizontal_lines(node, textmap):
    """
    :type node: NonTerminal or Terminal
    :type textmap: numpy.ndarray(shape=(R,C), dtype="O")
    :rtype: numpy.ndarray(shape=(R,C), dtype="O")
    """
    if node.is_terminal():
        return textmap

    min_position = np.inf
    max_position = -np.inf
    for c in node.children:
        if c.position < min_position:
            min_position = c.position
        if c.position > max_position:
            max_position = c.position

    row_i = node.height * 3 + 1 - 1
    left_col_i = min_position
    right_col_i = max_position

    textmap[row_i, left_col_i:right_col_i + 1] = HORIZONTAL

    for c in node.children:
        textmap = _edit_horizontal_lines(c, textmap)

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

def nltk_pretty_print(tree, LPAREN="(", RPAREN=")"):
    """
    :type tree: NonTerminal or Terminal
    :type LPAREN: str
    :type RPAREN: str
    :rtype: None
    """
    import nltk.tree
    text = tree.__str__()
    if not tree.with_nonterminal_labels:
        text = _insert_dummy_nonterminal_labels(text,
                with_terminal_labels=tree.with_terminal_labels,
                LPAREN=LPAREN)
    nltk.tree.Tree.fromstring(text).pretty_print()

def nltk_draw(tree, LPAREN="(", RPAREN=")"):
    """
    :type tree: NonTerminal or Terminal
    :type LPAREN: str
    :type RPAREN: str
    :rtype: None
    """
    import nltk.tree
    text = tree.__str__()
    if not tree.with_nonterminal_labels:
        text = _insert_dummy_nonterminal_labels(text,
                with_terminal_labels=tree.with_terminal_labels,
                LPAREN=LPAREN)
    nltk.tree.Tree.fromstring(text).draw()

def _insert_dummy_nonterminal_labels(text, with_terminal_labels, LPAREN="("):
    """
    :type text: str
    :type with_terminal_labels: bool
    :rtype: str
    """
    if not with_terminal_labels:
        text = text.replace(LPAREN, "%s * " % LPAREN)
    else:
        sexp = preprocess(text)
        for i in range(len(sexp)-1):
            if (sexp[i] == LPAREN) and (sexp[i+1] == LPAREN):
                sexp[i] = "%s * " % LPAREN
        text = " ".join(sexp)
    return text


