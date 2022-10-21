import inspect
from copy import deepcopy
from typing import List, Dict

import numpy as np
import re

from genepro.node import Node
from genepro import node_impl
from genepro.node_impl import Feature, Constant


def compute_linear_scaling(y, p):
    """
    Computes the optimal slope and intercept that realize the affine transformation that minimizes the mean-squared-error between the label and the prediction.
    See the paper: https://doi:10.1023/B:GENP.0000030195.77571.f9

    Parameters
    ----------
    y : np.array
      the label values
    p : np.array
      the respective predictions

    Returns
    -------
    float, float
      slope and intercept that represent
    """
    slope = np.cov(y, p)[0, 1] / (np.var(p) + 1e-12)
    intercept = np.mean(y) - slope * np.mean(p)
    return slope, intercept


def tree_from_prefix_repr(prefix_repr: str) -> Node:
    """
    Creates a tree from a string representation in prefix format (that is, pre-order tree traversal);
    the symbol in the string representation need to match those in the Node implementations (in `genepro.node_impl.py`)

    Parameters
    ----------
    prefix_repr : str
      the string representation of the tree as a list of nodes parsed with pre-order traversal (obtainable with `str(tree.get_subtree())`)

    Returns
    -------
    Node
      the tree that corresponds to the string representation
    """
    symb_list = prefix_repr.replace("[", "").replace("]", "").replace(", ", ",").split(",")
    # generate the tree
    node_classes = [c[1] for c in inspect.getmembers(node_impl, inspect.isclass)]
    possible_nodes = list()
    for node_cls in node_classes:
        # handle Features and Constants separetely (also, avoid base class Node)
        if node_cls == Node or node_cls == Feature or node_cls == Constant:
            continue
        node_obj = node_cls()
        possible_nodes.append(node_obj)
    tree, _ = __tree_from_symb_list_recursive(symb_list, possible_nodes)
    return tree


def __tree_from_symb_list_recursive(symb_list: list, possible_nodes: list):
    """
    Helper recursive function for `tree_from_prefix_repr`

    Parameters
    ----------
    symb_list : list
      list of str that are symbols (as per the attribute `symb` of Node)

    possible_nodes : list
      list of all possible Node objects from `genepro.node_impl`

    Returns
    -------
      Node, list
        the last-generated node and the updated symb_list, required for the recursive construction of the tree
    """
    symb = symb_list[0]
    symb_list = symb_list[1:]
    # check if it is a feature
    if symb.startswith("x_"):
        id = int(symb[2:])
        n = Feature(id)
        return n, symb_list

    # check if it is a function
    for pn in possible_nodes:
        if symb == str(pn):
            n = deepcopy(pn)
            for _ in range(n.arity):
                c, symb_list = __tree_from_symb_list_recursive(symb_list, possible_nodes)
                n.insert_child(c)
            return n, symb_list

    # if reached this line, it must be a constant
    n = Constant(float(symb))
    return n, symb_list


def counts_encode_tree(tree: Node, operators: list, n_features: int, additional_properties: bool = True) -> list:
    """
    Provides a counts encoded representation of a tree, traversing it from the root in a breadth-first manner,
    and considering it as a full tree. It outputs a list of integers.
    Each integer is the raw count of a given operator, feature or constant in the tree.
    The output list is fixed-length, i.e., if a given operator/feature does not belong to the tree,
    then its raw count is 0 in the output list.
    Optionally, it is possible, through the "additional_properties" flag, to append to the raw counts
    of operators/features/constants a list of numerical properties of the tree.
    These numerical properties are listed below:
        - (height + 1) / number of nodes
        - max arity / max breadth
        - number of leaf nodes / number of nodes

    Parameters
    ----------
    tree : Node
      tree to be counts encoded

    operators : list
      list of all possible symbols allowed for operators

    n_features : int
      amount of allowed features in the tree

    additional_properties: bool
      if this flag is True, then the output list will be extended with a list of numerical properties of the given tree.
      These properties are computed using the dictionary returned
      by the genepro.node.tree_numerical_properties method.
      Default value is False.

    Returns
    -------
      list
        the counts encoded tree
    """
    size = len(operators) + n_features + 1
    if additional_properties:
        counts = [0.0] * (size + 3)
    else:
        counts = [0.0] * size

    properties_dict = {k: 0.0 for k in
                       ["height", "n_nodes", "max_arity", "max_breadth", "n_leaf_nodes", "n_internal_nodes"]}
    levels = {}

    __counts_encode_tree_recursive(tree, 0, size, operators, n_features, properties_dict,
                                   levels, counts, additional_properties)
    if additional_properties:
        for l in levels:
            if levels[l] > properties_dict["max_breadth"]:
                properties_dict["max_breadth"] = levels[l]
        height_n_nodes_ratio = (properties_dict["height"] + 1.0) / float(properties_dict["n_nodes"])
        max_arity_breadth_ratio = properties_dict["max_arity"] / float(properties_dict["max_breadth"])
        leaf_nodes_ratio = properties_dict["n_leaf_nodes"] / float(properties_dict["n_nodes"])
        counts[size] = height_n_nodes_ratio
        counts[size + 1] = max_arity_breadth_ratio
        counts[size + 2] = leaf_nodes_ratio

    return counts


def __counts_encode_tree_recursive(tree: Node, depth: int, size: int, operators: List, n_features: int, properties_dict: Dict, levels: Dict, counts: List, additional_properties: bool = True):
    if additional_properties:
        properties_dict["n_nodes"] += 1.0
        curr_depth, curr_arity = depth, tree.arity
        if curr_depth > properties_dict["height"]:
            properties_dict["height"] = curr_depth
        if curr_arity > properties_dict["max_arity"]:
            properties_dict["max_arity"] = curr_arity
        if curr_arity == 0:
            properties_dict["n_leaf_nodes"] += 1.0
        else:
            properties_dict["n_internal_nodes"] += 1.0
        if curr_depth not in levels:
            levels[curr_depth] = 0.0
        levels[curr_depth] += 1.0
    node_content = tree.symb
    if node_content in operators:
        counts[operators.index(node_content)] += 1.0
    elif node_content.startswith("x_"):
        feature_index = int(node_content[2:])
        if feature_index < n_features:
            counts[len(operators) + feature_index] += 1.0
        else:
            raise Exception(
                f"More features than declared. Declared: {n_features}. Feature index found: {feature_index}.")
    elif re.search(r'^[+-]?\d+(\.\d+)?([Ee][+-]?\d+)?$', node_content):
        counts[size - 1] += 1.0
    else:
        raise Exception(f"Unexpected node content: {str(node_content)}.")
    for child_index in range(tree.arity):
        __counts_encode_tree_recursive(tree.get_child(child_index), depth + 1, size, operators, n_features, properties_dict, levels, counts, additional_properties)


def counts_level_wise_encode_tree(tree: Node, operators: list, n_features: int, max_depth: int,
                                  additional_properties: bool = True) -> list:
    """
    Provides a level-wise counts encoded representation of a tree, traversing it from the root in a breadth-first manner,
    and considering it as a full tree. It outputs a list of integers.
    Each integer is the raw count of a given operator, feature or constant in a given layer of the tree.
    The output list is fixed-length, i.e., if a given operator/feature does not belong to a given layer of the tree,
    then its raw count is 0 in the output list.
    Optionally, it is possible, through the "additional_properties" flag, to append to the raw counts
    of operators/features/constants a list of numerical properties of the tree.
    These numerical properties are listed below:
        - (height + 1) / number of nodes
        - max arity / max breadth
        - number of leaf nodes / number of nodes

    Parameters
    ----------
    tree : Node
      tree to be counts encoded

    operators : list
      list of all possible symbols allowed for operators

    n_features : int
      amount of allowed features in the tree

    max_depth : int
      maximal depth of the tree

    additional_properties: bool
      if this flag is True, then the output list will be extended with a list of numerical properties of the given tree.
      These properties are computed using the dictionary returned
      by the genepro.node.tree_numerical_properties method.
      Default value is False.

    Returns
    -------
      list
        the counts encoded tree
    """
    size = len(operators) + n_features + 1
    counts = [0.0] * (size * (max_depth + 1))

    stack = [(tree, 0)]
    properties_dict = {k: 0.0 for k in
                       ["height", "n_nodes", "max_arity", "max_breadth", "n_leaf_nodes", "n_internal_nodes"]}
    levels = {}
    length = 1
    while length > 0:
        curr_node, curr_level = stack.pop(length - 1)
        length = length - 1 + curr_node.arity
        if additional_properties:
            properties_dict["n_nodes"] += 1.0
            curr_depth, curr_arity = curr_level, curr_node.arity
            if curr_depth > properties_dict["height"]:
                properties_dict["height"] = curr_depth
            if curr_arity > properties_dict["max_arity"]:
                properties_dict["max_arity"] = curr_arity
            if curr_arity == 0:
                properties_dict["n_leaf_nodes"] += 1.0
            else:
                properties_dict["n_internal_nodes"] += 1.0
            if curr_depth not in levels:
                levels[curr_depth] = 0.0
            levels[curr_depth] += 1.0
        start_index = curr_level * size
        node_content = curr_node.symb
        if node_content in operators:
            counts[start_index + operators.index(node_content)] += 1.0
        elif node_content.startswith("x_"):
            feature_index = int(node_content[2:])
            if feature_index < n_features:
                counts[start_index + len(operators) + feature_index] += 1.0
            else:
                raise Exception(
                    f"More features than declared. Declared: {n_features}. Feature index found: {feature_index}.")
        elif re.search(r'^[+-]?\d+(\.\d+)?([Ee][+-]?\d+)?$', node_content):
            counts[start_index + size - 1] += 1.0
        else:
            raise Exception(f"Unexpected node content: {str(node_content)}.")
        for child_index in range(curr_node.arity - 1, -1, -1):
            stack.append((curr_node.get_child(child_index), curr_level + 1))

    if additional_properties:
        for l in levels:
            if levels[l] > properties_dict["max_breadth"]:
                properties_dict["max_breadth"] = levels[l]
        height_n_nodes_ratio = (properties_dict["height"] + 1.0) / float(properties_dict["n_nodes"])
        max_arity_breadth_ratio = properties_dict["max_arity"] / float(properties_dict["max_breadth"])
        leaf_nodes_ratio = properties_dict["n_leaf_nodes"] / float(properties_dict["n_nodes"])
        counts = counts + [height_n_nodes_ratio, max_arity_breadth_ratio, leaf_nodes_ratio]

    return counts


def one_hot_encode_tree(tree: Node, operators: list, n_features: int, max_depth: int, max_arity: int) -> list:
    """
    Provides a one-hot encoded representation of a tree, traversing it from the root in a breadth-first manner,
    and considering it as a full tree.

    Parameters
    ----------
    tree : Node
      tree to be one-hot encoded

    operators : list
      list of all possible symbols allowed for operators

    n_features : int
      amount of allowed features in the tree

    max_depth : int
      maximal depth of the tree

    max_arity : int
      maximal arity of the tree

    Returns
    -------
      list
        the one-hot encoded tree
    """
    dictionary_encoded_tree = tree.get_dict_repr(max_arity)
    size = len(operators) + n_features + 1
    one_hot = []
    n_nodes = int((max_arity ** (max_depth + 1) - 1) / (max_arity - 1))
    for node_index in range(n_nodes):
        current_encoding = [0.0] * size
        if node_index in dictionary_encoded_tree:
            node_content = dictionary_encoded_tree[node_index]
            if node_content in operators:
                current_encoding[operators.index(node_content)] = 1.0
            elif node_content.startswith("x_"):
                feature_index = int(node_content[2:])
                if feature_index < n_features:
                    current_encoding[len(operators) + feature_index] = 1.0
                else:
                    raise Exception(
                        f"More features than declared. Declared: {n_features}. Feature index found: {feature_index}.")
            elif re.search(r'^[+-]?\d+(\.\d+)?([Ee][+-]?\d+)?$', node_content):
                current_encoding[size - 1] = 1.0
            else:
                raise Exception(f"Unexpected node content: {str(node_content)}.")

        one_hot = one_hot + current_encoding
    return one_hot


def compute_linear_model_discovered_in_math_formula_interpretability_paper(tree: Node,
                                                                           difficult_operators: List[str] = None) -> float:
    """
    Compute the linear model discovered in math formula interpretability paper: https://arxiv.org/abs/2004.11170

    Parameters
    ----------
    tree : Node
      tree to be evaluated

    difficult_operators : list
      list of symbols representing difficult operations (if None, a default list of non-arithmetic operations in used)

    Returns
    -------
      float
        the interpretability of the tree, the higher, the better
    """
    if difficult_operators is None:
        difficult_operators = ['**2', '**3', '**', 'sqrt', 'log', 'exp', 'sin', 'cos',
                               'max', 'min', 'arcsin', 'arccos', 'tanh',
                               'sigmoid', 'cotanh', 'arctanh', 'arccotanh',
                               'sinh', 'cosh', 'arcsinh', 'arccosh', 'tan',
                               'cotan', 'arctan', 'arccotan']
    d = {"n_nodes": 0, "n_op": 0, "n_nop": 0}
    n_consecutive_non_arithmetic_operations = __count_linear_model_features(tree, difficult_operators, d)
    n_nodes, n_operations, n_non_arithmetic_operations = d["n_nodes"], d["n_op"], d["n_nop"]
    return 79.1 - 0.2*n_nodes - 0.5*n_operations - 3.4*n_non_arithmetic_operations - 4.5*n_consecutive_non_arithmetic_operations


def __count_linear_model_features(tree: Node, difficult_operators: List[str], d, count=None):
    if count is None:
        count = 0
    d["n_nodes"] += 1
    if tree.symb in difficult_operators:
        count += 1
        d["n_nop"] += 1
        d["n_op"] += 1
        count_args = []
        for c in tree._children:
            count_args.append(__count_linear_model_features(c, difficult_operators, d, count))
        count = max(count_args)
        return count
    else:
        if tree.arity > 0:
            d["n_op"] += 1
            count_args = []
            for c in tree._children:
                count_args.append(__count_linear_model_features(c, difficult_operators, d, count))
            count = max(count_args)
        return count


def replace_specified_operators_with_mean_value_constants(tree: Node, X: np.ndarray, operators: List[str]) -> Node:
    """
    This method generates a new tree starting from the input one. In this new tree, every node whose symbol
    is specified in the operators list is replaced with a constant containing the mean value of the predictions
    of that particular node applied to the training data X.

    Parameters
    ----------
    tree : Node
      starting tree

    X : ndarray
      training data to use to compute the mean values to be used as constants in place of the specified operators

    operators : list
      list of symbols representing operators to be replaced with constants

    Returns
    -------
      Node
        a new tree
    """
    tree = deepcopy(tree)
    nodes = [(tree, 0, None, -1)]
    length = 1
    while length > 0:
        curr_node, curr_level, curr_parent, curr_child_id = nodes.pop(0)
        length = length - 1 + curr_node.arity
        if curr_node.symb not in operators:
            for i in range(curr_node.arity):
                child = curr_node.get_child(i)
                nodes.append((child, curr_level + 1, child.parent, child.child_id))
        else:
            new_node = Constant(round(float(np.mean(curr_node(X))), 2))
            if curr_parent:
                curr_parent.replace_child(new_node, curr_child_id)
            else:
                tree = new_node
    return tree
