import inspect
from copy import deepcopy

import numpy as np
import re

from genepro.node import Node
from genepro import node_impl
from genepro.node_impl import Feature, Constant, InstantiableConstant, OOHRdyFeature, GSGPCrossover, GSGPMutation, Pointer, RandomGaussianConstant, Times


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


def tree_from_prefix_repr(prefix_repr: str, fix_properties: bool = False, enable_caching: bool = False, slack: float = 1.0, **kwargs) -> Node:
    """
    Creates a tree from a string representation in prefix format (that is, pre-order tree traversal);
    the symbol in the string representation need to match those in the Node implementations (in `genepro.node_impl.py`)

    Parameters
    ----------
    prefix_repr : str
      the string representation of the tree as a list of nodes parsed with pre-order traversal (obtainable with `str(tree.get_subtree())`)
    fix_properties : bool
      the fix_properties attribute of the Node class
    enable_caching : bool
      if True, it enables caching for GSGPCrossover and GSGPMutation operator
  
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
        if node_cls == Node or node_cls == Feature or node_cls == Constant or node_cls == InstantiableConstant or node_cls == OOHRdyFeature or node_cls == Pointer or node_cls == RandomGaussianConstant or node_cls == GSGPCrossover or node_cls == GSGPMutation:
            continue
        node_obj = node_cls(fix_properties=fix_properties, **kwargs)
        possible_nodes.append(node_obj)
    tree, _ = __tree_from_symb_list_recursive(symb_list, possible_nodes, fix_properties=fix_properties, enable_caching=enable_caching, slack=slack, **kwargs)
    return tree


def __tree_from_symb_list_recursive(symb_list: list, possible_nodes: list, fix_properties: bool, enable_caching: bool, slack: float, **kwargs):
    """
    Helper recursive function for `tree_from_prefix_repr`

    Parameters
    ----------
    symb_list : list
      list of str that are symbols (as per the attribute `symb` of Node)
    possible_nodes : list
      list of all possible Node objects from `genepro.node_impl`
    fix_properties : bool
      the fix_properties attribute of the Node class
    enable_caching : bool
      if True, it enables caching for GSGPCrossover and GSGPMutation operator

    Returns
    -------
      Node, list
        the last-generated node and the updated symb_list, required for the recursive construction of the tree
    """
    symb = symb_list[0]
    symb_list = symb_list[1:]
    # check if it is a pointer
    if symb.startswith('pointer'):
        raise ValueError(f'Pointer cannot be converted to a tree, before building your string in prefix representation, you must first unpack all pointers in your tree. To this, end, make use of the get_subtree_as_full_list method which is capable of automatically unpacking all pointers in the tree.')

    # check if it is a feature
    if symb.startswith('x_'):
        id = int(symb[2:])
        n = Feature(id, fix_properties=fix_properties, **kwargs)
        return n, symb_list
    
    # check if it is a constant
    if re.search(r'^[+-]?\d+(\.\d+)?([Ee][+-]?\d+)?$', symb):
        n = Constant(float(symb), fix_properties=fix_properties, **kwargs)
        return n, symb_list

    # check if it is a random gaussian constant
    if symb.startswith('rgc_'):
        l: list[float] = [float(i) for i in symb.split('_')[1:]]
        n = RandomGaussianConstant(mean=l[0], std=l[1], fix_properties=fix_properties, **kwargs)
        return n, symb_list
    
    # check if it is a gsgp crossover
    if symb.startswith('gsgpcx'):
        n = GSGPCrossover(enable_caching=enable_caching, fix_properties=fix_properties, **kwargs)
        for _ in range(n.arity):
            c, symb_list = __tree_from_symb_list_recursive(symb_list, possible_nodes, fix_properties=fix_properties, enable_caching=enable_caching, **kwargs)
            n.insert_child(c)
        return n, symb_list
    
    # check if it is a gsgp mutation
    if symb.startswith('gsgpmut'):
        m: float = float(symb[len('gsgpmut'):])
        n = GSGPMutation(m=m, enable_caching=enable_caching, fix_properties=fix_properties, **kwargs)
        for _ in range(n.arity):
            c, symb_list = __tree_from_symb_list_recursive(symb_list, possible_nodes, fix_properties=fix_properties, enable_caching=enable_caching, **kwargs)
            n.insert_child(c)
        return n, symb_list

    # check if it is a function
    for pn in possible_nodes:
        if symb == pn.symb:
            n = pn.create_new_empty_node(**kwargs)
            if 'slack' in symb:
                n.slack = slack
            for _ in range(n.arity):
                c, symb_list = __tree_from_symb_list_recursive(symb_list, possible_nodes, fix_properties=fix_properties, enable_caching=enable_caching, **kwargs)
                n.insert_child(c)
            return n, symb_list
    
    raise ValueError(f'{symb} unrecognized as symbol for possible functions of genepro.')


def get_subtree_as_full_list(tree: Node) -> list[Node]:
    """
    Given the tree, it retrieves the subtree in prefix order with pointers replaced with all the referenced nodes.

    Parameters
    ----------
    tree : Node
      the tree

    Returns
    -------
    list[Node]
      the list containing all the nodes of the tree with pointers replaced with the full tree
    """
    subtree = []
    __get_subtree_as_full_list_recursive(tree, subtree)
    return subtree


def __get_subtree_as_full_list_recursive(tree: Node, subtree: list[Node]) -> None:
    """
    Helper method for `get_subtree_as_full_list` that uses recursion to visit the descendant nodes and populate the given list

    Parameters
    ----------
    tree : Node
      current node
    subtree : list
      list that is populated by including this node and then calling this method again on the children of that node, while replacing pointers with referenced nodes
    """
    if isinstance(tree, Pointer):
        __get_subtree_as_full_list_recursive(tree.get_value(), subtree)
    elif isinstance(tree, OOHRdyFeature):
        value = float(tree.get_value())
        id = tree.id
        if value == 1.0:
            __get_subtree_as_full_list_recursive(Feature(id), subtree)
        else:
            mul_op = Times()
            mul_op.insert_child(Constant(value))
            mul_op.insert_child(Feature(id))
            __get_subtree_as_full_list_recursive(mul_op, subtree)
    else:
        subtree.append(tree)
        for c in tree._children:
            __get_subtree_as_full_list_recursive(c, subtree)


def get_subtree_as_full_string(tree: Node) -> str:
    """
    Given the tree, it retrieves string of the subtree in prefix order with pointers replaced with all the referenced nodes.
    Once you have this string, you can call the method tree_from_prefix_repr on this string to obtain the original Node object.
    Therefore, you can use this string to make your tree be persistent on disk.

    Parameters
    ----------
    tree : Node
      the tree

    Returns
    -------
    str
      the string containing the list containing all the nodes of the tree with pointers replaced with the full tree
    """
    return str(get_subtree_as_full_list(tree))


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


def __counts_encode_tree_recursive(tree: Node, depth: int, size: int, operators: list, n_features: int, properties_dict: dict, levels: dict, counts: list, additional_properties: bool = True):
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
                                                                           difficult_operators: list[str] = None) -> float:
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


def __count_linear_model_features(tree: Node, difficult_operators: list[str], d, count=None):
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


def concatenate_nodes_with_binary_operator(forest: list[Node], binary_operator: Node, copy_tree: bool = False) -> Node:
    """
    This method generates a new tree starting from input forest (list of trees). The new tree is generated
    by concatenating the trees in the forest with the specified binary operator.

    Parameters
    ----------
    forest : list
      list of trees to concatenate

    binary_operator : Node
      binary operator to use for concatenating the trees in the forest

    copy_tree : bool
      specify if you want to perform deepcopy of each tree in the forest when building the concatenated tree

    Returns
    -------
      Node
        a new tree
    """
    if binary_operator.arity != 2:
        raise AttributeError(f"The arity of the specified binary operator is {binary_operator.arity} instead of 2.")
    if len(forest) < 2:
        raise AttributeError(f"Forest (list of nodes to concatenate) must have at least two trees. Here we have only {len(forest)} trees.")
    c: Node = deepcopy(binary_operator)
    if len(forest) == 2:
        c.insert_child(deepcopy(forest[0]) if copy_tree else forest[0])
        c.insert_child(deepcopy(forest[1]) if copy_tree else forest[1])
        return c
    c.insert_child(deepcopy(forest[0]) if copy_tree else forest[0])
    c.insert_child(concatenate_nodes_with_binary_operator(forest[1:], binary_operator, copy_tree))
    return c


def replace_specified_operators_with_mean_value_constants(tree: Node, X: np.ndarray, operators: list[str]) -> Node:
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
