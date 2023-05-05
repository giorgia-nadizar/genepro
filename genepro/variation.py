import random
from typing import Callable, List

import numpy as np
from numpy.random import random as randu
from numpy.random import normal as randn
from numpy.random import choice as randc
from numpy.random import randint as randi
from numpy.random import shuffle
from copy import deepcopy

from genepro.node import Node
from genepro.node_impl import Constant, Minus, Plus, Sigmoid, Pointer, Times


def generate_random_tree(internal_nodes: list, leaf_nodes: list, max_depth: int, curr_depth: int = 0, ephemeral_func: Callable = None, p: List[float] = None):
    """
    Recursive method to generate a random tree containing the given types of nodes and up to the given maximum depth

    Parameters
    ----------
    internal_nodes : list
      list of nodes with arity > 0 (i.e., sub-functions or operations)
    leaf_nodes : list
      list of nodes with arity==0 (also called terminals, e.g., features and constants)
    max_depth : int
      maximum depth of the tree (recall that the root node has depth 0)
    curr_depth : int
      the current depth of the tree under construction, it is set by default to 0 so that calls to `generate_random_tree` need not specify it
    ephemeral_func: Callable
      lambda expression with no parameters that generates a potentially different random constant every time this method is called, default is None, meaning that no ephemeral constant is generated
    p : list
      probability distribution over internal nodes. Default is None, meaning uniform distribution.

    Returns
    -------
    Node
      the root node of the generated tree
    """
    if max_depth < 0:
        raise AttributeError(f"Max depth is negative: {max_depth}")
    if curr_depth < 0:
        raise AttributeError(f"Curr depth is negative: {curr_depth}")
    if len(internal_nodes) == 0:
        raise AttributeError("Internal nodes list is empty.")
    if len(leaf_nodes) == 0:
        raise AttributeError("Leaf nodes list is empty.")
    # heuristic to generate a semi-normal centered on relatively large trees
    prob_leaf = (0.01 + (curr_depth / max_depth) ** 3) if max_depth != 0 else 1.0

    if ephemeral_func is None:
        leaf_nodes_0 = leaf_nodes
    else:
        leaf_nodes_0 = leaf_nodes + [Constant(round(ephemeral_func(), 2))]

    if curr_depth == max_depth or randu() < prob_leaf:
        n = deepcopy(randc(leaf_nodes_0))
    else:
        n = deepcopy(randc(internal_nodes, p=p))

    for _ in range(n.arity):
        c = generate_random_tree(internal_nodes, leaf_nodes, max_depth, curr_depth + 1, ephemeral_func, p)
        n.insert_child(c)

    return n


def generate_random_forest(internal_nodes: list, leaf_nodes: list, max_depth: int, n_trees: int = None, n_trees_min: int = 2, n_trees_max: int = 10, tree_prob: float = 0.70, ephemeral_func: Callable = None, p: List[float] = None) -> List[Node]:
    """
    Recursive method to generate a random tree containing the given types of nodes and up to the given maximum depth

    Parameters
    ----------
    internal_nodes : list
      list of nodes with arity > 0 (i.e., sub-functions or operations)
    leaf_nodes : list
      list of nodes with arity==0 (also called terminals, e.g., features and constants)
    max_depth : int
      maximum depth of the tree (recall that the root node has depth 0)
    n_trees : int
      number of trees to generate. If None, this value is uniformly sampled between n_trees_min and n_trees_max (both included)
    n_trees_min : int
      when n_trees is None, the minimum number of trees to generate
    n_trees_max : int
      when n_trees is None, the maximal number of trees to generate
    tree_prob : float
      probability to generate a tree in the forest with a depth that is at least 1 (i.e., with probability tree_prob generate a tree, else generate a tree that consists of a single leaf only)
    ephemeral_func: Callable
      lambda expression with no parameters that generates a potentially different random constant every time this method is called, default is None, meaning that no ephemeral constant is generated
    p : list
      probability distribution over internal nodes. Default is None, meaning uniform distribution.

    Returns
    -------
    list
      list with generated nodes
    """
    if n_trees is not None and n_trees < 2:
        raise AttributeError(f"Number of trees to generate must be at least 2. Specified {n_trees} instead.")
    if not (0 <= tree_prob <= 1):
        raise AttributeError(f"Tree prob must be a float between 0 and 1. Specified {tree_prob} instead.")
    f: List[Node] = []
    if n_trees is None:
        if n_trees_min > n_trees_max:
            raise AttributeError(f"n_trees_min must not be greater than n_trees_max. Here n_trees_min is {n_trees_min} and n_trees_max is {n_trees_max}.")
        n_trees: int = randi(low=n_trees_min, high=n_trees_max + 1)
    for _ in range(n_trees):
        if randu() < tree_prob:
            current_max_depth: int = max_depth
        else:
            current_max_depth: int = 0
        f.append(generate_random_tree(internal_nodes, leaf_nodes, max_depth=current_max_depth, curr_depth=0, ephemeral_func=ephemeral_func, p=p))
    return f


def subtree_crossover(tree: Node, donor: Node, unif_depth: int = True) -> Node:
    """
    Performs subtree crossover and returns the resulting offspring

    Parameters
    ----------
    tree : Node
      the tree that participates and is modified by crossover
    donor : Node
      the second tree that participates in crossover, it provides candidate subtrees
    unif_depth : bool, optional
      whether uniform random depth sampling is used to pick the root of the subtrees to swap (default is True)

    Returns
    -------
    Node
      the tree after crossover (warning: replace the original tree with the returned one to avoid undefined behavior)
    """
    # pick a subtree to replace
    n = __sample_node(tree, unif_depth)
    m = deepcopy(__sample_node(donor, unif_depth))

    # remove ref to parent of m
    m.parent = None
    m.child_id = -1
    # swap
    p = n.parent
    if p:
        p.replace_child(m, n.child_id)
    else:
        tree = m
    return tree


def safe_subtree_crossover_two_children(tree1: Node, tree2: Node, unif_depth: int = True, max_depth: int = 4) -> tuple[Node, Node]:
    """
    Performs subtree crossover and returns the resulting offsprings without changing the original trees

    Parameters
    ----------
    tree1 : Node
      the first tree participating in the crossover
    tree2 : Node
      the second tree participating in the crossover
    unif_depth : bool, optional
      whether uniform random depth sampling is used to pick the root of the subtrees to swap (default is True)
    max_depth : int, optional
      max depth of the offsprings (default is 4)
    Returns
    -------
    Node
      the trees after crossover
    """
    tree1_get_height = tree1.get_height()
    tree2_get_height = tree2.get_height()
    if max_depth < 0:
        raise AttributeError(f"Max depth is negative: {max_depth}")
    if tree1_get_height > max_depth:
        raise ValueError(f"Max depth of offspring is set to be {max_depth} while height of the first tree is {tree1_get_height}. However, height of the first tree must be at most equal to max depth.")
    if tree2_get_height > max_depth:
        raise ValueError(f"Max depth of offspring is set to be {max_depth} while height of the second tree is {tree2_get_height}. However, height of the second tree must be at most equal to max depth.")

    tree1 = deepcopy(tree1)
    tree2 = deepcopy(tree2)
    if tree1_get_height < tree2_get_height:
        tree1, tree2 = tree2, tree1

    # pick a subtree to replace
    child1 = __sample_node(tree1, unif_depth)
    tree1_mutated_branch_max_depth = max_depth - child1.get_depth()
    child1_height = child1.get_height()
    candidates = []
    __find_cuts_subtree_crossover_two_children(tree2, max_depth, tree1_mutated_branch_max_depth, child1_height, candidates)
    child2 = randc(candidates)

    # swap
    parent1 = child1.parent
    parent2 = child2.parent

    new_child1 = deepcopy(child1)
    new_child2 = deepcopy(child2)
    new_child1.parent = None
    new_child1.child_id = -1
    new_child2.parent = None
    new_child2.child_id = -1

    if parent1:
        parent1.replace_child(new_child2, child1.child_id)
    else:
        tree1 = new_child2

    if parent2:
        parent2.replace_child(new_child1, child2.child_id)
    else:
        tree2 = new_child1

    return tree1, tree2


def __find_cuts_subtree_crossover_two_children(tree: Node, max_depth: int, tree1_mutated_branch_max_depth: int, child1_height: int, candidates: List[Node]) -> None:
    tree2_mutated_branch_max_depth = max_depth - tree.get_depth()
    child2_height = tree.get_height()
    if child2_height <= tree1_mutated_branch_max_depth and child1_height <= tree2_mutated_branch_max_depth:
        candidates.append(tree)
    for i in range(tree.arity):
        __find_cuts_subtree_crossover_two_children(tree.get_child(i), max_depth, tree1_mutated_branch_max_depth, child1_height, candidates)


def geometric_semantic_single_tree_crossover(tree1: Node, tree2: Node, internal_nodes: list[Node], leaf_nodes: list[Node], max_depth: int = 4, ephemeral_func: Callable = None, p: List[float] = None, cache: dict[Node, np.ndarray] = None, store_in_cache: bool = False) -> Node:
    """
    Performs geometric semantic crossover and returns the resulting offspring without changing the original trees

    Parameters
    ----------
    tree1 : Node
      the first tree participating in the crossover
    tree2 : Node
      the second tree participating in the crossover
    internal_nodes : list
      list of possible internal nodes to generate the random tree
    leaf_nodes : list
      list of possible leaf nodes to generate the random tree
    max_depth : int, optional
      the maximal depth of the generated random tree (default is 4)
    ephemeral_func: Callable
      lambda expression with no parameters that generates a potentially different random constant every time this method is called, default is None, meaning that no ephemeral constant is generated
    p : list
      probability distribution over internal nodes. Default is None, meaning uniform distribution.
    cache : dict
      cache that stores results for trees that have been seen during the evolution
    store_in_cache : bool
      if True, it uses the cache dictionary to store already computed results, otherwise, it computes every time the results of the whole tree
      
    Returns
    -------
    Node
      the tree after crossover
    """
    if cache is None:
        cache = dict()

    r = Sigmoid()
    r.insert_child(generate_random_tree(internal_nodes=internal_nodes, leaf_nodes=leaf_nodes, curr_depth=0, max_depth=max_depth, ephemeral_func=ephemeral_func, p=p))
    
    minus = Minus()
    minus.insert_child(Constant(1.0))
    minus.insert_child(Pointer(r, cache=cache, store_in_cache=store_in_cache))

    mul1 = Times()
    mul1.insert_child(Pointer(tree1, cache=cache, store_in_cache=store_in_cache))
    mul1.insert_child(Pointer(r, cache=cache, store_in_cache=store_in_cache))

    mul2 = Times()
    mul2.insert_child(Pointer(tree2, cache=cache, store_in_cache=store_in_cache))
    mul2.insert_child(minus)

    plus = Plus()
    plus.insert_child(mul1)
    plus.insert_child(mul2)
    return plus


def node_level_crossover(tree: Node, donor: Node, same_depth: bool = False, prob_swap: float = 0.1) -> Node:
    """
    Performs crossover at the level of single nodes

    Parameters
    ----------
    tree : Node
      the tree that participates and is modified by crossover
    donor : Node
      the second tree for crossover, which provides candidate nodes
    same_depth : bool, optional
      whether node-level swaps should occur only between nodes at the same depth level (default is False)
    prob_swap : float, optional
      the probability of swapping a node in tree with one in donor (default is 0.1)

    Returns
    -------
    Node
      the tree after crossover
    """
    nodes = tree.get_subtree()
    donor_nodes = donor.get_subtree()

    donor_node_arity = dict()
    donor_node_arity_n_depth = dict()
    for n in donor_nodes:
        arity = n.arity
        if arity not in donor_node_arity:
            donor_node_arity[arity] = [n]
        else:
            donor_node_arity[arity].append(n)
        # also record depths if same_depth==True
        if same_depth:
            depth = n.get_depth()
            ar_n_dep = (arity, depth)
            if ar_n_dep not in donor_node_arity_n_depth:
                donor_node_arity_n_depth[ar_n_dep] = [n]
            else:
                donor_node_arity_n_depth[ar_n_dep].append(n)

    for n in nodes:
        if randu() < prob_swap:
            # find compatible donor
            arity = n.arity
            if same_depth:
                depth = n.get_depth()
                compatible_nodes = donor_node_arity_n_depth[(arity, depth)] if (arity,
                                                                                depth) in donor_node_arity_n_depth else None
            else:
                compatible_nodes = donor_node_arity[arity] if arity in donor_node_arity else None
            # if no compatible nodes, skip
            if compatible_nodes is None or len(compatible_nodes) == 0:
                continue
            # swap
            m = deepcopy(randc(compatible_nodes))
            m.parent = None
            m.child_id = -1
            m._children = list()
            p = n.parent
            if p:
                p.replace_child(m, n.child_id)
            else:
                tree = m
            for c in n._children:
                m.insert_child(c)

    return tree


def safe_node_level_crossover_two_children(tree1: Node, tree2: Node, same_depth: bool = False,
                                           prob_swap: float = 0.1) -> tuple[Node, Node]:
    """
    Performs crossover at the level of single nodes, without changing the original trees

    Parameters
    ----------
    tree1 : Node
      the first tree participating in the crossover
    tree2 : Node
    same_depth : bool, optional
      whether node-level swaps should occur only between nodes at the same depth level (default is False)
    prob_swap : float, optional
      the probability of swapping a node in tree with one in donor (default is 0.1)

    Returns
    -------
    Node
      the tree after crossover
    """
    tree1 = deepcopy(tree1)
    tree2 = deepcopy(tree2)

    nodes1 = tree1.get_subtree()
    nodes2 = tree2.get_subtree()

    nodes2_arity = dict()
    nodes2_arity_n_depth = dict()

    for child2 in nodes2:
        arity = child2.arity
        if arity not in nodes2_arity:
            nodes2_arity[arity] = [child2]
        else:
            nodes2_arity[arity].append(child2)
        # also record depths if same_depth==True
        if same_depth:
            depth = child2.get_depth()
            ar_n_dep = (arity, depth)
            if ar_n_dep not in nodes2_arity_n_depth:
                nodes2_arity_n_depth[ar_n_dep] = [child2]
            else:
                nodes2_arity_n_depth[ar_n_dep].append(child2)

    for child1 in nodes1:
        if randu() < prob_swap:
            # find compatible donor
            arity = child1.arity
            if same_depth:
                depth = child1.get_depth()
                compatible_nodes = nodes2_arity_n_depth[(arity, depth)] if (arity,
                                                                            depth) in nodes2_arity_n_depth else None
            else:
                compatible_nodes = nodes2_arity[arity] if arity in nodes2_arity else None
            # if no compatible nodes, skip
            if compatible_nodes is None or len(compatible_nodes) == 0:
                continue

            # swap
            child2 = randc(compatible_nodes)

            parent1 = child1.parent
            parent2 = child2.parent

            children1 = child1._children
            children2 = child2._children

            new_child1 = deepcopy(child1)
            new_child2 = deepcopy(child2)

            new_child1._children = list()
            new_child2._children = list()

            new_child1.parent = None
            new_child1.child_id = -1
            new_child2.parent = None
            new_child2.child_id = -1

            if parent1:
                parent1.replace_child(new_child2, child1.child_id)
            else:
                tree1 = new_child2
            for c in children1:
                new_child2.insert_child(c)

            if parent2:
                parent2.replace_child(new_child1, child2.child_id)
            else:
                tree2 = new_child1
            for c in children2:
                new_child1.insert_child(c)

    return tree1, tree2


def safe_subforest_one_point_crossover_two_children(forest: List[Node], donor: List[Node], max_length: int = None) -> tuple[List[Node], List[Node]]:
    """
    Performs subtree crossover and returns the resulting offsprings without changing the original trees

    Parameters
    ----------
    forest : List[Node]
      the first list of trees participating in the crossover
    donor : List[Node]
      the second list of trees participating in the crossover
    max_length : int
      the maximal allowed length of an offspring. If None, the resulting forests have no limit in length.
    Returns
    -------
    tuple
      the lists of trees after crossover
    """
    if max_length is not None and max_length < 2:
        raise AttributeError(f"Max length must be greater than 1. Specified {max_length} instead.")
    forest_1: List[Node] = [deepcopy(x) for x in forest]
    forest_2: List[Node] = [deepcopy(x) for x in donor]
    if len(forest_2) < len(forest_1):
        forest_2, forest_1 = forest_1, forest_2
    cut_index: int = randc(list(range(len(forest_1))))
    child_1: List[Node] = forest_1[:cut_index] + forest_2[cut_index:]
    child_2: List[Node] = forest_2[:cut_index] + forest_1[cut_index:]
    if max_length is None:
        return child_1, child_2
    else:
        if len(child_1) > max_length:
            possible_cuts: List[int] = list(range(len(child_1) - max_length + 1))
            cut: int = randc(possible_cuts)
            child_1 = child_1[cut:(cut + max_length)]
        if len(child_2) > max_length:
            possible_cuts: List[int] = list(range(len(child_2) - max_length + 1))
            cut: int = randc(possible_cuts)
            child_2 = child_2[cut:(cut + max_length)]
        return child_1, child_2


def subtree_mutation(tree: Node, internal_nodes: list, leaf_nodes: list,
                     unif_depth: bool = True, max_depth: int = 4, ephemeral_func: Callable = None, p: List[float] = None) -> Node:
    """
    Performs subtree mutation and returns the resulting offspring

    Parameters
    ----------
    tree : Node
      the tree that participates and is modified by crossover
    internal_nodes : list
      list of possible internal nodes to generate the mutated branch
    leaf_nodes : list
      list of possible leaf nodes to generate the mutated branch
    unif_depth : bool, optional
      whether uniform random depth sampling is used to pick the root of the subtree to mutate (default is True)
    max_depth : int, optional
      the maximal depth of the offspring (default is 4)
    ephemeral_func: Callable
      lambda expression with no parameters that generates a potentially different random constant every time this method is called, default is None, meaning that no ephemeral constant is generated
    p : list
      probability distribution over internal nodes. Default is None, meaning uniform distribution.

    Returns
    -------
    Node
      the tree after mutation (warning: replace the original tree with the returned one to avoid undefined behavior)
    """
    tree_get_height = tree.get_height()
    if max_depth < 0:
        raise AttributeError(f"Max depth is negative: {max_depth}")
    if tree_get_height > max_depth:
        raise ValueError(f"Max depth of offspring is set to be {max_depth} while height of the input tree is {tree_get_height}. However, height of the tree must be at most equal to max depth.")
    # pick a subtree to replace
    n = __sample_node(tree, unif_depth)
    # generate a random branch
    branch = generate_random_tree(internal_nodes=internal_nodes, leaf_nodes=leaf_nodes,
                                  max_depth=max_depth - n.get_depth(), curr_depth=0, ephemeral_func=ephemeral_func,
                                  p=p)
    # swap
    p = n.parent
    if p:
        p.replace_child(branch, n.child_id)
    else:
        tree = branch
    return tree


def safe_subtree_mutation(tree: Node, internal_nodes: list, leaf_nodes: list,
                          unif_depth: bool = True, max_depth: int = 4, ephemeral_func: Callable = None, p: List[float] = None) -> Node:
    """
    Performs subtree mutation and returns the resulting offspring.
    Differs from subtree mutation as it is not done in place.

    Parameters
    ----------
    tree : Node
      the tree that participates and is modified by crossover
    internal_nodes : list
      list of possible internal nodes to generate the mutated branch
    leaf_nodes : list
      list of possible leaf nodes to generate the mutated branch
    unif_depth : bool, optional
      whether uniform random depth sampling is used to pick the root of the subtree to mutate (default is True)
    max_depth : int, optional
      the maximal depth of the offspring (default is 4)
    ephemeral_func: Callable
      lambda expression with no parameters that generates a potentially different random constant every time this method is called, default is None, meaning that no ephemeral constant is generated
    p : list
      probability distribution over internal nodes. Default is None, meaning uniform distribution.

    Returns
    -------
    Node
      the tree after mutation (warning: replace the original tree with the returned one to avoid undefined behavior)
    """
    return subtree_mutation(deepcopy(tree), internal_nodes, leaf_nodes, unif_depth=unif_depth, max_depth=max_depth, ephemeral_func=ephemeral_func, p=p)


def geometric_semantic_tree_mutation(tree: Node, internal_nodes: list, leaf_nodes: list, max_depth: int = 4, ephemeral_func: Callable = None, p: List[float] = None, m: float = 0.5, cache: dict[Node, np.ndarray] = None, store_in_cache: bool = False) -> Node:
    """
    Performs geometric semantic tree mutation. Pointers are used to avoid generation very large trees to store in memory.

    Parameters
    ----------
    tree : Node
      the tree that should be mutated
    internal_nodes : list
      list of possible internal nodes to generate the random tree
    leaf_nodes : list
      list of possible leaf nodes to generate the random tree
    max_depth : int, optional
      the maximal depth of the generated random tree (default is 4)
    ephemeral_func: Callable
      lambda expression with no parameters that generates a potentially different random constant every time this method is called, default is None, meaning that no ephemeral constant is generated
    p : list
      probability distribution over internal nodes. Default is None, meaning uniform distribution.
    m : float
      a coefficient in the geometric semantic mutation
    cache : dict
      cache that stores results for trees that have been seen during the evolution
    store_in_cache : bool
      if True, it uses the cache dictionary to store already computed results, otherwise, it computes every time the results of the whole tree
      
    Returns
    -------
    Node
      the tree after mutation (warning: replace the original tree with the returned one to avoid undefined behavior)
    """
    if cache is None:
        cache = dict()
    
    r = generate_random_tree(internal_nodes=internal_nodes, leaf_nodes=leaf_nodes, curr_depth=0, max_depth=max_depth, ephemeral_func=ephemeral_func, p=p)
    
    mul = Times()
    mul.insert_child(Constant(m))
    mul.insert_child(Pointer(r, cache=cache, store_in_cache=store_in_cache))

    plus = Plus()
    plus.insert_child(Pointer(tree, cache=cache, store_in_cache=store_in_cache))
    plus.insert_child(mul)
    return plus


def safe_subforest_mutation(forest: List[Node], internal_nodes: list, leaf_nodes: list,
                            unif_depth: bool = True, max_depth: int = 4, ephemeral_func: Callable = None, p: List[float] = None) -> List[Node]:
    """
    Performs subforest mutation and returns the resulting offspring. The mutation is applied at each tree in the forest
    with probability 1/len(forest). The single tree mutation is performed by calling subtree_mutation method.

    Parameters
    ----------
    forest : List[Node]
      the forest that participates and is modified by crossover
    internal_nodes : list
      list of possible internal nodes to generate the mutated branch
    leaf_nodes : list
      list of possible leaf nodes to generate the mutated branch
    unif_depth : bool, optional
      whether uniform random depth sampling is used to pick the root of the subtree to mutate (default is True)
    max_depth : int, optional
      the maximal depth of the offspring (default is 4)
    ephemeral_func: Callable
      lambda expression with no parameters that generates a potentially different random constant every time this method is called, default is None, meaning that no ephemeral constant is generated
    p : list
      probability distribution over internal nodes. Default is None, meaning uniform distribution.

    Returns
    -------
    List[Node]
      the forest after mutation (warning: replace the original tree with the returned one to avoid undefined behavior)
    """
    f: List[Node] = []
    mutation_prob: float = 1.0/float(len(forest))
    for n in forest:
        if randu() < mutation_prob:
            f.append(subtree_mutation(deepcopy(n), internal_nodes, leaf_nodes, unif_depth=unif_depth, max_depth=max_depth, ephemeral_func=ephemeral_func, p=p))
        else:
            f.append(deepcopy(n))
    return f


def coeff_mutation(tree: Node, prob_coeff_mut: float = 0.25, temp: float = 0.25) -> Node:
    """
    Applies random coefficient mutations to constant nodes

    Parameters
    ----------
    tree : Node
      the tree to which coefficient mutations are applied
    prob_coeff_mut : float, optional
      the probability with which coefficients are mutated (default is 0.25)
    temp : float, optional
      "temperature" that indicates the strength of coefficient mutation, it is relative to the current value (i.e., v' = v + temp*abs(v)*N(0,1))

    Returns
    -------
    Node
      the tree after coefficient mutation (it is the same as the tree in input)
    """
    coeffs = [n for n in tree.get_subtree() if type(n) == Constant]
    for c in coeffs:
        # decide whether it should be applied
        if randu() < prob_coeff_mut:
            v = c.get_value()
            # update the value by +- temp relative to current value
            new_v = v + temp * np.abs(v) * randn()
            c.set_value(new_v)

    return tree


def safe_coeff_mutation(tree: Node, prob_coeff_mut: float = 0.25, temp: float = 0.25) -> Node:
    """
    Applies random coefficient mutations to constant nodes.
    Differs from coefficient mutation as it is not done in place.

    Parameters
    ----------
    tree : Node
      the tree to which coefficient mutations are applied
    prob_coeff_mut : float, optional
      the probability with which coefficients are mutated (default is 0.25)
    temp : float, optional
      "temperature" that indicates the strength of coefficient mutation, it is relative to the current value (i.e., v' = v + temp*abs(v)*N(0,1))

    Returns
    -------
    Node
      the tree after coefficient mutation (it is the same as the tree in input)
    """
    return coeff_mutation(deepcopy(tree), prob_coeff_mut, temp)


def __sample_node(tree: Node, unif_depth: bool = True) -> Node:
    """
    Helper method that samples a random node from a tree

    Parameters
    ----------
    tree : Node
      the tree from which a random node should be sampled
    unif_depth : bool, optional
      whether the depth of the random node should be sampled uniformly at random first (default is True)

    Returns
    -------
    Node
      the randomly sampled node
    """
    nodes = tree.get_subtree()
    if unif_depth:
        nodes = __sample_uniform_depth_nodes(nodes)
    return randc(nodes)


def __sample_uniform_depth_nodes(nodes: list) -> list:
    """
    Helper method for `__sample_node` that returns candidate nodes that all have a depth which was sampled uniformly at random

    Parameters
    ----------
    nodes : list
      list of nodes from which to sample candidates that share a random depth (typically the result of `get_subtree()`)

    Returns
    -------
    list:
      list of nodes that share a depth that was sampled uniformly at random
    """
    depths = [n.get_depth() for n in nodes]
    possible_depths = list(set(depths))
    d = randc(possible_depths)
    return [n for i, n in enumerate(nodes) if depths[i] == d]


def generate_offspring(parent: Node,
                       crossovers: list, mutations: list, coeff_opts: list,
                       donors: list, internal_nodes: list, leaf_nodes: list,
                       constraints: dict = {"max_tree_size": 100}) -> Node:
    """
    Generates an offspring from a given parent (possibly using a donor from the population for crossover).
    Variation operators are applied in a random order.
    The application of the variation operator is handled by `__undergo_variation_operator`

    Parameters
    ----------
    parent : Node
      the parent tree from which the offspring is generated by applying the variation operators
    crossovers : list
      list of dictionaries each specifying a type of crossover and respective hyper-parameters
    mutations : list
      list of dictionaries each specifying a type of mutation and respective hyper-parameters
    coeff_opts : list
      list of dictionaries each specifying a type of coefficient optimization and respective hyper-parameters
    donors : list
      list of Node, each representing a donor tree that can be used by crossover
    internal_nodes : list
      list of internal nodes to be used by mutation
    leaf_nodes : list
      list of internal nodes to be used by mutation
    constraints : dict, optional
      constraints the generated offspring must meet (default is {"max_size": 100})

    Returns
    -------
    Node
      the offspring after having applied the variation operators
    """
    # set the offspring to a copy (to be modified) of the parent
    offspring = deepcopy(parent)
    # create a backup for constraint violation
    backup = deepcopy(offspring)

    # apply variation operators in a random order
    all_var_ops = crossovers + mutations + coeff_opts
    random_order = np.arange(len(all_var_ops))
    shuffle(random_order)
    for i in random_order:
        var_op = all_var_ops[i]
        offspring = __undergo_variation_operator(var_op, offspring,
                                                 crossovers, mutations, coeff_opts,
                                                 randc(donors), internal_nodes, leaf_nodes)
        # check offspring meets constraints, else revert to backup
        if not __check_tree_meets_all_constraints(offspring, constraints):
            # revert to backup
            offspring = deepcopy(backup)
        else:
            # update backup
            backup = deepcopy(offspring)

    return offspring


def __undergo_variation_operator(var_op: dict, offspring: Node,
                                 crossovers, mutations, coeff_opts,
                                 donor, internal_nodes, leaf_nodes) -> Node:
    # decide whether to actually do something
    if var_op["rate"] < randu():
        # nope
        return offspring

        # prepare the function to call
        var_op_fun = var_op["fun"]
        # next, we need to provide the right arguments based on the type of ops
        if var_op in crossovers:
            # we need a donor
            offspring = var_op_fun(offspring, donor, **var_op["kwargs"])
        elif var_op in mutations:
            # we need to provide node types
            offspring = var_op_fun(offspring, internal_nodes, leaf_nodes, **var_op["kwargs"])
        elif var_op in coeff_opts:
            offspring = var_op_fun(offspring, **var_op["kwargs"])

        return offspring


def __check_tree_meets_all_constraints(tree: Node, constraints: dict = dict()) -> bool:
    """
    """
    meets = True
    for constraint_name in constraints.keys():
        if constraint_name == "max_tree_size":
            if len(tree.get_subtree()) > constraints["max_tree_size"]:
                meets = False
                break
        else:
            raise ValueError("Unrecognized constraint name: {}".format(constraint_name))
    return meets
