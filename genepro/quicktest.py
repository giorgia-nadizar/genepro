from genepro.node import Node
from genepro.util import counts_encode_tree, counts_level_wise_encode_tree
import numpy as np
import genepro.variation
import genepro
import random
from copy import deepcopy
import time

if __name__ == "__main__":
    random.seed(2)
    np.random.seed(2)

    data = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])

    and_gate: Node = genepro.node_impl.And()
    and_gate.insert_child(genepro.node_impl.Feature(0))
    and_gate.insert_child(genepro.node_impl.Feature(1))
    print(and_gate(data))

    or_gate: Node = genepro.node_impl.Or()
    or_gate.insert_child(genepro.node_impl.Feature(0))
    or_gate.insert_child(genepro.node_impl.Feature(1))
    print(or_gate(data))

    xor_gate: Node = genepro.node_impl.Xor()
    xor_gate.insert_child(genepro.node_impl.Feature(0))
    xor_gate.insert_child(genepro.node_impl.Feature(1))
    print(xor_gate(data))

    not_gate: Node = genepro.node_impl.Not()
    not_gate.insert_child(genepro.node_impl.Feature(0))
    print(not_gate(np.array([[1], [0]])))

    a = genepro.variation.generate_random_forest([genepro.node_impl.And()],
                                                 [genepro.node_impl.Feature(0), genepro.node_impl.Feature(1),
                                                  genepro.node_impl.Feature(2), genepro.node_impl.Feature(3)
                                                  ],
                                                 max_depth=2, n_trees=3
                                                 )
    b = genepro.variation.safe_subforest_mutation(a, [genepro.node_impl.And()],
                                                  [genepro.node_impl.Feature(0), genepro.node_impl.Feature(1),
                                                  genepro.node_impl.Feature(2), genepro.node_impl.Feature(3)
                                                   ], max_depth=2)
    bc = genepro.util.concatenate_nodes_with_binary_operator(b, genepro.node_impl.Xor())
    ac = genepro.util.concatenate_nodes_with_binary_operator(a, genepro.node_impl.Xor())
    print(ac.get_string_as_lisp_expr())
    print(bc.get_string_as_lisp_expr())
    data = np.array([[0, 0, 1, 1],
                     [1, 0, 1, 1],
                     [0, 1, 0, 0],
                     [1, 0, 0, 1]])
    print(ac(data))
    a1 = genepro.variation.generate_random_forest([genepro.node_impl.And()],
                                                 [genepro.node_impl.Feature(0), genepro.node_impl.Feature(1),
                                                  genepro.node_impl.Feature(2), genepro.node_impl.Feature(3)
                                                  ],
                                                 max_depth=2, n_trees=3
                                                 )
    a1c = genepro.util.concatenate_nodes_with_binary_operator(a1, genepro.node_impl.Xor())
    print(a1c.get_string_as_lisp_expr())
    off1, off2 = genepro.variation.safe_subforest_one_point_crossover_two_children(a, a1)
    off1 = genepro.util.concatenate_nodes_with_binary_operator(off1, genepro.node_impl.Xor())
    off2 = genepro.util.concatenate_nodes_with_binary_operator(off2, genepro.node_impl.Xor())
    print(off1.get_string_as_lisp_expr())
    print(off2.get_string_as_lisp_expr())
    

    a = genepro.variation.generate_random_tree([genepro.node_impl.Plus(),
                                                genepro.node_impl.Minus(),
                                                genepro.node_impl.Times(),
                                                genepro.node_impl.Div(),
                                                genepro.node_impl.Log(),
                                                genepro.node_impl.Max(), genepro.node_impl.Min()],
                                               [genepro.node_impl.Feature(0), genepro.node_impl.Feature(1),
                                                genepro.node_impl.Feature(2), genepro.node_impl.Feature(3),
                                                genepro.node_impl.Constant(5.0)],
                                               max_depth=6)
    b = deepcopy(a)
    print(hash(a))
    print(hash(b))
    print(a.__hash__())
    print(b.__hash__())
    print(a.get_readable_repr())
    print()
    print(a.get_string_as_tree())
    print()
    print(a.get_string_as_lisp_expr())
    print()
    cache: dict[Node, np.ndarray] = {}
    operators: list[Node] = [genepro.node_impl.Plus(),
                                                    genepro.node_impl.Minus(),
                                                    genepro.node_impl.Times(),
                                                    genepro.node_impl.Div(),
                                                    genepro.node_impl.Log()
                                                    ]
    terminals: list[Node] = [genepro.node_impl.Feature(0), genepro.node_impl.Feature(1),
                                                    genepro.node_impl.Feature(2), genepro.node_impl.Feature(3),
                                                    genepro.node_impl.Constant(5.0)]
    for _ in range(2):
        tree1: Node = genepro.variation.generate_random_tree(operators,
                                                terminals,
                                                max_depth=3)
    for _ in range(4):
        tree2: Node = genepro.variation.generate_random_tree(operators, terminals,
                                                max_depth=3)
    print('========================')
    print(tree1.get_string_as_lisp_expr())
    print()
    print(tree2.get_string_as_lisp_expr())
    crossover: Node = genepro.variation.geometric_semantic_single_tree_crossover(tree1=tree1, tree2=tree2, internal_nodes=operators, leaf_nodes=terminals, max_depth=3, cache=cache, store_in_cache=True)
    print()
    print(crossover.get_string_as_lisp_expr())
    print()
    print()
    print(tree1.get_readable_repr())
    print()
    print(tree2.get_readable_repr())
    print()
    print(crossover.get_readable_repr())
    print(hash(crossover))
    print()
    print()
    print(genepro.util.tree_from_prefix_repr(genepro.util.get_subtree_as_full_string(crossover)).get_readable_repr())
    print(hash(genepro.util.tree_from_prefix_repr(genepro.util.get_subtree_as_full_string(crossover))))
    print()
    print()
    print(tree1.get_string_as_tree())
    print()
    print(tree2.get_string_as_tree())
    print()
    print(crossover.get_string_as_tree())
    print()
    print()
    print('========================')
    mutation: Node = genepro.variation.geometric_semantic_tree_mutation(crossover, internal_nodes=operators, leaf_nodes=terminals, max_depth=3, m=0.85, cache=cache, store_in_cache=True)
    print()
    print(mutation.get_string_as_lisp_expr())
    print()
    print()
    print(mutation.get_readable_repr())
    print(hash(mutation))
    print(genepro.util.tree_from_prefix_repr(genepro.util.get_subtree_as_full_string(mutation)).get_readable_repr())
    print(hash(genepro.util.tree_from_prefix_repr(genepro.util.get_subtree_as_full_string(mutation))))
    print()
    print()
    print('========================')
    X: np.ndarray = np.array([[234, 21, -2, 0.45],
                              [78.342, 2.1, -343.1, 34.2],
                              [134, -21.3, 1.2, 9.45],
                              [34.78, 121.2, 75.5, 65.45],
                              [24.23, -51, 2, -23.45],
                              [7.3, 11.23, 42.6, 0.75]])
    print(cache)
    print(deepcopy(crossover)(X))
    print(cache)
    print(deepcopy(mutation)(X))
    print(cache)
    exit(1)
    st = time.time()
    for _ in range(10 ** 6):
        c = counts_encode_tree(a, ["+","-","*","/","log","max","min"], 6, True)
    en = time.time()
    print(c)
    print((en - st)*(1/3600)*60)
    exit(1)

    print(a.get_readable_repr())
    print(counts_level_wise_encode_tree(a, ["+","-","*","/","log","max","min"], 4, 6, 2, True))
    print(a == b)
    print(a.semantically_equals(b, np.array([[1,5,3,1],[6,4,3,5],[2,1,2,4],[6,4,5,7],[2,3,4,1],[5,4,2,1]])))
    b.symb = "/"
    print(a == b)
    print(a.semantically_equals(b, np.array([[1,5,3,1],[6,4,3,5],[2,1,2,4],[6,4,5,7],[2,3,4,1],[5,4,2,1]])))
    d = {0: "a"}
    dd = {1: "b"}
    ddd = {2: "c", 3: "d"}
    d.update(dd)
    d.update(ddd)
    print(d)
