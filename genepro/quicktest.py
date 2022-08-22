from genepro.util import counts_encode_tree
import numpy as np
import genepro.variation
import genepro

from copy import deepcopy

if __name__ == "__main__":

    a = genepro.variation.generate_random_tree([genepro.node_impl.Plus(),
                                                genepro.node_impl.Minus(),
                                                genepro.node_impl.Times(),
                                                genepro.node_impl.Div(),
                                                genepro.node_impl.Log(),
                                                genepro.node_impl.Max(), genepro.node_impl.Min()],
                                               [genepro.node_impl.Feature(0), genepro.node_impl.Feature(1),
                                                genepro.node_impl.Feature(2), genepro.node_impl.Feature(3),
                                                genepro.node_impl.Constant()],
                                               6)
    b = deepcopy(a)
    print(counts_encode_tree(a, ["+","-","*","/","log","max","min"], 4, 6, 2, True))
    print(a == b)
    print(a.semantically_equals(b, np.array([[1,5,3,1],[6,4,3,5],[2,1,2,4],[6,4,5,7],[2,3,4,1],[5,4,2,1]])))
    b.symb = "/"
    print(a == b)
    print(a.semantically_equals(b, np.array([[1,5,3,1],[6,4,3,5],[2,1,2,4],[6,4,5,7],[2,3,4,1],[5,4,2,1]])))
