from genepro.util import counts_encode_tree, counts_level_wise_encode_tree
import numpy as np
import genepro.variation
import genepro
import random
from copy import deepcopy
import time

if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    a = genepro.variation.generate_random_tree([genepro.node_impl.Plus(),
                                                genepro.node_impl.Minus(),
                                                genepro.node_impl.Times(),
                                                genepro.node_impl.Div(),
                                                genepro.node_impl.Log(),
                                                genepro.node_impl.Max(), genepro.node_impl.Min()],
                                               [genepro.node_impl.Feature(0), genepro.node_impl.Feature(1),
                                                genepro.node_impl.Feature(2), genepro.node_impl.Feature(3),
                                                genepro.node_impl.Constant(5)],
                                               6)
    b = deepcopy(a)
    print(a.__hash__())
    print(b.__hash__())
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
