from __future__ import annotations
import numpy as np

from genepro.util import tree_from_prefix_repr


class Node:
    """
    Class to represent a node in a tree.
    When used to represent the root, it also represents the tree.
    Implementations (in `genepro.node_impl`) inherit from this class and extend it to represent operations, features, or constants.

    Attributes
    ----------
    symb : str
      the symbol that represents the function of this node (e.g., "+" for Plus)
    fitness : float
      the fitness of the tree, only meaningful for the root note
    parent : Node
      the parent node of this node, it is None for the root note
    arity : int
      number of input arguments this node accepts (e.g., 2 for Plus, 1 for Log)
    _children : list
      list of nodes that are children whose parent is this node (warning: do not write on this field directly, use `insert_child` or `detach_child`)
    """

    def __init__(self):
        self.symb = None
        self.fitness = -np.inf
        self.parent = None
        self.arity = 0
        self._children = list()

    def __repr__(self) -> str:
        """
        Returns the string representation of this node

        Returns
        -------
        str
          the attribute symb
        """
        return self.symb

    def __len__(self) -> int:
        """
        Returns the length of the subtree rooted at this node

        Returns
        -------
          length of the subtree rooted at this node
        """
        return len(self.get_subtree())

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Shorthand for get_output

        Parameters
        ----------
        X : np.ndarray
          the input to be processed as an numpy ndarray with dimensions (num. observations, num. features)

        Returns
        -------
        np.ndarray
          the output obtained by processing the input
        """
        return self.get_output(X)

    def get_subtree(self) -> list:
        """
        Returns the subtree rooted at this node, as a list of nodes visited in prefix order

        Returns
        -------
        list
          the subtree (including this node) as a list of descendant nodes in prefix order
        """
        subtree = list()
        self.__get_subtree_recursive(subtree)
        return subtree

    def get_readable_repr(self) -> str:
        """
        Builds a human-readable representation of the subtree

        Returns
        -------
        str
          Human-readable representation of the subtree
        """
        repr = [""]  # trick to pass string by reference
        self.__get_readable_repr_recursive(repr)
        return repr[0]

    def insert_child(self, c: Node, i: int = None):
        """
        Inserts the given node among the children of this node

        Parameters
        ----------
        c : Node
          the node to insert as a child of this node
        i : int, optional
          the position at which to insert c (default is None, in which case c is appended)
        """
        if i is None:
            self._children.append(c)
        else:
            self._children.insert(i, c)
        c.parent = self

    def detach_child(self, c: Node) -> int:
        """
        Removes the given node from the children of this node and returns its position relative to the other children

        Parameters
        ----------
        c : Node
          the node to detach

        Returns
        -------
        int
          index specifying the position of c in the attribute _children
        """
        assert (c in self._children)
        for i, oc in enumerate(self._children):
            if c == oc:
                self._children.pop(i)
                c.parent = None
                break
        return i

    def get_output(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the output of this node after processing the given input

        Parameters
        ----------
        X : np.ndarray
          the input to be processed as an numpy ndarray with dimensions (num. observations, num. features)

        Returns
        -------
        np.ndarray
          the output obtained by processing the input
        """
        raise NotImplementedError()

    def get_depth(self) -> int:
        """
        Returns the depth of this node (the root node has depth 0)

        Returns
        -------
        int
          the depth of this node
        """
        n = self
        d = 0
        while n.parent:
            d += 1
            n = n.parent
        return d

    def get_height(self) -> int:
        """
        Computes and returns the height of this node

        Returns
        -------
        int
          the height of this node
        """
        curr_depth = self.get_depth()
        leaves = [x for x in self.get_subtree() if x.arity == 0]
        max_h = 0
        for l in leaves:
            d = l.get_depth()
            if d > max_h:
                max_h = d
        return max_h - curr_depth

    def get_n_nodes(self) -> int:
        """
        Computes and returns the number of nodes in the tree

        Returns
        -------
        int
          the amount of nodes in this tree
        """
        size = 1
        for child in self._children:
            size = size + child.get_n_nodes()
        return size

    def _get_child_outputs(self, X: np.ndarray) -> list:
        """
        Returns the output of the children for the given input as a list

        Parameters
        ----------
        X : np.ndarray
          the input to be processed as an numpy ndarray with dimensions (num. observations, num. features)

        Returns
        -------
        list
          list containing the output of the children, each as a numpy.ndarray
        """
        c_outs = list()
        for i in range(self.arity):
            c_o = self._children[i].get_output(X)
            c_outs.append(c_o)
        return c_outs

    def _get_args_repr(self, args: list) -> str:
        """
        Returns a string representing the way this node processes its input arguments (i.e., its children)

        Parameters
        ----------
        args : list
          list of strings, each representing the way a child node processes its input arguments

        Returns
        -------
        str
          representation of the way this node processes its input arguments
        """
        raise NotImplementedError()

    def _get_typical_repr(self, args: list, name: str) -> str:
        """
        Helper method for `_get_args_repr` implementing typical representations

        Parameters
        ----------
        args : list
          list of strings, each representing the way a child node processes its input arguments
        name : str, options are "between", "before", and "after"
          name of the typical representation, needs to match with the length of args (e.g., "between" for Plus results in "args[0] + args[1]")

        Returns
        -------
        str
          typical representation of the way this node processes its input arguments
        """
        if name == 'between':
            if len(args) != 2:
                raise ValueError("Invalid representation 'between' for len(args)!=2")
            return '(' + args[0] + self.symb + args[1] + ')'
        elif name == 'before':
            repr = self.symb + '('
            for arg in args:
                repr += arg + ','
            repr = repr[:-1] + ')'
            return repr
        elif name == 'after':
            repr = '('
            for arg in args:
                repr += arg + ','
            repr = repr[:-1] + ')'
            repr += self.symb
            return repr
        else:
            raise ValueError("Unrecognized option {}".format(name))

    def __get_subtree_recursive(self, subtree: list):
        """
        Helper method for `get_subtree` that uses recursion to visit the descendant nodes and populate the given list

        Parameters
        ----------
        subtree : list
          list that is populated by including this node and then calling this method again on the children of that node
        """
        subtree.append(self)
        for c in self._children:
            c.__get_subtree_recursive(subtree)

    def __get_readable_repr_recursive(self, repr: list):
        """
        Helper method for get_subtree_repr that uses recursion to visit the descendant nodes and populate the given list

        Parameters
        ----------
        repr : list
          list that is used as container to fill a string with the result of `_get_args_repr` of this node and its descendants
        """
        args = list()
        for i in range(self.arity):
            self._children[i].__get_readable_repr_recursive(repr)
            args.append(repr[0])
        repr[0] = self._get_args_repr(args)

    def get_child(self, idx: int) -> Node:
        """
        Returns the child placed at the given index.

        Parameters
        ----------
        idx : int
          the index of the child

        Returns
        -------
        Node
          the node representing the child at the given index
        """
        if not (0 <= idx < self.arity):
            raise IndexError(f"{idx} is out of range for current node arity.")
        return self._children[idx]

    def get_dict_repr(self, max_arity: int, node_index=0) -> dict:
        """
        Returns a dictionary-encoded tree, where the keys are the indexes of the node (considering a breadth-first
        exploration of a complete tree of max_arity) and the values are the symbolic content of the node.

        Parameters
        ----------
        max_arity : int
          the maximal arity of the tree (or of the forest considered)
        node_index : int
          the index of the starting node

        Returns
        -------
        dict
          dictionary mapping the index of the node to its symbolic content
        """
        representation = {node_index: self.symb}
        for i in range(self.arity):
            representation = representation | self._children[i].get_dict_repr(max_arity, node_index * max_arity + 1 + i)
        return representation

    def tree_numerical_properties(self) -> dict:
        """
        Returns a dict of integers, where each integer represents a numerical property of the given tree.
        The keys are described as follows:
        - height: contains a value that is equals to the output value of get_height()
        - n_nodes: contains a value that is equals to the output value of get_n_nodes()
        - max_arity: the arity of the node with the maximum number of children
        - max_breadth: the number of nodes in the level with the maximum number of nodes
        - n_leaf_nodes: the number of leaf nodes
        - n_internal_nodes: the number of internal nodes

        Returns
        -------
        dict
          dict of integers, where each integer represents a numerical property of the given tree
        """
        queue = [self]
        properties_dict = {k: 0.0 for k in
                           ["height", "n_nodes", "max_arity", "max_breadth", "n_leaf_nodes", "n_internal_nodes"]}
        levels = {}
        while len(queue) > 0:
            curr_node = queue.pop(0)
            properties_dict["n_nodes"] += 1.0
            curr_depth, curr_arity = curr_node.get_depth(), curr_node.arity
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
            for child_index in range(curr_arity):
                queue.append(curr_node.get_child(child_index))
        for l in levels:
            if levels[l] > properties_dict["max_breadth"]:
                properties_dict["max_breadth"] = levels[l]
        return properties_dict

    def __deepcopy__(self):
        return tree_from_prefix_repr(str(self.get_subtree()))
