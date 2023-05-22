from __future__ import annotations

import re
import zlib
from typing import List

import numpy as np


class Node:
    """
    Class to represent a node in a tree.
    When used to represent the root, it also represents the tree.
    Implementations (in `genepro.node_impl`) inherit from this class and extend it to represent operations, features, or constants.

    Parameters
    ----------
    fix_properties : bool
      if True, it computes hash, readable representation, number of nodes, height, and lisp expr of the node the first time these
      properties need to be computed and it stores these values within variables, the next time these methods are invoked again,
      they avoid to re-compute these properties from scratch and they simply look-up the stored values, this avoids useless
      waste of time in cases in which the tree does not change.
      Mind that if the tree changes then you must compute again all these properties since the stored values are not valid anymore,
      to do this, you must disable the fix_properties flag and set it to False.

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
    child_id : id
      id that goes from 0 to len(parent._children)-1. It represents an ID for the node as children of a given parent (-1 if the node has no parent)
    attr : dict
      dictionary where you can store any type of information about the tree
    """

    def __init__(self,
                 fix_properties: bool = False,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.symb = None
        self.fitness = -np.inf
        self.parent = None
        self.arity = 0
        self._children = list()
        self.child_id = -1
        self.attr = dict()

        self.__fix_properties: bool = fix_properties
        self.__hash_value: int = None
        self.__readable_repr_value: str = None
        self.__lisp_expr_value: str = None
        self.__n_nodes_value: int = None
        self.__height_value: int = None

    def get_fix_properties(self) -> bool:
        """
        Returns whether or not you chose to fix the properties in this node

        Returns
        -------
        bool
          the fix properties value
        """
        return self.__fix_properties
    
    def set_fix_properties(self, fix_properties: bool) -> bool:
        """
        Set the fix properties value again

        Parameters
        ----------
        fix_properties : bool
          the new fix properties value

        Returns
        -------
        bool
          the previous fix properties value
        """
        old_fix_properties: bool = self.get_fix_properties()
        self.__fix_properties = fix_properties
        return old_fix_properties

    def __repr__(self) -> str:
        """
        Returns the string representation of this node

        Returns
        -------
        str
          the attribute symb
        """
        return self.symb
    
    def __str__(self) -> str:
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
        return self.get_n_nodes()

    def __call__(self, X: np.ndarray, **kwargs) -> np.ndarray:
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
        return self.get_output(X, **kwargs)

    def __eq__(self, other: Node) -> bool:
        """
        Shorthand for structurally_equal method.

        Parameters
        ----------
        other : Node
          another tree

        Returns
        -------
        bool
          whether the two trees are structurally equal
        """
        return self.structurally_equal(other)

    def __hash__(self) -> int:
        """
        Hash method for the tree

        Returns
        -------
        int
          hash code of the tree
        """
        if not self.get_fix_properties() or self.__hash_value is None:
          molt = 31
          h = 0
          h = h * molt + self._single_hash_value()
          for c in self._children:
              h = h * molt + hash(c)
          self.__hash_value = h
        
        return self.__hash_value

    def _single_hash_value(self) -> int:
        return zlib.adler32(bytes(self.symb, "utf-8"))

    def structurally_equal(self, other: Node) -> bool:
        """
        Check if two trees are exactly the same as regards their structure.

        Parameters
        ----------
        other : Node
          another tree

        Returns
        -------
        bool
          whether the two trees are structurally equal
        """
        return self.get_readable_repr() == other.get_readable_repr()

    def semantically_equal(self, other: Node, X: np.ndarray, **kwargs) -> bool:
        """
        Check if two trees are semantically equal.

        Parameters
        ----------
        other : Node
          another tree

        X : np.ndarray
          small numerical dataset that is used to check whether the two trees output the same results

        Returns
        -------
        bool
          whether the two trees are semantically equal
        """
        return self(X, **kwargs).tolist() == other(X, **kwargs).tolist()

    def get_subtree(self) -> list:
        """
        Returns the subtree rooted at this node, as a list of nodes visited in prefix order

        Returns
        -------
        list
          the subtree (including this node) as a list of descendant nodes in prefix order
        """
        subtree = []
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
        if not self.get_fix_properties() or self.__readable_repr_value is None:
          repr = [""]  # trick to pass string by reference
          self.__get_readable_repr_recursive(repr)
          self.__readable_repr_value = repr[0]
        
        return self.__readable_repr_value

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
        children_list_length = len(self._children)
        if i is None:
            if children_list_length == 0:
                new_child_id = 0
            else:
                new_child_id = self._children[children_list_length - 1].child_id + 1
            self._children.append(c)
            c.child_id = new_child_id
        else:
            self._children.insert(i, c)
            for iii in range(i, len(self._children)):
                self._children[iii].child_id = iii
        c.parent = self

    def replace_child(self, c: Node, i: int):
        """
        Replace the node at the given position with the provided node

        Parameters
        ----------
        c : Node
          the node to insert as a child of this node
        i : int, optional
          the position at which to insert c while replacing the previous node
        """
        self._children[i].parent = None
        self._children[i].child_id = -1
        self._children[i] = c
        c.child_id = i
        c.parent = self

    def remove_child(self, i: int):
        """
        Remove the node at the given position

        Parameters
        ----------
        i : int, optional
          the position of the node to remove
        """
        c = self._children.pop(i)
        c.parent = None
        c.child_id = -1

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
        i = -1
        assert (c in self._children)
        for i, oc in enumerate(self._children):
            if c is oc:
                self._children.pop(i)
                c.parent = None
                c.child_id = -1
                break
        return i

    def get_output(self, X: np.ndarray, **kwargs) -> np.ndarray:
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

    def create_new_empty_node(self, **kwargs) -> Node:
        """
        Returns a new empty instance of the given type of node

        Returns
        -------
        Node
          a new empty instance of the given type of node
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
        if not self.get_fix_properties() or self.__height_value is None:
          if self.arity == 0:
              return 0
          a = []
          for c in self._children:
              a.append(c.get_height())
          self.__height_value = 1 + max(a)
        
        return self.__height_value

    def get_n_nodes(self) -> int:
        """
        Computes and returns the number of nodes in the tree

        Returns
        -------
        int
          the amount of nodes in this tree
        """
        if not self.get_fix_properties() or self.__n_nodes_value is None:
          if self.arity == 0:
              return 1
          a = []
          for c in self._children:
              a.append(c.get_n_nodes())
          self.__n_nodes_value = 1 + sum(a)

        return self.__n_nodes_value

    def _get_child_outputs(self, X: np.ndarray, **kwargs) -> list:
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
        c_outs = []
        for i in range(self.arity):
            c_outs.append(self._children[i].get_output(X, **kwargs))
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
        args = []
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
            representation.update(self._children[i].get_dict_repr(max_arity, node_index * max_arity + 1 + i))
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
        length = 1
        while length > 0:
            curr_node = queue.pop(0)
            length = length - 1 + curr_node.arity
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

    def retrieve_operators_from_tree(self) -> List[str]:
        """
        Returns a list of operator symbols that are represented in the tree

        Returns
        -------
        list
          list of operator symbols that are represented in the tree
        """
        stack = [self]
        operators = []
        length = 1
        while length > 0:
            curr_node = stack.pop(length - 1)
            length = length - 1 + curr_node.arity
            node_content = curr_node.symb
            if not((isinstance(node_content, str) and re.search(r'^[+-]?\d+(\.\d+)?([Ee][+-]?\d+)?$',
                                                              node_content)) or isinstance(node_content,
                                                                                           float) or isinstance(
                    node_content, int)) and not(node_content.startswith("x_")):
                operators.append(node_content)
            for i in range(curr_node.arity):
                stack.append(curr_node.get_child(i))
        return operators

    def retrieve_features_from_tree(self) -> List[str]:
        """
        Returns a list of feature symbols that are represented in the tree

        Returns
        -------
        list
          list of feature symbols that are represented in the tree
        """
        stack = [self]
        features = []
        length = 1
        while length > 0:
            curr_node = stack.pop(length - 1)
            length = length - 1 + curr_node.arity
            node_content = curr_node.symb
            if node_content.startswith("x_"):
                features.append(node_content)
            for i in range(curr_node.arity):
                stack.append(curr_node.get_child(i))
        return features

    def retrieve_constants_from_tree(self) -> List[str]:
        """
        Returns a list of constant symbols that are represented in the tree

        Returns
        -------
        list
          list of constant symbols that are represented in the tree
        """
        stack = [self]
        constants = []
        length = 1
        while length > 0:
            curr_node = stack.pop(length - 1)
            length = length - 1 + curr_node.arity
            node_content = curr_node.symb
            if (isinstance(node_content, str) and re.search(r'^[+-]?\d+(\.\d+)?([Ee][+-]?\d+)?$',
                                                              node_content)) or isinstance(node_content,
                                                                                           float) or isinstance(
                    node_content, int):
                constants.append(node_content)
            for i in range(curr_node.arity):
                stack.append(curr_node.get_child(i))
        return constants

    def get_string_as_tree(self) -> str:
        """
        Returns a string representation of tree in the form of a tree with square brackets around nodes

        Returns
        -------
        str
          string representation
        """
        s = ""
        depth = self.get_height()
        s += "  " * (depth + 1)
        nodes = [(self, 0)]
        subtree = []
        length = 1
        while length > 0:
            curr_node, curr_level = nodes.pop(0)
            length = length - 1 + curr_node.arity
            subtree.append((curr_node, curr_level))
            for i in range(curr_node.arity):
                nodes.append((curr_node.get_child(i), curr_level + 1))
        curr_layer = 0
        for curr_node, curr_level in subtree:
            if curr_level > curr_layer:
                curr_layer = curr_level
                s += "\n"
                depth -= 1
                s += "  " * (depth + 1)
            s += "  [" + curr_node._get_single_string_repr_tree() + "] "
        return s

    def get_string_as_lisp_expr(self) -> str:
        """
        Returns a string representation of tree in the form of a lisp expression

        Returns
        -------
        str
          string representation
        """
        if not self.get_fix_properties() or self.__lisp_expr_value is None:
          s = ""
          nodes = [self]
          while len(nodes) > 0:
              curr_node = nodes.pop(len(nodes) - 1)
              if isinstance(curr_node, str):
                  s += curr_node + " "
              else:
                  s += curr_node._get_single_string_repr_lisp() + " "
                  if curr_node.arity > 0:
                      nodes.append(")")
                      for i in range(curr_node.arity - 1, -1, -1):
                          nodes.append(curr_node.get_child(i))
                      nodes.append("(")
          self.__lisp_expr_value = s.strip()

        return self.__lisp_expr_value
    
    def _get_single_string_repr_tree(self) -> str:
        return self.symb
    
    def _get_single_string_repr_lisp(self) -> str:
        return self.symb
