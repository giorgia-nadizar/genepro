import numpy as np
from genepro.node import Node
from genepro.storage import WeakCache


class Plus(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 2
        self.symb = '+'

    def create_new_empty_node(self) -> Node:
        return Plus(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'between')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return c_outs[0] + c_outs[1]


class Minus(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 2
        self.symb = '-'

    def create_new_empty_node(self) -> Node:
        return Minus(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'between')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return c_outs[0] - c_outs[1]


class Times(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 2
        self.symb = '*'

    def create_new_empty_node(self) -> Node:
        return Times(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'between')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.multiply(np.core.umath.clip(c_outs[0], -1e+100, 1e+100), np.core.umath.clip(c_outs[1], -1e+100, 1e+100))


class Div(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 2
        self.symb = '/'

    def create_new_empty_node(self) -> Node:
        return Div(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'between')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        # implements a protection to avoid dividing by 0
        c_outs[0] = np.core.umath.clip(c_outs[0], -1e+100, 1e+100)
        c_outs[1] = np.core.umath.clip(c_outs[1], -1e+100, 1e+100)
        sign_b = np.sign(c_outs[1])
        sign_b = np.where(sign_b == 0, 1, sign_b)
        protected_div = sign_b * ( c_outs[0] / ( np.core.umath.clip(np.abs(c_outs[1]), 1e-9, 1e+100) ) )
        return protected_div


class Mod(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 2
        self.symb = '%'

    def create_new_empty_node(self) -> Node:
        return Mod(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'between')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        # implements a protection to avoid dividing by 0
        c_outs[0] = np.core.umath.clip(c_outs[0], -1e+100, 1e+100)
        c_outs[1] = np.core.umath.clip(c_outs[1], -1e+100, 1e+100)
        sign_b = np.sign(c_outs[1])
        sign_b = np.where(sign_b == 0, 1, sign_b)
        protected_div = sign_b * ( c_outs[0] % ( np.core.umath.clip(np.abs(c_outs[1]), 1, 1e+100) ) )
        return protected_div


class Square(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 1
        self.symb = '**2'

    def create_new_empty_node(self) -> Node:
        return Square(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'after')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        # Implement protection with clip to avoid overflow encoutered error and avoid getting inf as outputs
        return np.square(np.core.umath.clip(c_outs[0], -1.340780792993396e+100, 1.340780792993396e+100))


class Cube(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 1
        self.symb = '**3'

    def create_new_empty_node(self) -> Node:
        return Cube(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'after')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        # Implement protection with clip to avoid overflow encoutered error and avoid getting inf as outputs
        return np.power(np.core.umath.clip(c_outs[0], -5.643803094119938e+70, 5.643803094119938e+70), 3)


class Sqrt(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 1
        self.symb = 'sqrt'

    def create_new_empty_node(self) -> Node:
        return Sqrt(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        # let's report also protection
        return "sqrt(abs(" + args[0] + "))"

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        # implements a protection to avoid arg <= 0
        return np.sqrt(np.abs(c_outs[0]))


class Log(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 1
        self.symb = 'log'

    def create_new_empty_node(self) -> Node:
        return Log(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        # let's report also protection (to some level of detail)
        return "log(abs(" + args[0] + "))"

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        # implements a protection to avoid arg <= 0
        protected_log = np.log(np.abs(c_outs[0]) + 1e-9)
        return protected_log


class Exp(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 1
        self.symb = "exp"

    def create_new_empty_node(self) -> Node:
        return Exp(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        # Implement protection with clip to avoid overflow encoutered error and avoid getting inf as outputs
        return np.exp(np.core.umath.clip(c_outs[0], -700.78, 700.78))


class Sin(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 1
        self.symb = "sin"

    def create_new_empty_node(self) -> Node:
        return Sin(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.sin(c_outs[0])


class Cos(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 1
        self.symb = "cos"

    def create_new_empty_node(self) -> Node:
        return Cos(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.cos(c_outs[0])


class Max(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 2
        self.symb = "max"

    def create_new_empty_node(self) -> Node:
        return Max(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, "before")

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.where(c_outs[0] > c_outs[1], c_outs[0], c_outs[1])


class Min(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 2
        self.symb = "min"

    def create_new_empty_node(self) -> Node:
        return Min(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, "before")

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.where(c_outs[0] < c_outs[1], c_outs[0], c_outs[1])


class And(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 2
        self.symb = "and"

    def create_new_empty_node(self) -> Node:
        return And(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, "between")

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.where(np.logical_or(c_outs[0] == 0, c_outs[1] == 0), 0, 1)


class Or(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 2
        self.symb = "or"

    def create_new_empty_node(self) -> Node:
        return Or(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, "between")

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.where(np.logical_and(c_outs[0] == 0, c_outs[1] == 0), 0, 1)


class Xor(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 2
        self.symb = "xor"

    def create_new_empty_node(self) -> Node:
        return Xor(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, "between")

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.where(
            np.logical_or(
                np.logical_and(c_outs[0] == 0, c_outs[1] != 0),
                np.logical_and(c_outs[0] != 0, c_outs[1] == 0)
            ), 1, 0)


class IfThenElse(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 3
        self.symb = "if-then-else"

    def create_new_empty_node(self) -> Node:
        return IfThenElse(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return "if(" + args[0] + " >= 0)then(" + args[1] + ")else(" + args[2] + ")"

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.where(c_outs[0] >= 0, c_outs[1], c_outs[2])


class Feature(Node):
    def __init__(self,
                 id: int,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 0
        self.id = id
        self.symb = 'x_' + str(id)

    def create_new_empty_node(self) -> Node:
        return Feature(id=self.id, fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self.symb

    def get_output(self, X):
        return X[:, self.id]


class Constant(Node):
    def __init__(self,
                 value: float,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        if value is None:
            raise AttributeError("The value provided in the constructor of Constant is None.")
        self.arity = 0
        self.__value = value
        self.symb = str(value)

    def create_new_empty_node(self) -> Node:
        return Constant(value=self.__value, fix_properties=self.get_fix_properties())

    def get_value(self):
        return self.__value

    def set_value(self, value: float):
        self.__value = value
        self.symb = str(value)

    def _get_args_repr(self, args):
        # make sure it is initialized
        self.get_value()
        return self.symb

    def get_output(self, X: np.ndarray) -> np.ndarray:
        # make sure it is initialized
        v = self.get_value()
        return np.repeat(v, len(X))


class Pointer(Node):
    def __init__(self,
                 value: Node,
                 cache: WeakCache = None,
                 store_in_cache: bool = False,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        if value is None:
            raise AttributeError("The value provided in the constructor of Pointer is None.")
        self.__cache: WeakCache = cache
        self.__store_in_cache: bool = store_in_cache
        self.arity = 0
        self.__value = value
        self.symb = self.__value.get_readable_repr()

    def create_new_empty_node(self) -> Node:
        return Pointer(value=self.__value, cache=self.__cache, store_in_cache=self.__store_in_cache, fix_properties=self.__fix_properties)

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        return Pointer(value=self.__value, cache=self.__cache, store_in_cache=self.__store_in_cache, fix_properties=self.__fix_properties)

    def get_height(self) -> int:
        return self.__value.get_height()

    def get_n_nodes(self) -> int:
        return self.__value.get_n_nodes()

    def get_value(self):
        return self.__value
    
    def get_cache(self):
        return self.__cache

    def get_store_in_cache(self) -> bool:
        return self.__store_in_cache

    def set_store_in_cache(self, store_in_cache: bool) -> bool:
        old_store_in_cache = self.get_store_in_cache()
        self.__store_in_cache = store_in_cache
        return old_store_in_cache

    def set_value(self, value: Node):
        self.__value = value
        self.symb = self.__value.get_readable_repr()

    def _get_args_repr(self, args):
        return self.__value.get_readable_repr()
    
    def _single_hash_value(self) -> int:
        return hash(self.__value)
    
    def _get_single_string_repr_tree(self) -> str:
        return self.__value.get_string_as_tree()
    
    def _get_single_string_repr_lisp(self) -> str:
        return self.__value.get_string_as_lisp_expr()

    def get_output(self, X: np.ndarray) -> np.ndarray:
        if not self.__store_in_cache:
            return self.__value(X)
        r = self.__cache.get(self.__value)
        if r is not None:
            return r
        result = self.__value(X)
        self.__cache.set(self.__value, result)
        return result


class Power(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 2
        self.symb = '**'

    def create_new_empty_node(self) -> Node:
        return Power(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'between')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        # implements a protection to avoid raising negative values to non-integral values
        base = np.abs(c_outs[0]) + 1e-9
        exponent = c_outs[1]
        exponent = np.core.umath.clip(exponent, -30.0, 30.0)
        base = np.core.umath.clip(base, -90.0, 90.0)
        return np.power(base, exponent)


class Arcsin(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 1
        self.symb = "arcsin"

    def create_new_empty_node(self) -> Node:
        return Arcsin(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        # implements a protection to avoid arg out of [-1,1]
        return np.arcsin(np.core.umath.clip(c_outs[0], -1, 1))


class Arccos(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 1
        self.symb = "arccos"

    def create_new_empty_node(self) -> Node:
        return Arccos(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        # implements a protection to avoid arg out of [-1,1]
        return np.arccos(np.core.umath.clip(c_outs[0], -1, 1))


class Tanh(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 1
        self.symb = "tanh"

    def create_new_empty_node(self) -> Node:
        return Tanh(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.tanh(np.core.umath.clip(c_outs[0], -1e+100, 1e+100))
    

class Identity(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 1
        self.symb = "identity"

    def create_new_empty_node(self) -> Node:
        return Identity(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return c_outs[0]


class ReLU(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 1
        self.symb = "relu"

    def create_new_empty_node(self) -> Node:
        return ReLU(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.where(c_outs[0] > 0, c_outs[0], 0)


class Sigmoid(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 1
        self.symb = "sigmoid"

    def create_new_empty_node(self) -> Node:
        return Sigmoid(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        c_outs[0] = np.core.umath.clip(c_outs[0], -700.78, 700.78)
        return 1.0/(1.0 + np.exp(-c_outs[0]))


class UnaryMinus(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 1
        self.symb = "u-"

    def create_new_empty_node(self) -> Node:
        return UnaryMinus(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return -1 * c_outs[0]


class Not(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 1
        self.symb = "not"

    def create_new_empty_node(self) -> Node:
        return Not(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.where(c_outs[0] == 0, 1, 0)


class Even(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 1
        self.symb = "even"

    def create_new_empty_node(self) -> Node:
        return Even(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.where(c_outs[0] % 2 == 0, 1, 0)


class Odd(Node):
    def __init__(self,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__(fix_properties=fix_properties)
        self.arity = 1
        self.symb = "odd"

    def create_new_empty_node(self) -> Node:
        return Odd(fix_properties=self.get_fix_properties())

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.where(c_outs[0] % 2 != 0, 1, 0)
