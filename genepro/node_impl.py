import numpy as np
from genepro.node import Node


class Plus(Node):
    def __init__(self):
        super(Plus, self).__init__()
        self.arity = 2
        self.symb = '+'

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'between')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return c_outs[0] + c_outs[1]


class Minus(Node):
    def __init__(self):
        super(Minus, self).__init__()
        self.arity = 2
        self.symb = '-'

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'between')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return c_outs[0] - c_outs[1]


class Times(Node):
    def __init__(self):
        super(Times, self).__init__()
        self.arity = 2
        self.symb = '*'

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'between')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.multiply(np.core.umath.clip(c_outs[0], -1e+100, 1e+100), np.core.umath.clip(c_outs[1], -1e+100, 1e+100))


class Div(Node):
    def __init__(self):
        super(Div, self).__init__()
        self.arity = 2
        self.symb = '/'

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'between')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        # implements a protection to avoid dividing by 0
        sign_b = np.sign(c_outs[1])
        sign_b = np.where(sign_b == 0, 1, sign_b)
        protected_div = sign_b * np.core.umath.clip(c_outs[0], -1e+100, 1e+100) / (np.core.umath.clip(np.abs(np.core.umath.clip(c_outs[1], -1e+100, 1e+100)), 1e-9, 1e+100))
        return protected_div


class Mod(Node):
    def __init__(self):
        super(Mod, self).__init__()
        self.arity = 2
        self.symb = '%'

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'between')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        # implements a protection to avoid dividing by 0
        sign_b = np.sign(c_outs[1])
        sign_b = np.where(sign_b == 0, 1, sign_b)
        protected_div = sign_b * np.core.umath.clip(c_outs[0], -1e+100, 1e+100) % ( np.core.umath.clip(np.abs(np.core.umath.clip(c_outs[1], -1e+100, 1e+100)), 1, 1e+100))
        return protected_div


class Square(Node):
    def __init__(self):
        super(Square, self).__init__()
        self.arity = 1
        self.symb = '**2'

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'after')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        # Implement protection with clip to avoid overflow encoutered error and avoid getting inf as outputs
        return np.square(np.core.umath.clip(c_outs[0], -1.340780792993396e+100, 1.340780792993396e+100))


class Cube(Node):
    def __init__(self):
        super(Cube, self).__init__()
        self.arity = 1
        self.symb = '**3'

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'after')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        # Implement protection with clip to avoid overflow encoutered error and avoid getting inf as outputs
        return np.power(np.core.umath.clip(c_outs[0], -5.643803094119938e+70, 5.643803094119938e+70), 3)


class Sqrt(Node):
    def __init__(self):
        super(Sqrt, self).__init__()
        self.arity = 1
        self.symb = 'sqrt'

    def _get_args_repr(self, args):
        # let's report also protection
        return "sqrt(abs(" + args[0] + "))"

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        # implements a protection to avoid arg <= 0
        return np.sqrt(np.abs(c_outs[0]))


class Log(Node):
    def __init__(self):
        super(Log, self).__init__()
        self.arity = 1
        self.symb = 'log'

    def _get_args_repr(self, args):
        # let's report also protection (to some level of detail)
        return "log(abs(" + args[0] + "))"

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        # implements a protection to avoid arg <= 0
        protected_log = np.log(np.abs(c_outs[0]) + 1e-9)
        return protected_log


class Exp(Node):
    def __init__(self):
        super(Exp, self).__init__()
        self.arity = 1
        self.symb = "exp"

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        # Implement protection with clip to avoid overflow encoutered error and avoid getting inf as outputs
        return np.exp(np.core.umath.clip(c_outs[0], -700.78, 700.78))


class Sin(Node):
    def __init__(self):
        super(Sin, self).__init__()
        self.arity = 1
        self.symb = "sin"

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.sin(c_outs[0])


class Cos(Node):
    def __init__(self):
        super(Cos, self).__init__()
        self.arity = 1
        self.symb = "cos"

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.cos(c_outs[0])


class Max(Node):
    def __init__(self):
        super(Max, self).__init__()
        self.arity = 2
        self.symb = "max"

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, "before")

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.where(c_outs[0] > c_outs[1], c_outs[0], c_outs[1])


class Min(Node):
    def __init__(self):
        super(Min, self).__init__()
        self.arity = 2
        self.symb = "min"

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, "before")

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.where(c_outs[0] < c_outs[1], c_outs[0], c_outs[1])


class And(Node):
    def __init__(self):
        super(And, self).__init__()
        self.arity = 2
        self.symb = "and"

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, "between")

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.where(np.logical_or(c_outs[0] == 0, c_outs[1] == 0), 0, 1)


class Or(Node):
    def __init__(self):
        super(Or, self).__init__()
        self.arity = 2
        self.symb = "or"

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, "between")

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.where(np.logical_and(c_outs[0] == 0, c_outs[1] == 0), 0, 1)


class Xor(Node):
    def __init__(self):
        super(Xor, self).__init__()
        self.arity = 2
        self.symb = "xor"

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
    def __init__(self):
        super(IfThenElse, self).__init__()
        self.arity = 3
        self.symb = "if-then-else"

    def _get_args_repr(self, args):
        return "if(" + args[0] + " >= 0)then(" + args[1] + ")else(" + args[2] + ")"

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.where(c_outs[0] >= 0, c_outs[1], c_outs[2])


class Feature(Node):
    def __init__(self, id):
        super(Feature, self).__init__()
        self.arity = 0
        self.id = id
        self.symb = 'x_' + str(id)

    def _get_args_repr(self, args):
        return self.symb

    def get_output(self, X):
        return X[:, self.id]


class Constant(Node):
    def __init__(self, value: float):
        super(Constant, self).__init__()
        if value is None:
            raise AttributeError("The value provided in the constructor of Constant is None.")
        self.arity = 0
        self.__value = value
        self.symb = str(value)

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


class Power(Node):
    def __init__(self):
        super(Power, self).__init__()
        self.arity = 2
        self.symb = '**'

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
    def __init__(self):
        super(Arcsin, self).__init__()
        self.arity = 1
        self.symb = "arcsin"

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        # implements a protection to avoid arg out of [-1,1]
        return np.arcsin(np.core.umath.clip(c_outs[0], -1, 1))


class Arccos(Node):
    def __init__(self):
        super(Arccos, self).__init__()
        self.arity = 1
        self.symb = "arccos"

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        # implements a protection to avoid arg out of [-1,1]
        return np.arccos(np.core.umath.clip(c_outs[0], -1, 1))


class Tanh(Node):
    def __init__(self):
        super(Tanh, self).__init__()
        self.arity = 1
        self.symb = "tanh"

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.tanh(c_outs[0])


class UnaryMinus(Node):
    def __init__(self):
        super(UnaryMinus, self).__init__()
        self.arity = 1
        self.symb = "u-"

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return -1 * c_outs[0]


class Not(Node):
    def __init__(self):
        super(Not, self).__init__()
        self.arity = 1
        self.symb = "not"

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.where(c_outs[0] == 0, 1, 0)


class Even(Node):
    def __init__(self):
        super(Even, self).__init__()
        self.arity = 1
        self.symb = "even"

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.where(c_outs[0] % 2 == 0, 1, 0)


class Odd(Node):
    def __init__(self):
        super(Odd, self).__init__()
        self.arity = 1
        self.symb = "odd"

    def _get_args_repr(self, args):
        return self._get_typical_repr(args, 'before')

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.where(c_outs[0] % 2 != 0, 1, 0)
