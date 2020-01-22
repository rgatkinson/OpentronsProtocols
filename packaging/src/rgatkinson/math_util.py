#
# math_util.py
#

import math
from numbers import Number


def is_indexable(value):
    return hasattr(type(value), '__getitem__')


def square(value):
    return value * value


def cube(value):
    return value * value * value


def sqrt(value):
    return math.sqrt(value)


def cube_root(value):
    return pow(value, 1.0/3.0)


def cubeRoot(value):
    return pow(value, 1.0/3.0)


def zeroify(value, digits=2):  # clamps small values to zero, leaves others alone
    rounded = round(value, digits)
    return rounded if rounded == 0 else value


def is_integer(n):
    return n.__class__ is int


def is_scalar(x):
    return float is x.__class__ or int is x.__class__ or isinstance(x, Number)


def is_nan(x):
    return x != x


def is_infinite_scalar(x):
    return is_scalar(x) and (x == infinity or x == -infinity)


def is_finite_scalar(x):
    return is_scalar(x) and not is_nan(x) and not is_infinite_scalar(x)


def is_close(x, y, atol=1e-08, rtol=1e-05):  # after numpy.isclose, but faster, and only for scalars
    if x == y:
        return True
    return abs(x-y) <= atol + rtol * abs(y)

def float_range(start, stop, count):
    cur = start
    step = (stop - start) / (count - 1)
    for i in range(count):
        yield cur
        cur = cur + step

def mean(a, b):
    return (a + b) / 2

infinity = float('inf')
