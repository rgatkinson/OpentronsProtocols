#
# util.py
#
import math
import threading
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

def first(iterable):
    for value in iterable:
        return value
    return None

def instance_count(predicate):
    """
    For debugging only: this is VERY slow
    """
    import gc
    count = 0
    for obj in gc.get_objects():
        if predicate(obj):
            count += 1
    return count


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


infinity = float('inf')

# make thread local storage
class TLS(threading.local):
    def __init__(self):
        # This gets called on every thread we're used on. Define default values
        from rgatkinson.configuration import config, TopConfigurationContext
        self.update_pose_tree_in_place = False
        self.config: TopConfigurationContext = config

tls = TLS()
