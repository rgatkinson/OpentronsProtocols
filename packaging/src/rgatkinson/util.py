#
# util.py
#
import math

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
