#
# collection_util.py
#

from opentrons.legacy_api.containers import unpack_location


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


def well_vector(location):
    return unpack_location(location)