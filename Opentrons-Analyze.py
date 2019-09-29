#
# OpentronsAnalyze.py
#
# Runs opentrons.simulate, then outputs a summary
#
import argparse
import json
import logging
import string
import sys
from numbers import Number
from typing import List, Mapping, Any
from functools import wraps

import opentrons
import opentrons.simulate
from opentrons.legacy_api.containers import Well, Slot
from opentrons.legacy_api.containers.placeable import Placeable

########################################################################################################################
# Interval, adapted from pyinterval
########################################################################################################################

# region Floating point helper

# noinspection PyPep8Naming
class fpu(object):
    float = float
    _min = min
    _max = max
    infinity = float('inf')
    nan = infinity / infinity

    _fe_upward = None
    _fe_downward = None
    _fegetround = None
    _fesetround = None

    class NanException(ValueError):
        # Exception thrown when an unwanted nan is encountered.
        pass

    @classmethod
    def _init_libm(cls):  # pragma: nocover
        import platform
        processor = platform.processor()
        if processor == 'powerpc':
            cls._fe_upward, cls._fe_downward = 2, 3
        elif processor == 'sparc':
            cls._fe_upward, cls._fe_downward = 0x80000000, 0xC0000000
        else:
            cls._fe_upward, cls._fe_downward = 0x0800, 0x0400

        from ctypes import cdll
        from ctypes.util import find_library
        libm = cdll.LoadLibrary(find_library('m'))
        cls._fegetround, cls._fesetround = libm.fegetround, libm.fesetround

    @classmethod
    def _init_msvc(cls):  # pragma: nocover
        from ctypes import cdll
        controlfp = cdll.msvcrt._controlfp

        def local_fegetround():
            return controlfp(0, 0)

        def local_fesetround(flag):
            controlfp(flag, 0x300)

        cls._fe_upward, cls._fe_downward = 0x0200, 0x0100
        cls._fegetround = local_fegetround
        cls._fesetround = local_fesetround

    @classmethod
    def init(cls):  # pragma: nocover
        for f in cls._init_libm, cls._init_msvc:
            # noinspection PyBroadException
            try:
                f()
            except:
                pass
            else:
                break
        else:
            import warnings
            warnings.warn("Cannot determine FPU control primitives. The fpu module is not correctly initialized.", stacklevel=2)

    @staticmethod
    def isnan(x):
        return x != x

    @classmethod
    def down(cls, f):
        # Perform a computation with the FPU rounding downwards
        saved = cls._fegetround()
        try:
            cls._fesetround(cls._fe_downward)
            return f()
        finally:
            cls._fesetround(saved)

    @classmethod
    def up(cls, f):
        # Perform a computation with the FPU rounding upwards.
        saved = cls._fegetround()
        try:
            cls._fesetround(cls._fe_upward)
            return f()
        finally:
            cls._fesetround(saved)

    @classmethod
    def ensure_nonan(cls, x):
        if cls.isnan(x):
            raise cls.NanException
        return x

    @classmethod
    def min(cls, values):
        try:
            return cls._min(cls.ensure_nonan(x) for x in values)
        except cls.NanException:
            return cls.nan

    @classmethod
    def max(cls, values):
        try:
            return cls._max(cls.ensure_nonan(x) for x in values)
        except cls.NanException:
            return cls.nan

    @staticmethod
    def isinteger(n):
        return isinstance(n, int)

    @classmethod
    def power_rn(cls, x, n):
        # Raise x to the n-th power (with n positive integer), rounded to nearest.
        assert cls.isinteger(n) and n >= 0
        value = ()
        while n > 0:
            n, y = divmod(n, 2)
            value = (y, value)
        result = 1.0
        while value:
            y, value = value
            if y:
                result = result * result * x
            else:
                result = result * result
        return result

    @classmethod
    def power_ru(cls, x, n):
        # Raise x to the n-th power (with n positive integer), rounded toward +inf.
        if x >= 0:
            return cls.up(lambda: cls.power_rn(x, n))
        elif n % 2:
            return - cls.down(lambda: cls.power_rn(-x, n))
        else:
            return cls.up(lambda: cls.power_rn(-x, n))

    @classmethod
    def power_rd(cls, x, n):
        # Raise x to the n-th power (with n positive integer), rounded toward -inf.
        if x >= 0:
            return cls.down(lambda: cls.power_rn(x, n))
        elif n % 2:
            return - cls.up(lambda: cls.power_rn(-x, n))
        else:
            return cls.down(lambda: cls.power_rn(-x, n))


fpu.init()
# endregion

# region interval
def coercing(f):
    @wraps(f)
    def wrapper(self, other):
        try:
            return f(self, self.cast(other))
        except self.ScalarError:
            return NotImplemented
    return wrapper

def comp_by_comp(f):
    @wraps(f)
    def wrapper(self, other):
        try:
            return self._canonical(
                self.Component(*f(x, y))
                for x in self
                for y in self.cast(other))
        except self.ScalarError:
            return NotImplemented
    return wrapper

class Metaclass(type):
    def __getitem__(self, arg):
        return self(arg)


# noinspection PyPep8Naming
class interval(tuple, metaclass=Metaclass):

    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], cls):
            return args[0]

        def make_component(x, y=None):
            if y is None:
                return cls.cast(x)
            else:
                return cls.hull((cls.cast(x), cls.cast(y)))

        def process(x):
            try:
                return make_component(*x if hasattr(x, '__iter__') else (x,))
            except:
                raise cls.ComponentError("Invalid interval component: " + repr(x))

        return cls.union(process(x) for x in args)

    def __getnewargs__(self):
        return tuple(tuple(c) for c in self)

    @classmethod
    def new(cls, components):
        return tuple.__new__(cls, components)

    @classmethod
    def cast(cls, x):
        if isinstance(x, cls):
            return x
        try:
            y = fpu.float(x)
        except:
            raise cls.ScalarError("Invalid scalar: " + repr(x))
        if fpu.isinteger(x) and x != y:
            # Special case for an integer with more bits than in a float's mantissa
            if x > y:
                return cls.new((cls.Component(y, fpu.up(lambda: y + 1)),))
            else:
                return cls.new((cls.Component(fpu.down(lambda: y - 1), y),))
        return cls.new((cls.Component(y, y),))

    @classmethod
    def function(cls, f):
        @wraps(f)
        def wrapper(x):
            return cls._canonical(cls.Component(*t) for c in cls.cast(x) for t in f(c))
        return wrapper

    @classmethod
    def _canonical(cls, components):
        from operator import itemgetter
        components = [c for c in components if c.infimum <= c.supremum]
        components.sort(key=itemgetter(0))
        value = []
        for c in components:
            if not value or c.infimum > value[-1].supremum:
                value.append(c)
            elif c.supremum > value[-1].supremum:
                value[-1] = cls.Component(value[-1].infimum, c.supremum)
        return cls.new(value)

    @classmethod
    def union(cls, intervals):
        return cls._canonical(c for i in intervals for c in i)

    @classmethod
    def hull(cls, intervals):
        components = [c for i in intervals for c in i]
        return cls.new((cls.Component(fpu.min(c.infimum for c in components), fpu.max(c.supremum for c in components)),))

    @property
    def components(self):
        return (self.new((x,)) for x in self)

    @property
    def midpoint(self):
        return self.new(self.Component(x, x) for x in (sum(c) / 2 for c in self))

    @property
    def extrema(self):
        return self._canonical(self.Component(x, x) for c in self for x in c)

    def __repr__(self):
        return self.format_percent("%r")

    def __str__(self):
        return self.format("{0:s}")

    def format(self, format_spec, formatter=None):
        if formatter is None:
            formatter = string.Formatter
        return type(self).__name__ + '(' + ', '.join('[' + ', '.join(formatter.format(format_spec, x) for x in sorted(set(c))) + ']' for c in self) + ')'

    def format_percent(self, format_spec):
        return type(self).__name__ + '(' + ', '.join('[' + ', '.join(format_spec % x for x in sorted(set(c))) + ']' for c in self) + ')'

    @property
    def infimum(self):
        return fpu.min(c.infimum for c in self)

    @property
    def supremum(self):
        return fpu.max(c.supremum for c in self)

    def __pos__(self):
        return self

    def __neg__(self):
        return self.new(self.Component(-x.supremum, -x.infimum) for x in self)

    @comp_by_comp
    def __add__(x, y):
        return (fpu.down(lambda: x.infimum + y.infimum), fpu.up(lambda: x.supremum + y.supremum))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    @comp_by_comp
    def __mul__(x, y):
        return (
            fpu.down(lambda: fpu.min((x.infimum * y.infimum, x.infimum * y.supremum, x.supremum * y.infimum, x.supremum * y.supremum))),
            fpu.up  (lambda: fpu.max((x.infimum * y.infimum, x.infimum * y.supremum, x.supremum * y.infimum, x.supremum * y.supremum))))

    def __rmul__(self, other):
        return self * other

    @coercing
    def __div__(self, other):
        return self * other.inverse()

    __truediv__ = __div__

    @coercing
    def __rdiv__(self, other):
        return self.inverse() * other

    __rtruediv__ = __rdiv__

    def __pow__(self, n):
        if not fpu.isinteger(n):
            return NotImplemented
        if n < 0:
            return (self ** -n).inverse()
        if n % 2:
            def pow(c):
                return (fpu.power_rd(c.infimum, n), fpu.power_ru(c.supremum, n))
        else:
            def pow(c):
                if c.infimum > 0:
                    return (fpu.power_rd(c.infimum, n), fpu.power_ru(c.supremum, n))
                if c.supremum < 0:
                    return (fpu.power_rd(c.supremum, n), fpu.power_ru(c.infimum, n))
                else:
                    return (0.0, fpu.max(fpu.power_ru(x, n) for x in c))
        return self._canonical(self.Component(*pow(c)) for c in self)

    @comp_by_comp
    def __and__(x, y):
        return (fpu.max((x.infimum, y.infimum)), fpu.min((x.supremum, y.supremum)))

    def __rand__(self, other):
        return self & other

    @coercing
    def __or__(self, other):
        return self.union((self, other))

    def __ror__(self, other):
        return self | other

    @coercing
    def __contains__(self, other):
        return all(any(x.infimum <= y.infimum and y.supremum <= x.supremum for x in self) for y in other)

    def __abs__(self):
        return type(self)[0, fpu.infinity] & (self | (-self))

    class ComponentError(ValueError):
        pass

    class ScalarError(ValueError):
        pass

    class Component(tuple):
        def __new__(cls, inf, sup):
            if fpu.isnan(inf) or fpu.isnan(sup):
                return tuple.__new__(cls, (-fpu.infinity, +fpu.infinity))
            return tuple.__new__(cls, (inf, sup))

        @property
        def infimum(self):
            return self[0]

        @property
        def supremum(self):
            return self[1]

        @property
        def infimum_inv(self):
            return fpu.up(lambda: 1 / self.infimum)

        @property
        def supremum_inv(self):
            return fpu.down(lambda: 1 / self.supremum)

    def newton(self, f, p, maxiter=10000, tracer_cb=None):
        if tracer_cb is None:
            def tracer_cb(tag, interval):
                pass

        def step(x, i):
            return (x - f(x) / p(i)) & i

        def some(i):
            yield i.midpoint
            for x in i.extrema.components:
                yield x

        def branch(current):
            tracer_cb('branch', current)
            for n in range(maxiter):
                previous = current
                for anchor in some(current):
                    current = step(anchor, current)
                    if current != previous:
                        tracer_cb('step', current)
                        break
                else:
                    return current
                if not current:
                    return current
                if len(current) > 1:
                    return self.union(branch(c) for c in current.components)
            tracer_cb("abandon", current)
            return self.new(())

        return self.union(branch(c) for c in self.components)

    def inverse(c):
        if c.infimum <= 0 <= c.supremum:
            return ((-fpu.infinity, c.infimum_inv if c.infimum != 0 else -fpu.infinity),
                    (c.supremum_inv if c.supremum != 0 else +fpu.infinity, +fpu.infinity))
        else:
            return (c.supremum_inv, c.infimum_inv),


interval.inverse = interval.function(getattr(interval.inverse, '__func__', interval.inverse))
del coercing, comp_by_comp, Metaclass
# endregion

########################################################################################################################
# Mixtures
########################################################################################################################

class IndeterminateVolume():
    def __init__(self):
        pass  # NYI


class Aliquot(object):
    def __init__(self, well_monitor, volume):
        self.well_monitor = well_monitor
        self.volume = volume


class Mixture(object):
    def __init__(self, initial_aliquot=None):
        self.aliquots = dict()
        if initial_aliquot is not None:
            self.adjust_aliquot(initial_aliquot)

    def get_volume(self):
        result = 0.0
        for volume in self.aliquots.values():
            result += volume
        return result

    def is_empty(self):
        return self.get_volume() == 0

    def adjust_aliquot(self, aliquot):
        assert isinstance(aliquot, Aliquot)
        existing = self.aliquots.get(aliquot.well_monitor, 0)
        existing += aliquot.volume
        assert existing >= 0
        if existing == 0:
            self.aliquots.pop(aliquot.well_monitor, None)
        else:
            self.aliquots[aliquot.well_monitor] = existing

    def adjust_mixture(self, mixture):
        assert isinstance(mixture, Mixture)
        for well_monitor, volume in mixture.aliquots.items():
            self.adjust_aliquot(Aliquot(well_monitor, volume))

    def clear(self):
        self.aliquots = dict()

    def slice(self, volume):
        existing = self.get_volume()
        assert existing >= 0
        result = Mixture()
        ratio = float(volume) / float(existing)
        for well_monitor, volume in self.aliquots.items():
            result.adjust_aliquot(Aliquot(well_monitor, volume * ratio))
        return result

    def negated(self):
        result = Mixture()
        for well_monitor, volume in self.aliquots.items():
            result.adjust_aliquot(Aliquot(well_monitor, -volume))
        return result

########################################################################################################################
# Monitors
########################################################################################################################

class Monitor(object):

    def __init__(self, controller, location_path):
        self.controller = controller
        self.location_path = location_path
        self.target = None

    def set_target(self, target):  # idempotent
        assert self.target is None or self.target is target
        self.target = target

    def get_slot(self):
        placeable = self.target
        assert placeable is not None  #
        while not isinstance(placeable, Slot):
            placeable = placeable.parent
        return placeable

class WellVolume(object):
    def __init__(self, well):
        self.well = well
        self.initial_volume_known = False
        self.initial_volume = interval([0, fpu.infinity])
        self.cum_delta = 0
        self.min_delta = 0
        self.max_delta = 0

    def set_initial_volume(self, initial_volume):  # idempotent
        if self.initial_volume_known:
            assert self.initial_volume == initial_volume
        else:
            assert not self.initial_volume_known
            assert self.cum_delta == 0
            self.initial_volume_known = True
            self.initial_volume = initial_volume

    @property
    def current_volume(self):
        return self.initial_volume + self.cum_delta

    @property
    def min_volume(self):
        return self.initial_volume + self.min_delta

    @property
    def max_volume(self):
        return self.initial_volume + self.max_delta

    def aspirate(self, volume):
        assert volume >= 0
        if not self.initial_volume_known:
            self.set_initial_volume(interval([volume, fpu.infinity]))
        self._track_volume(-volume)

    def dispense(self, volume):
        assert volume >= 0
        if not self.initial_volume_known:
            self.set_initial_volume(0)
        self._track_volume(volume)

    def _track_volume(self, delta):
        self.cum_delta = self.cum_delta + delta
        self.min_delta = min(self.min_delta, self.cum_delta)
        self.max_delta = max(self.max_delta, self.cum_delta)


class WellMonitor(Monitor):
    def __init__(self, controller, location_path):
        super(WellMonitor, self).__init__(controller, location_path)
        self.volume = WellVolume(self)
        self.liquid_name = None
        self.mixture = Mixture()

    def aspirate(self, volume, mixture):
        self.volume.aspirate(volume)

    def dispense(self, volume, mixture):
        self.volume.dispense(volume)

    def set_liquid_name(self, name):  # idempotent
        assert self.liquid_name is None or self.liquid_name == name
        self.liquid_name = name

    def set_initial_volume(self, initial_volume):
        self.volume.set_initial_volume(initial_volume)

    def formatted(self):
        result = 'well "{0:s}"'.format(self.target.get_name())
        if self.liquid_name is not None:
            result += ' ("{0:s}")'.format(self.liquid_name)
        result += ':'
        result += Pretty().format(' lo={0:n} hi={1:n} cur={2:n}\n',
            self.volume.min_volume,
            self.volume.max_volume,
            self.volume.current_volume)
        return result

class AbstractContainerMonitor(Monitor):
    def __init__(self, controller, location_path):
        super(AbstractContainerMonitor, self).__init__(controller, location_path)


class WellContainerMonitor(AbstractContainerMonitor):
    def __init__(self, controller, location_path):
        super(WellContainerMonitor, self).__init__(controller, location_path)
        self.wells = dict()

    def add_well(self, well_monitor):  # idempotent
        name = well_monitor.target.get_name()
        if name in self.wells:  # avoid change on idempotency (might be iterating over self.wells)
            assert self.wells[name] is well_monitor
        else:
            self.wells[name] = well_monitor

    def formatted(self):
        result = ''
        result += 'container "%s" in "%s"\n' % (self.target.get_name(), self.get_slot().get_name())
        for well in self.target.wells():
            if self.controller.has_well(well):
                result += '   '
                result += self.controller.well_monitor(well).formatted()
        return result


class TipRackMonitor(AbstractContainerMonitor):
    def __init__(self, controller, location_path):
        super(TipRackMonitor, self).__init__(controller, location_path)
        self.tips_picked = dict()
        self.tips_dropped = dict()

    def pick_up_tip(self, well):
        self.tips_picked[well] = 1 + self.tips_picked.get(well, 0)

    def drop_tip(self, well):
        self.tips_dropped[well] = 1 + self.tips_dropped.get(well, 0)  # trash will have multiple

    def formatted(self):
        result = ''
        result += 'tip rack "%s" in "%s" picked %d tips\n' % (self.target.get_name(), self.get_slot().get_name(), len(self.tips_picked))
        return result


# Returns a unique name for the given location. Must track in protocols.
def get_location_path(location):
    return '/'.join(list(reversed([str(item)
                                   for item in location.get_trace(None)
                                   if str(item) is not None])))

class MonitorController(object):
    def __init__(self):
        self._monitors = dict()  # maps location path to monitor

    def note_liquid_name(self, liquid_name, location_path, initial_volume=None):
        well_monitor = self._monitor_from_location_path(WellMonitor, location_path)
        well_monitor.set_liquid_name(liquid_name)
        if initial_volume is not None:
            if isinstance(initial_volume, list):
                initial_volume = interval(*initial_volume)
            well_monitor.set_initial_volume(initial_volume)

    def well_monitor(self, well):
        well_monitor = self._monitor_from_location_path(WellMonitor, get_location_path(well))
        well_monitor.set_target(well)

        well_container_monitor = self._monitor_from_location_path(WellContainerMonitor, get_location_path(well.parent))
        well_container_monitor.set_target(well.parent)
        well_container_monitor.add_well(well_monitor)

        return well_monitor

    def has_well(self, well):
        well_monitor = self._monitors.get(get_location_path(well), None)
        return well_monitor is not None and well_monitor.target is not None

    def tip_rack_monitor(self, tip_rack):
        tip_rack_monitor = self._monitor_from_location_path(TipRackMonitor, get_location_path(tip_rack))
        tip_rack_monitor.set_target(tip_rack)
        return tip_rack_monitor

    def formatted(self):
        result = ''

        monitors = self._all_monitors(TipRackMonitor)
        slot_numbers = list(monitors.keys())
        slot_numbers.sort()
        for slot_num in slot_numbers:
            if slot_num == 12: continue  # ignore trash
            for monitor in monitors[slot_num]:
                result += monitor.formatted()

        monitors = self._all_monitors(WellContainerMonitor)
        slot_numbers = list(monitors.keys())
        slot_numbers.sort()
        for slot_num in slot_numbers:
            for monitor in monitors[slot_num]:
                result += monitor.formatted()

        return result

    def _all_monitors(self, monitor_type):  # -> map from slot number to list of monitor
        result = dict()
        for monitor in set(monitor for monitor in self._monitors.values() if monitor.target is not None):
            if isinstance(monitor, monitor_type):
                slot_num = int(monitor.get_slot().get_name())
                result[slot_num] = result.get(slot_num, list()) + [monitor]
        return result

    def _monitor_from_location_path(self, monitor_type, location_path):
        if location_path not in self._monitors:
            monitor = monitor_type(self, location_path)
            self._monitors[location_path] = monitor
        return self._monitors[location_path]


########################################################################################################################
# Utility
########################################################################################################################

def log(msg: str):
    print("*********** %s ***********" % msg)

def info(msg: str, prefix="***********", suffix=' ***********'):
    print("%s%s%s%s" % (prefix, '' if len(prefix) == 0 else ' ', msg, suffix))

def warn(msg: str, prefix="***********", suffix=' ***********'):
    print("%s%sWARNING: %s%s" % (prefix, '' if len(prefix) == 0 else ' ', msg, suffix))

class Pretty(string.Formatter):
    def format_field(self, value, spec):
        if spec.endswith('n'):  # 'n' for number
            precision = 2
            if spec.startswith('.', 0, -1):
                precision = int(spec[1:-1])
            if isinstance(value, Number):
                factor = 1
                for i in range(precision):
                    if value * factor == int(value * factor):
                        precision = i
                        break
                    factor *= 10
                return "{:.{}f}".format(value, precision)
            elif hasattr(value, 'format'):
                return value.format(format_spec="{0:%s}" % spec, formatter=self)
            else:
                return str(value)
        return super().format_field(value, spec)


########################################################################################################################
# Analyzing
########################################################################################################################

def analyzeRunLog(run_log):

    controller = MonitorController()
    pipette_contents = Mixture()

    # locations are either placeables or (placable, vector) pairs
    def placeable_from_location(location):
        if isinstance(location, Placeable):
            return location
        else:
            return location[0]

    for log_item in run_log:
        # log_item is a dict with string keys:
        #       level
        #       payload
        #       logs
        payload = log_item['payload']

        # payload is a dict with string keys:
        #       instrument
        #       location
        #       volume
        #       repetitions
        #       text
        #       rate
        text = payload['text']
        lower_words = list(map(lambda word: word.lower(), text.split()))
        if len(lower_words) == 0: continue  # paranoia
        selector = lower_words[0]
        if len(payload) > 1:
            # a non-comment
            if selector == 'aspirating' or selector == 'dispensing':
                well = placeable_from_location(payload['location'])
                volume = payload['volume']
                monitor = controller.well_monitor(well)
                if selector == 'aspirating':
                    monitor.aspirate(volume, pipette_contents)
                else:
                    monitor.dispense(volume, pipette_contents)
            elif selector == 'picking' or selector == 'dropping':
                well = placeable_from_location(payload['location'])
                rack = well.parent
                monitor = controller.tip_rack_monitor(rack)
                if selector == 'picking':
                    monitor.pick_up_tip(well)
                else:
                    monitor.drop_tip(well)
                pipette_contents.clear()
            elif selector == 'mixing' \
                    or selector == 'transferring' \
                    or selector == 'distributing' \
                    or selector == 'blowing' \
                    or selector == 'touching' \
                    or selector == 'homing'\
                    or selector == 'setting' \
                    or selector == 'thermocycler' \
                    or selector == 'delaying' \
                    or selector == 'consolidating':
                pass
            else:
                warn('unexpected run item: %s' % text)
        else:
            # a comment
            if selector == 'liquid:':
                # Remainder after selector is json dictionary
                serialized = text[len(selector):]  # will include initial white space, but that's ok
                serialized = serialized.replace("}}", "}").replace("{{", "{")
                d = json.loads(serialized)
                controller.note_liquid_name(d['name'], d['location'], initial_volume=d.get('initial_volume', None))
            elif selector == 'air' \
                    or selector == 'returning' \
                    or selector == 'engaging' \
                    or selector == 'disengaging' \
                    or selector == 'calibrating' \
                    or selector == 'deactivating' \
                    or selector == 'waiting' \
                    or selector == 'setting' \
                    or selector == 'opening' \
                    or selector == 'closing' \
                    or selector == 'pausing' \
                    or selector == 'resuming':
                pass
            else:
                pass  # nothing to process

    return controller


def main():
    parser = argparse.ArgumentParser(prog='opentrons-analyze', description=__doc__)
    parser.add_argument(
        'protocol', metavar='PROTOCOL_FILE',
        type=argparse.FileType('r'),
        help='The protocol file to simulate (specify - to read from stdin).')
    parser.add_argument(
        '-v', '--version', action='version',
        version=f'%(prog)s {opentrons.__version__}',
        help='Print the opentrons package version and exit')
    parser.add_argument(
        '-l', '--log-level', action='store',
        help='Log level for the opentrons stack. Anything below warning can be chatty',
        choices=['error', 'warning', 'info', 'debug'],
        default='warning'
    )
    args = parser.parse_args()

    run_log = simulate(args.protocol, log_level=args.log_level)
    analysis = analyzeRunLog(run_log)
    print(opentrons.simulate.format_runlog(run_log))
    print("\n")
    print(analysis.formatted())
    return 0


# Cloned from opentrons.simulate.simulate in order to replace 'exec(proto,{})' with something more debugger-friendly
def simulate(protocol_file,
             propagate_logs=False,
             log_level='warning') -> List[Mapping[str, Any]]:
    stack_logger = logging.getLogger('opentrons')
    stack_logger.propagate = propagate_logs

    contents = protocol_file.read()
    protocol_file_name = None
    if protocol_file is not sys.stdin:
        protocol_file_name = protocol_file.name

    if opentrons.config.feature_flags.use_protocol_api_v2():
        try:
            execute_args = {'protocol_json': json.loads(contents)}
        except json.JSONDecodeError:
            execute_args = {'protocol_code': contents}
        context = opentrons.protocol_api.contexts.ProtocolContext()
        context.home()
        scraper = opentrons.simulate.CommandScraper(stack_logger, log_level, context.broker)
        execute_args.update({'simulate': True, 'context': context})
        opentrons.protocol_api.execute.run_protocol(**execute_args)
    else:
        try:
            proto = json.loads(contents)
        except json.JSONDecodeError:
            proto = contents
        opentrons.robot.disconnect()
        scraper = opentrons.simulate.CommandScraper(stack_logger, log_level, opentrons.robot.broker)
        if isinstance(proto, dict):
            opentrons.protocols.execute_protocol(proto)
        else:
            if protocol_file_name is not None:
                # https://stackoverflow.com/questions/436198/what-is-an-alternative-to-execfile-in-python-3
                code = compile(proto, protocol_file_name, 'exec')
                exec(code, {})
            else:
                exec(proto, {})
    return scraper.commands


if __name__ == '__main__':
    rc = main()
    sys.exit(rc)
