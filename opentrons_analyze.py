#
# OpentronsAnalyze.py
#
# Runs opentrons.simulate, then outputs a summary
#
import argparse
import json
import string
import sys
import warnings
from enum import Enum
from numbers import Number
from numpy import isclose
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
class Fpu(object):
    def __init__(self):
        self.float = float
        self._min = min
        self._max = max
        self.infinity = float('inf')
        self.nan = self.infinity / self.infinity
        self._fe_upward = None
        self._fe_downward = None
        self._fegetround = None
        self._fesetround = None
        for f in self._init_libm, self._init_msvc:
            # noinspection PyBroadException
            try:
                f()
            except:
                pass
            else:
                break
        else:
            warnings.warn("Cannot determine FPU control primitives. The fpu module is not correctly initialized.", stacklevel=2)

    def _init_libm(self):  # pragma: nocover
        import platform
        processor = platform.processor()
        if processor == 'powerpc':
            self._fe_upward, self._fe_downward = 2, 3
        elif processor == 'sparc':
            self._fe_upward, self._fe_downward = 0x80000000, 0xC0000000
        else:
            self._fe_upward, self._fe_downward = 0x0800, 0x0400
        from ctypes import cdll
        from ctypes.util import find_library
        libm = cdll.LoadLibrary(find_library('m'))
        self._fegetround, self._fesetround = libm.fegetround, libm.fesetround

    def _init_msvc(self):  # pragma: nocover
        from ctypes import cdll
        controlfp = cdll.msvcrt._controlfp
        self._fe_upward, self._fe_downward = 0x0200, 0x0100
        self._fegetround = lambda: controlfp(0, 0)
        self._fesetround = lambda flag: controlfp(flag, 0x300)

    class NanException(ValueError):
        # Exception thrown when an unwanted nan is encountered.
        pass

    def down(self, f):
        # Perform a computation with the FPU rounding downwards
        saved = self._fegetround()
        try:
            self._fesetround(self._fe_downward)
            return f()
        finally:
            self._fesetround(saved)

    def up(self, f):
        # Perform a computation with the FPU rounding upwards.
        saved = self._fegetround()
        try:
            self._fesetround(self._fe_upward)
            return f()
        finally:
            self._fesetround(saved)

    def ensure_nonan(self, x):
        if is_nan(x):
            raise self.NanException
        return x

    def min(self, values):
        try:
            return self._min(self.ensure_nonan(x) for x in values)
        except self.NanException:
            return self.nan

    def max(self, values):
        try:
            return self._max(self.ensure_nonan(x) for x in values)
        except self.NanException:
            return self.nan

    def power_rn(self, x, n):
        # Raise x to the n-th power (with n positive integer), rounded to nearest.
        assert is_integer(n) and n >= 0
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

    def power_ru(self, x, n):
        # Raise x to the n-th power (with n positive integer), rounded toward +inf.
        if x >= 0:
            return self.up(lambda: self.power_rn(x, n))
        elif n % 2:
            return - self.down(lambda: self.power_rn(-x, n))
        else:
            return self.up(lambda: self.power_rn(-x, n))

    def power_rd(self, x, n):
        # Raise x to the n-th power (with n positive integer), rounded toward -inf.
        if x >= 0:
            return self.down(lambda: self.power_rn(x, n))
        elif n % 2:
            return - self.up(lambda: self.power_rn(-x, n))
        else:
            return self.down(lambda: self.power_rn(-x, n))


fpu = Fpu()

def is_integer(n):
    return isinstance(n, int)

def is_scalar(x):
    return isinstance(x, Number)

def is_interval(x):
    return isinstance(x, interval)

def is_nan(x):
    return x != x

def is_infinite(x):
    return is_scalar(x) and (x == fpu.infinity or x == -fpu.infinity)

def is_finite(x):
    return is_scalar(x) and not is_nan(x) and not is_infinite(x)

def supremum(x):
    if is_interval(x):
        return x.supremum
    else:
        return x

def infimum(x):
    if is_interval(x):
        return x.infimum
    else:
        return x

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
        if is_integer(x) and x != y:
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
        if not is_integer(n):
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
            if is_nan(inf) or is_nan(sup):
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

class Liquid:
    def __init__(self, name):
        self.name = name
        self.concentration = Concentration('dc')

    def __str__(self) -> str:
        if self.concentration.flavor == Concentration.Flavor.DontCare:
            return f'Liquid({self.name})'
        else:
            return f'Liquid([{self.name}]={self.concentration})'


class Concentration(object):

    class Flavor(Enum):
        Molar = 0
        X = 1
        DontCare = 2

    def __init__(self, value, unit=None, flavor=None):
        if flavor is not None:
            if flavor == Concentration.Flavor.Molar:
                unit = 'M'
            elif flavor == Concentration.Flavor.X:
                unit = 'x'
            elif flavor == Concentration.Flavor.DontCare:
                unit = 'dc'
        if unit is not None:
            value = str(value) + str(unit)
        else:
            value = str(value)
        self.flavor = Concentration.Flavor.Molar
        units = [('mM', 0.001), ('uM', 0.000001), ('nM', 1e-9), ('M', 1)]
        for unit, factor in units:
            if value.endswith(unit):
                quantity = value[0:-len(unit)]
                self.value = float(quantity) * factor
                return
        if value.lower().endswith('x'):
            quantity = value[0:-1]
            try:
                self.value = float(quantity)
            except ValueError:
                self.value = 0
            self.flavor = Concentration.Flavor.X
            return
        if value.lower().endswith('dc'):
            self.value = 0
            self.flavor = Concentration.Flavor.DontCare
            return
        # default is micro-molar
        factor = 1e-6
        self.value = float(value) * factor

    def __mul__(self, scale):
        return Concentration(self.value * scale, flavor=self.flavor)

    def __rmul__(self, scale):
        return self * scale

    def __str__(self) -> str:
        def test(scale):
            return int(self.value * scale) != 0

        def emit(scale, unit):
            return Pretty().format('{0:.3n}{1}', self.value * scale, unit)

        if self.flavor == Concentration.Flavor.Molar:
            if self.value == 0:
                return emit(1, 'M')
            elif test(1):
                return emit(1, 'M')
            elif test(1e3):
                return emit(1e3, 'mM')
            elif test(1e6):
                return emit(1e6, 'uM')
            else:
                return emit(1e9, 'nM')
        elif self.flavor == Concentration.Flavor.X:
            return Pretty().format('{0:.3n}x', self.value)
        else:
            return 'DC'


class Mixture(object):
    def __init__(self, liquid=None, volume=0):
        self.liquids = dict()  # map from liquid to volume
        if liquid is not None:
            self.set_initial_liquid(liquid=liquid, volume=volume)

    def set_initial_liquid(self, liquid, volume):
        assert len(self.liquids) == 0
        if liquid is not None:
            self._adjust_liquid(liquid, volume)

    def __str__(self) -> str:
        if self.is_empty:
            return '{}'
        else:
            result = '{ '
            is_first = True
            total_volume = self.volume
            for liquid, volume in self.liquids.items():
                if not is_first:
                    result += ', '
                if is_scalar(total_volume) and liquid.concentration.flavor != Concentration.Flavor.DontCare:
                    dilution_factor = volume / total_volume
                    concentration = liquid.concentration * dilution_factor
                    result += Pretty().format('{0}:{1:n}={2}', liquid.name, volume, concentration)
                else:
                    result += Pretty().format('{0}:{1:n}', liquid.name, volume)
                is_first = False
            result += ' }'
        return result

    @property
    def volume(self):
        result = 0.0
        for volume in self.liquids.values():
            result += volume
        return result

    @property
    def is_empty(self):
        return supremum(self.volume) <= 0

    @property
    def is_homogeneous(self):
        return len(self.liquids) <= 1

    def _adjust_liquid(self, liquid, volume):
        existing = self.liquids.get(liquid, 0)
        existing += volume
        if supremum(existing) <= 0:
            self.liquids.pop(liquid, None)
        else:
            self.liquids[liquid] = existing

    def to_pipette(self, volume, pipette_contents):
        removed = self.remove_volume(volume)
        pipette_contents.mixture.add_mixture(removed)

    def from_pipette(self, volume, pipette_contents):
        removed = pipette_contents.mixture.remove_volume(volume)
        self.add_mixture(removed)

    def add_mixture(self, them):
        assert self is not them
        for liquid, volume in them.liquids.items():
            self._adjust_liquid(liquid, volume)

    def remove_volume(self, removal_volume):
        assert is_scalar(removal_volume)
        if self.is_homogeneous:
            # If the liquid is homogeneous, we can remove from non-scalar volumes
            liquid = first(self.liquids.keys())
            self._adjust_liquid(liquid, -removal_volume)
            return Mixture(liquid, removal_volume)
        else:
            current_volume = self.volume
            assert is_scalar(current_volume)
            removal_fraction = removal_volume / current_volume
            result = Mixture()
            new_liquids = Mixture()  # avoid changing while iterating
            for liquid, volume in self.liquids.items():
                result._adjust_liquid(liquid, volume * removal_fraction)
                new_liquids._adjust_liquid(liquid, volume * (1.0 - removal_fraction))
            self.liquids = new_liquids.liquids
            return result


class PipetteContents(object):
    def __init__(self):
        self.mixture = Mixture()

    def pick_up_tip(self):
        self.clear()

    def drop_tip(self):
        self.clear()

    def clear(self):
        self.mixture = Mixture()

########################################################################################################################
# Monitors
########################################################################################################################

class PlaceableMonitor(object):

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
    def __init__(self):
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


class WellMonitor(PlaceableMonitor):
    def __init__(self, controller, location_path):
        super(WellMonitor, self).__init__(controller, location_path)
        self.volume = WellVolume()
        self.liquid = Liquid(location_path)  # unique to this well unless we're told a better name later
        self.liquid_known = False
        self.mixture = Mixture()

    def aspirate(self, volume, pipette_contents: PipetteContents):
        self.volume.aspirate(volume)
        self.mixture.to_pipette(volume, pipette_contents)

    def dispense(self, volume, pipette_contents: PipetteContents):
        self.volume.dispense(volume)
        self.mixture.from_pipette(volume, pipette_contents)

    def set_liquid(self, liquid):  # idempotent
        assert not self.liquid_known or self.liquid is liquid
        self.liquid = liquid
        self.liquid_known = True

    def set_initial_volume(self, initial_volume):
        self.volume.set_initial_volume(initial_volume)
        self.mixture.set_initial_liquid(self.liquid, initial_volume)

    def formatted(self):
        result = 'well "{0:s}"'.format(self.target.get_name())
        if not getattr(self.target, 'has_labelled_well_name', False):
            if self.liquid is not None:
                result += ' ("{0:s}")'.format(self.liquid)
        result += ':'
        result += Pretty().format(' lo={0:n} hi={1:n} cur={2:n} mix={3:s}\n',
            self.volume.min_volume,
            self.volume.max_volume,
            self.volume.current_volume,
            self.mixture.__str__())
        return result

# region Containers
class AbstractContainerMonitor(PlaceableMonitor):
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
# endregion

# Returns a unique name for the given location. Must track in protocols.
def get_location_path(location):
    return '/'.join(list(reversed([str(item)
                                   for item in location.get_trace(None)
                                   if str(item) is not None])))

class MonitorController(object):
    def __init__(self):
        self._monitors = dict()  # maps location path to monitor
        self._liquids = dict()

    def get_liquid(self, liquid_name):
        try:
            return self._liquids[liquid_name]
        except KeyError:
            self._liquids[liquid_name] = Liquid(liquid_name)
            return self._liquids[liquid_name]

    def note_liquid_name(self, liquid_name, location_path, initial_volume=None, concentration=None):
        well_monitor = self._monitor_from_location_path(WellMonitor, location_path)
        liquid = self.get_liquid(liquid_name)
        if concentration is not None:
            concentration = Concentration(concentration)
            liquid.concentration = concentration
        well_monitor.set_liquid(liquid)
        if initial_volume is not None:
            if isinstance(initial_volume, list):  # work around json parsing deficiency
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

def first(iterable):
    for value in iterable:
        return value
    return None

class Pretty(string.Formatter):
    def format_field(self, value, spec):
        if spec.endswith('n'):  # 'n' for number
            precision = 2
            if spec.startswith('.', 0, -1):
                precision = int(spec[1:-1])
            if isinstance(value, Number) and is_finite(value):
                factor = 1
                for i in range(precision):
                    if isclose(value * factor, int(value * factor)):
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
    pipette_contents = PipetteContents()

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
                well_monitor = controller.well_monitor(well)
                if selector == 'aspirating':
                    well_monitor.aspirate(volume, pipette_contents)
                else:
                    well_monitor.dispense(volume, pipette_contents)
            elif selector == 'picking' or selector == 'dropping':
                well = placeable_from_location(payload['location'])
                rack = well.parent
                tip_rack_monitor = controller.tip_rack_monitor(rack)
                if selector == 'picking':
                    tip_rack_monitor.pick_up_tip(well)
                    pipette_contents.pick_up_tip()
                else:
                    tip_rack_monitor.drop_tip(well)
                    pipette_contents.drop_tip()
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
                controller.note_liquid_name(d['name'], d['location'], initial_volume=d.get('initial_volume', None), concentration=d.get('concentration', None))
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


def main() -> int:
    parser = argparse.ArgumentParser(prog='opentrons-analyze', description='Analyze an OT-2 protocol')
    parser = opentrons.simulate.get_arguments(parser)
    parser.add_argument(
        '-v', '--version', action='version',
        version=f'%(prog)s {opentrons.__version__}',
        help='Print the opentrons package version and exit')
    args = parser.parse_args()

    run_log = opentrons.simulate.simulate(args.protocol, log_level=args.log_level)
    analysis = analyzeRunLog(run_log)
    print(opentrons.simulate.format_runlog(run_log))
    print("\n")
    print(analysis.formatted())
    return 0


if __name__ == '__main__':
    rc = main()
    sys.exit(rc)
