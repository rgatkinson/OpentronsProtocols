#
# atkinson.opentrons/__init__.py
#

# region Enhancements

import collections
import enum
import json
import math
import random
import string
import warnings
from abc import abstractmethod
from enum import Enum
from functools import wraps
from numbers import Number
from typing import List

import opentrons
from opentrons import labware, instruments, robot, modules, types
from opentrons.commands import stringify_location, make_command, command_types
from opentrons.helpers import helpers
from opentrons.legacy_api.instruments import Pipette
from opentrons.legacy_api.instruments.pipette import SHAKE_OFF_TIPS_DROP_DISTANCE, SHAKE_OFF_TIPS_SPEED
from opentrons.legacy_api.containers.placeable import unpack_location, Well, WellSeries, Placeable
from opentrons.trackers import pose_tracker
from opentrons.util.vector import Vector

########################################################################################################################
# Enhancements Configuration. Evolving; could use improvement
########################################################################################################################

class ConfigurationContext(object):
    pass

class TopConfigurationContext(ConfigurationContext):

    def well_geometry(self, well):
        assert is_well(well)
        try:
            result = well.geometry
            assert result.config is self
            return result
        except AttributeError:
            return self.set_well_geometry(well, UnknownWellGeometry)

    def well_volume(self, well):
        assert is_well(well)
        try:
            result = well.contents
            assert result.config is self
            return result
        except AttributeError:
            well.contents = WellVolume(well, self)
            return well.contents

    def set_well_geometry(self, well, geometry_class):
        result = geometry_class(well, self)
        assert well.geometry is result
        return result


config = TopConfigurationContext()
config.enable_enhancements = True
config.trash_control = True
config.blow_out_rate_factor = 3.0
config.min_aspirate_factor_hack = 15.0
config.allow_blow_elision_default = True
config.allow_overspill_default = True

config.aspirate = ConfigurationContext()
config.aspirate.bottom_clearance = 1.0  # see Pipette._position_for_aspirate
config.aspirate.top_clearance = -3.5
config.aspirate.extra_top_clearance_name = 'extra_aspirate_top_clearance'
config.aspirate.pre_wet = ConfigurationContext()
config.aspirate.pre_wet.default = True
config.aspirate.pre_wet.count = 2  # save some time vs 3
config.aspirate.pre_wet.max_volume_fraction = 1  # https://github.com/Opentrons/opentrons/issues/2901 would pre-wet only 2/3, but why not everything?
config.aspirate.pre_wet.rate_func = lambda aspirate_rate: 1  # could instead just use the aspirate
config.aspirate.pause = ConfigurationContext()
config.aspirate.pause.ms_default = 750

config.dispense = ConfigurationContext()
config.dispense.bottom_clearance = 0.5  # see Pipette._position_for_dispense
config.dispense.top_clearance = -2.0
config.dispense.extra_top_clearance_name = 'extra_dispense_top_clearance'
config.dispense.full_dispense = ConfigurationContext()
config.dispense.full_dispense.default = True

config.layered_mix = ConfigurationContext()
config.layered_mix.top_clearance = -1.5  # close, so we mix top layers too
config.layered_mix.aspirate_bottom_clearance = 1.0
config.layered_mix.aspirate_rate_factor = 4.0
config.layered_mix.dispense_rate_factor = 4.0
config.layered_mix.incr = 1.0
config.layered_mix.count = None  # so we default to using incr, not count
config.layered_mix.min_incr = 0.5
config.layered_mix.count_per_incr = 2
config.layered_mix.ms_pause = 0
config.layered_mix.keep_last_tip = False
config.layered_mix.initial_turnover = None
config.layered_mix.max_tip_cycles = None
config.layered_mix.max_tip_cycles_large = None
config.layered_mix.enable_radial_randomness = True
config.layered_mix.radial_clearance_tolerance = 0.5

config.wells = ConfigurationContext()
config.wells.radial_clearance_tolerance = 0.5

# region Other Enhancements Stuff

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
        for f in self._init_libm, self._init_msvc, self._init_degenerate:
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

    def _init_degenerate(self):
        # a do-nothing fallback for the case where we just can't control fpu by other means
        self._fe_upward, self._fe_downward = 0, 0
        self._fegetround = lambda: 0  # nop
        self._fesetround = lambda flag: 0  # nop
        warnings.warn("Using degenerate FPU control", stacklevel=2)

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

def is_indexable(value):
    return hasattr(type(value), '__getitem__')

def is_integer(n):
    return n.__class__ is int

def is_scalar(x):
    return float is x.__class__ or int is x.__class__ or isinstance(x, Number)

def is_interval(x):
    return x.__class__ is interval

def is_nan(x):
    return x != x

def is_infinite_scalar(x):
    return is_scalar(x) and (x == fpu.infinity or x == -fpu.infinity)

def is_finite_scalar(x):
    return is_scalar(x) and not is_nan(x) and not is_infinite_scalar(x)

def is_close(x, y, atol=1e-08, rtol=1e-05):  # after numpy.isclose, but faster, and only for scalars
    if x == y:
        return True
    return abs(x-y) <= atol + rtol * abs(y)

def is_well(location):
    return isinstance(location, Well)

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

class Metaclass(type):  # See https://docs.python.org/3/reference/datamodel.html, Section 3.3.5. Emulating generic types
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
        def test(v, scale):
            return int(v * scale) != 0

        def emit(scale, unit):
            v = self.value * scale
            if v >= 100:
                return pretty.format('{0:.0n}{1}', v, unit)
            if v >= 10:
                return pretty.format('{0:.1n}{1}', v, unit)
            if v >= 1:
                return pretty.format('{0:.2n}{1}', v, unit)
            return pretty.format('{0:.3n}{1}', v, unit)

        if self.flavor == Concentration.Flavor.Molar:
            if self.value == 0:
                return emit(1, 'M')
            elif test(self.value, 1):
                return emit(1, 'M')
            elif test(self.value, 1e3):
                return emit(1e3, 'mM')
            elif test(self.value, 1e6):
                return emit(1e6, 'uM')
            else:
                return emit(1e9, 'nM')
        elif self.flavor == Concentration.Flavor.X:
            return pretty.format('{0:.3n}x', self.value)
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
                    result += pretty.format('{0}:{1:n}={2}', liquid.name, volume, concentration)
                else:
                    result += pretty.format('{0}:{1:n}', liquid.name, volume)
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


# Must keep in sync with Opentrons-Analyze controller.note_liquid_name
def note_liquid(location, name=None, initial_volume=None, min_volume=None, concentration=None, local_config=None):
    if local_config is None:
        local_config = config
    well, __ = unpack_location(location)
    assert is_well(well)
    if name is None:
        name = well.label
    else:
        well.label = name
    d = {'name': name, 'location': get_location_path(well)}
    if initial_volume is None and min_volume is not None:
        initial_volume = interval([min_volume, local_config.well_geometry(well).well_capacity])
    if initial_volume is not None:
        d['initial_volume'] = initial_volume
        local_config.well_volume(well).set_initial_volume(initial_volume)
    if concentration is not None:
        d['concentration'] = str(Concentration(concentration))
    serialized = json.dumps(d).replace("{", "{{").replace("}", "}}")  # runtime calls comment.format(...) on our comment; avoid issues therewith
    robot.comment('Liquid: %s' % serialized)

########################################################################################################################
# Well Volume
########################################################################################################################

# region Well Volume

class WellVolume(object):

    #-------------------------------------------------------------------------------------------------------------------
    # Construction
    #-------------------------------------------------------------------------------------------------------------------

    def __init__(self, well, config):
        self.well = well
        self.config = config
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

    #-------------------------------------------------------------------------------------------------------------------
    # Accessing
    #-------------------------------------------------------------------------------------------------------------------

    @property
    def current_volume(self):  # may be interval
        return self.initial_volume + self.cum_delta

    @property
    def current_volume_min(self):  # (scalar) minimum known to be currently occupied
        vol = self.current_volume
        if is_interval(vol):
            return vol.infimum
        else:
            return vol

    @property
    def available_volume_min(self):
        return max(0, self.current_volume_min - self._min_aspiratable_volume)

    @property
    def min_volume(self):  # minimum historically seen
        return self.initial_volume + self.min_delta

    @property
    def max_volume(self):  # maximum historically seen
        return self.initial_volume + self.max_delta

    @property
    def _min_aspiratable_volume(self):
        if self.well is None:
            return 0
        else:
            return self.config.well_geometry(self.well).min_aspiratable_volume

    #-------------------------------------------------------------------------------------------------------------------
    # Actions
    #-------------------------------------------------------------------------------------------------------------------

    def aspirate(self, volume):
        assert volume >= 0
        if not self.initial_volume_known:
            self.set_initial_volume(interval([volume,
                                              fpu.infinity if self.well is None else self.config.well_geometry(self.well).well_capacity]))
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

# endregion

########################################################################################################################
# Well Geometry
########################################################################################################################

# region Well Geometry

class WellGeometry(object):

    #-------------------------------------------------------------------------------------------------------------------
    # Construction
    #-------------------------------------------------------------------------------------------------------------------

    def __init__(self, well, config):
        self.__well = None
        self.well = well
        self.config = config

    @property
    def well(self):
        return self.__well

    @well.setter
    def well(self, value):
        if self.__well is not None:
            self.__well.geometry = None
        self.__well = value
        if self.__well is not None:
            self.__well.geometry = self

    #-------------------------------------------------------------------------------------------------------------------
    # Accessing
    #-------------------------------------------------------------------------------------------------------------------

    @property
    def well_capacity(self):  # default to what Opentrons gave us
        result = self.well.max_volume() if self.well is not None else None
        if result is None:
            result = fpu.infinity
        return result

    @property
    def well_depth(self):  # default to what Opentrons gave us
        return self.well.z_size() if self.well is not None else fpu.infinity

    @property
    def outside_height(self):  # outside height of the tube
        return 0

    @property
    def rim_lip_height(self):  # when hanging in a rack, this is how much the tube sits about the reference plane of the rack
        return 0

    def height_above_reference_plane(self, hangable_tube_height, rack):
        return max(0, self.outside_height - hangable_tube_height, self.rim_lip_height);

    @property
    def well_diameter_at_top(self):
        return self.radius_from_depth(self.well_depth) * 2  # a generic impl; subclasses can optimize

    @property
    def min_aspiratable_volume(self):  # minimum volume we can aspirate from (i.e.: we leave at least this much behind when aspirating)
        return 0

    @property
    def radial_clearance_tolerance(self):
        return self.config.wells.radial_clearance_tolerance

    @abstractmethod
    def depth_from_volume(self, volume):  # best calc'n of depth from the given volume. may be an interval
        pass

    @abstractmethod
    def volume_from_depth(self, depth):
        pass

    @abstractmethod
    def radius_from_depth(self, depth):
        pass

    #-------------------------------------------------------------------------------------------------------------------
    # Calculations
    #-------------------------------------------------------------------------------------------------------------------

    def depth_from_volume_min(self, volume):  # lowest possible depth for the given volume
        vol = self.depth_from_volume(volume)
        if is_interval(vol):
            return vol.infimum
        else:
            return vol


class UnknownWellGeometry(WellGeometry):
    def __init__(self, well, config):
        super().__init__(well, config)

    def depth_from_volume(self, vol):
        return interval([0, self.well_depth])

    def volume_from_depth(self, depth):
        return interval([0, self.well_capacity])

    def radius_from_depth(self, depth):
        return self.well.properties['diameter'] / 2 if self.well is not None else fpu.infinity


class IdtTubeWellGeometry(WellGeometry):
    def __init__(self, well, config):
        super().__init__(well, config)

    @property
    def radial_clearance_tolerance(self):
        return 1.5  # extra because these tubes have some slop in their labware, don't want to rattle tube todo: make retval dependent on labware

    def depth_from_volume(self, vol):
        if vol <= 0.0:
            return 0.0
        if vol <= 67.1109:
            return 0.9095678851543723*cubeRoot(vol)
        return 2.464193794602757 + 0.018359120058446303*vol

    def volume_from_depth(self, depth):
        if depth <= 0:
            return 0
        if depth <= 3.69629:
            return 1.3289071745212766*cube(depth)
        return -134.221781150621 + 54.46884147042437*depth

    def radius_from_depth(self, depth):
        if depth <= 0:
            return 0
        if depth <= 3.69629:
            return 1.126504715663486*depth
        return 4.163888894893057

    @property
    def well_capacity(self):
        return 2153.47

    @property
    def well_depth(self):
        return 42

    @property
    def well_diameter_at_top(self):
        return 8.32778

    @property
    def min_aspiratable_volume(self):
        return 75  # a rough estimate, but seems functionally useful

    @property
    def rim_lip_height(self):
        raise 6.38

    @property
    def outside_height(self):
        return 45.01


class Biorad96WellPlateWellGeometry(WellGeometry):
    def __init__(self, well, config):
        super().__init__(well, config)

    def depth_from_volume(self, vol):
        if vol <= 0.0:
            return 0.0
        if vol <= 122.784:
            return -8.576177710037857 + 3.105085210707993*cubeRoot(21.069816179177707 + 1.3464508185574342*vol)
        return 3.9168885170626426 + 0.04270953403155694*vol

    def volume_from_depth(self, depth):
        if depth <= 0.0:
            return 0.0
        if depth <= 9.16092:
            return depth*(5.473911039614858 + (0.6382693111883633 + 0.024807839139547*depth)*depth)
        return -91.7099332942513 + 23.41397588793937*depth

    def radius_from_depth(self, depth):
        if depth <= 0:
            return 0
        if depth <= 9.16092:
            return 1.32 + 0.15391472105982892*depth
        return 2.73

    @property
    def well_capacity(self):
        return 255.051

    @property
    def well_depth(self):
        return 14.81

    @property
    def well_diameter_at_top(self):
        return 5.46


class Eppendorf1point5mlTubeGeometry(WellGeometry):
    def __init__(self, well, config):
        super().__init__(well, config)

    def depth_from_volume(self, vol):
        if vol <= 0:
            return 0
        if vol <= 0.67718:
            return 3.2346271740790455/cubeRoot(-3*vol + sqrt(106.32185983362221 + 9*square(vol))) - 0.6827840632552957*cubeRoot(-3*vol + sqrt(106.32185983362221 + 9*square(vol)))
        if vol <= 463.316:
            return -8.597168410942386 + 2.324577069455727*cubeRoot(52.28910925575565 + 2.660323800283652*vol)
        return -214.3418544824842 + 19.561686679619903*cubeRoot(1474.2109760964095 + 0.37305557584692783*vol)

    def volume_from_depth(self, depth):
        if depth <= 0:
            return 0
        if depth <= 0.194095:
            return 3.4691851885468576*depth + 0.5235987755982988*cube(depth)
        if depth <= 16.6742:
            return -0.6400085458081618 + depth*(6.6353796285770015 + (0.7718098926771707 + 0.029924965049919594*depth)*depth)
        return -425.3442361254281 + depth*(49.356322291514765 + (0.23026917636167102 + 0.00035810267810680666*depth)*depth)

    def radius_from_depth(self, depth):
        if depth <= 0:
            return 0
        if depth <= 0.194095:
            return 2.2698281410529737*sqrt((2.2462247020231834 - 0.19409486595347666*depth)*depth)
        if depth <= 16.6742:
            return 1.4533089603930036 + 0.16904507285715673*depth
        return 3.963660597359791 + 0.018492238050892146*depth

    @property
    def well_capacity(self):
        return 1788.68

    @property
    def well_depth(self):
        return 37.8

    @property
    def outside_height(self):
        return 38.9

    @property
    def well_diameter_at_top(self):
        return 9.32533

    @property
    def rim_lip_height(self):
        return 2


class Eppendorf5point0mlTubeGeometry(WellGeometry):
    def __init__(self, well, config):
        super().__init__(well, config)

    def depth_from_volume(self, vol):
        if vol <= 0:
            return 0
        if vol <= 4.97137:
            return 3.33494414484718/cubeRoot(-3*vol + sqrt(116.52404878202921 + 9*square(vol))) - 0.6827840632552957*cubeRoot(-3*vol + sqrt(116.52404878202921 + 9*square(vol)))
        if vol <= 1014.06:
            return -4.527482527717488 + 1.4293857409730928*cubeRoot(39.92570761834668 + 4.646502581189681*vol)
        return -302.1252531106694 + 15.294554814291097*cubeRoot(8610.391237925141 + 0.6794188941067961*vol)

    def volume_from_depth(self, depth):
        if depth <= 0:
            return 0
        if depth <= 1.16088:
            return 3.576776614210246*depth + 0.5235987755982988*cube(depth)
        if depth <= 19.5033:
            return -1.7535856833793924 + depth*(4.531691130053943 + (1.000929567880315 + 0.07369287175618172*depth)*depth)
        return -1327.949641310943 + depth*(112.65414304343028 + (0.3728723166420145 + 0.0004113882270161532*depth)*depth)

    def radius_from_depth(self, depth):
        if depth <= 0:
            return 0
        if depth <= 1.16088:
            return 0.9281232870726978*sqrt((3.624697354653752 - 1.1608835603563077*depth)*depth)
        if depth <= 19.5033:
            return 1.2010337579883252 + 0.26527628779029744*depth
        return 5.9882324145182295 + 0.01982036374935098*depth

    @property
    def well_capacity(self):
        return 6127.44

    @property
    def well_depth(self):
        return 55.4

    @property
    def outside_height(self):
        return 56.7

    @property
    def well_diameter_at_top(self):
        return 14.1726

    @property
    def rim_lip_height(self):
        return 2.2


class FalconTube15mlGeometry(WellGeometry):
    def __init__(self, well, config):
        super().__init__(well, config)

    def depth_from_volume(self, vol):
        if vol <= 0.0:
            return 0.0
        if vol <= 1232.34:
            return -4.605312927271903 + 1.4295474713971166*cubeRoot(33.43348831212188 + 5.259708112808352*vol)
        return -803.7743858256094 + 27.100445027181177*cubeRoot(27390.881699748476 + 0.7386443071956942*vol)

    def volume_from_depth(self, depth):
        if depth <= 0:
            return 0.0
        if depth <= 22.0945:
            return depth*(4.1407799998941535 + (0.8991310830091779 + 0.06507926078773585*depth)*depth)
        return -1761.2447144832822 + depth*(131.83324928521762 + (0.1640177288677879 + 0.00006801980413086297*depth)*depth)

    def radius_from_depth(self, depth):
        if depth <= 0:
            return 0.0
        if depth <= 22.0945:
            return 1.1480641142716852 + 0.2492912278496944*depth
        return 6.477949256918969 + 0.008059412406212692*depth

    @property
    def well_capacity(self):
        return 16202.8  # compare to 15000 in opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical

    @property
    def outside_height(self):
        return 119.40

    @property
    def well_depth(self):
        return 118.07  # compare to 117.5 in opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical

    @property
    def well_diameter_at_top(self):
        return 14.859

    @property
    def rim_lip_height(self):
        return 7.28


class FalconTube50mlGeometry(WellGeometry):  # not yet finished
    def __init__(self, well, config):
        super().__init__(well, config)

    def depth_from_volume(self, volume):
        pass

    def volume_from_depth(self, depth):
        pass

    def radius_from_depth(self, depth):
        pass

    @property
    def well_capacity(self):
        return 50000  # nominal

    @property
    def outside_height(self):
        return 114.11

    @property
    def well_depth(self):
        return 113.5

    @property
    def well_diameter_at_top(self):
        return 27.74

    @property
    def rim_lip_height(self):
        return 10.26


def well_geometry(well):
    assert is_well(well)
    try:
        return well.geometry
    except AttributeError:
        return UnknownWellGeometry(well)

# endregion

########################################################################################################################
# Pipette Geometry
########################################################################################################################

# region Pipette Geometry

class RadialClearanceManager(object):

    def __init__(self, config):
        self.config = config
        self._functions = {
            ('p50_single_v1.4', 'opentrons/opentrons_96_tiprack_300ul/1', FalconTube15mlGeometry): self.p50_single_v1_4_opentrons_96_tiprack_300ul_falcon15ml,
            ('p50_single_v1.4', 'opentrons/opentrons_96_tiprack_300ul/1', Eppendorf1point5mlTubeGeometry): self.p50_single_v1_4_opentrons_96_tiprack_300ul_eppendorf1_5ml,
            ('p50_single_v1.4', 'opentrons/opentrons_96_tiprack_300ul/1', Eppendorf5point0mlTubeGeometry): self.p50_single_v1_4_opentrons_96_tiprack_300ul_eppendorf5_0ml,
            ('p50_single_v1.4', 'opentrons/opentrons_96_tiprack_300ul/1', IdtTubeWellGeometry): self.p50_single_v1_4_opentrons_96_tiprack_300ul_idt_tube,
            ('p50_single_v1.4', 'opentrons/opentrons_96_tiprack_300ul/1', Biorad96WellPlateWellGeometry): self.p50_single_v1_4_opentrons_96_tiprack_300ul_biorad_plate_well,
        }

    def get_clearance_function(self, pipette, well):
        key = (pipette.model, pipette.current_tip_tiprack.uri, self.config.well_geometry(well).__class__)
        return self._functions.get(key, None)

    def _free_sailing(self):
        return fpu.infinity

    def p50_single_v1_4_opentrons_96_tiprack_300ul_falcon15ml(self, depth):
        if depth < 0:
            return 0
        if depth < 4.21826:
            return 0.3181014675267553 + 0.2492912278496944*depth
        if depth < 52.59:
            return 1.3356795812777742 + 0.008059412406212692*depth
        if depth < 59.9064:
            return -14.87960706782048 + 0.3163934426229509*depth
        if depth <= 118.07:
            return 3.5915771349525776 + 0.008059412406212692*depth
        return self._free_sailing()

    def p50_single_v1_4_opentrons_96_tiprack_300ul_eppendorf1_5ml(self, depth):
        if depth <= 0.0220787:
            return 0
        if depth < 0.114979:
            return -0.505 + 2.2698281410529737*sqrt((2.2462247020231834 - 0.19409486595347666*depth)*depth)
        if depth < 12.2688:
            return 0.6233463136480737 + 0.16904507285715673*depth
        if depth < 12.6:
            return 2.3141559767629554 + 0.031231049120679123*depth
        if depth < 19.9:
            return 2.05177678472461 + 0.05205479452054796*depth
        if depth <= 37.8:
            return 2.2593854454167044 + 0.04162219850586984*depth
        return self._free_sailing()

    def p50_single_v1_4_opentrons_96_tiprack_300ul_eppendorf5_0ml(self, depth):
        if depth <= 0.0839332:
            return 0
        if depth < 0.333631:
            return -0.505 + 0.9281232870726978*sqrt((3.624697354653752 - 1.1608835603563077*depth)*depth)
        if depth <= 16.3089:
            return 0.3710711112433953 + 0.26527628779029744*depth
        if depth <= 55.4:
            return 4.188102882787763 + 0.031231049120679123*depth
        return self._free_sailing()

    def p50_single_v1_4_opentrons_96_tiprack_300ul_idt_tube(self, depth):
        if depth <= 0.448289:
            return 0
        if depth <= 1.99878:
            return -0.505 + 1.126504715663486*depth
        if depth <= 42:
            return 1.6842072696656454 + 0.031231049120679123*depth
        return self._free_sailing()

    def p50_single_v1_4_opentrons_96_tiprack_300ul_biorad_plate_well(self, depth):
        if depth < 0:
            return 0
        if depth < 7.47114:
            return 0.4900373532550715 + 0.15391472105982892*depth
        if depth <= 14.81:
            return 1.0443669402110198 + 0.07971864009378668*depth
        return self._free_sailing()

# endregion


########################################################################################################################
# Custom Pipette objects
#   see http://code.activestate.com/recipes/577555-object-wrapper-class/
#   see https://stackoverflow.com/questions/1081253/inheriting-from-instance-in-python
#
# Goals:
#   * avoid blowout before trashing tip (that's useless, just creates aerosols)
#   * support speed and flow rate retrieval
#   * support option to leave tip attached at end of transfer
########################################################################################################################

@enum.unique
class TipWetness(Enum):
    NONE = enum.auto()
    DRY = enum.auto()
    WETTING = enum.auto()
    WET = enum.auto()

class EnhancedPipette(Pipette):

    #-------------------------------------------------------------------------------------------------------------------
    # Construction
    #-------------------------------------------------------------------------------------------------------------------

    def __new__(cls, parentInst, config):
        parentInst.__class__ = EnhancedPipette
        return parentInst

    class AspirateParamsHack(object):
        def __init__(self) -> None:
            self.pre_wet_during_transfer_kw = '_do_pre_wet_during_transfer'
            self.pre_wet_during_transfer = None
            self.ms_pause_during_transfer_kw = '_do_pause_during_transfer'
            self.ms_pause_during_transfer = None

    class DispenseParamsHack(object):
        def __init__(self) -> None:
            self.full_dispense_from_dispense = False
            self.full_dispense_during_transfer_kw = '_do_full_dispense_during_transfer'
            self.full_dispense_during_transfer = False
            self.fully_dispensed = False

    # noinspection PyMissingConstructor
    def __init__(self, parentInst, config):
        self.config = config
        self.prev_aspirated_location = None
        self.aspirate_params_hack = EnhancedPipette.AspirateParamsHack()
        self.dispense_params_hack = EnhancedPipette.DispenseParamsHack()
        self.tip_wetness = TipWetness.NONE
        self.mixes_in_progress = list()
        self.radial_clearance_manager = RadialClearanceManager(self.config)

        # load the config (again) in order to extract some more data later
        from opentrons.config import pipette_config
        from opentrons.config.pipette_config import configs
        pipette_model_version, pip_id = instruments._pipette_details(self.mount, self.name)
        self.pipette_config = pipette_config.load(pipette_model_version, pip_id)
        if not hasattr(self.pipette_config, 'drop_tip_min'):  # future-proof
            cfg = configs[pipette_model_version]  # ignores the id-based overrides done by pipette_config.load, but we can live with that
            self.pipette_config_drop_tip_min = cfg['dropTip']['min']  # hack: can't add field to pipette_config, so we do it this way
        else:
            self.pipette_config_drop_tip_min = self.pipette_config.drop_tip_min

        # try to mitigate effects of static electricity on small pipettes: they can cling to the tip on drop, causing disasters when next tips are picked up
        if self.config.enable_enhancements and self.name == 'p10_single':
            # dropping twice probably will help
            # if 'doubleDropTip' not in self.quirks:  # not necessary, in the end, it seems
            #     self.quirks.append('doubleDropTip')
            # plunging lower also helps, clearly
            self.plunger_positions['drop_tip'] = self.pipette_config_drop_tip_min

    #-------------------------------------------------------------------------------------------------------------------
    # Rates and speeds
    #-------------------------------------------------------------------------------------------------------------------

    def _get_speed(self, func):
        return self.speeds[func]

    def get_speeds(self):
        return {'aspirate': self._get_speed('aspirate'),
                'dispense': self._get_speed('dispense'),
                'blow_out': self._get_speed('blow_out')}

    def get_flow_rates(self):
        return {'aspirate': self._get_speed('aspirate') * self._get_ul_per_mm('aspirate'),
                'dispense': self._get_speed('dispense') * self._get_ul_per_mm('dispense'),
                'blow_out': self._get_speed('blow_out') * self._get_ul_per_mm('dispense')}

    def _get_ul_per_mm(self, func):  # hack, but there seems no public way
        return self._ul_per_mm(self.max_volume, func)

    #-------------------------------------------------------------------------------------------------------------------
    # Transfers
    #-------------------------------------------------------------------------------------------------------------------

    def _operations(self, plan, i):
        while i < len(plan):
            step = plan[i]
            if step.get('aspirate'):
                yield ('aspirate', i,)
            if step.get('dispense'):
                yield ('dispense', i,)
            i += 1

    def _has_aspirate(self, plan, i):
        return plan[i].get('aspirate')

    def _has_dispense(self, plan, i):
        return plan[i].get('dispense')

    # Does the aspiration starting here have an active disposal volume? Also,
    # record for all steps that value together with the number of remaining 'dispense'
    # steps before the next aspirate
    def has_disposal_vol(self, plan, i, step_info_map, **kwargs):
        assert self._has_aspirate(plan, i)

        in_distribute_with_disposal_vol = kwargs.get('mode', 'transfer') == 'distribute' and kwargs.get('disposal_vol', 0) > 0

        dispense_count = 0
        i_aspirate_next = len(plan)  # default in case we don't go through the loop at all
        for op, j in self._operations(plan, i):
            if op == 'dispense':
                dispense_count += 1
            elif op == 'aspirate' and j > i:
                i_aspirate_next = j
                break

        result = dispense_count > 1 and in_distribute_with_disposal_vol

        remaining_dispense_count = dispense_count-1 if self._has_dispense(plan, i) else dispense_count
        while i < i_aspirate_next:
            assert remaining_dispense_count >= 0
            existing = step_info_map.get(i, None)
            value = (remaining_dispense_count, result,)
            if existing is None:
                step_info_map[i] = value
            else:
                assert existing == value  # should be idempotent
            i += 1
            remaining_dispense_count -= 1

        return result

    # Copied and overridden
    # New kw args:
    #   'keep_last_tip': if true, then tip is not dropped at end of transfer
    #   'full_dispense': if true, and if a dispense empties the pipette, then dispense to blow_out 'position' instead of 'bottom' position
    #   'allow_overspill': if true, we allow over spill (of disposal_vol) from one run of (asp, disp+) to the next
    #   'allow_blow_elision': if true, then blow-outs which are not logically needed (such as before a tip drop) are elided
    def _run_transfer_plan(self, tips, plan, **kwargs):
        air_gap = kwargs.get('air_gap', 0)
        touch_tip = kwargs.get('touch_tip', False)
        is_distribute = kwargs.get('mode', 'transfer') == 'distribute'

        total_transfers = len(plan)
        seen_aspirate = False
        assert len(plan) == 0 or plan[0].get('aspirate')  # first step must be an aspirate

        step_info_map = dict()
        for i, step in enumerate(plan):
            aspirate = step.get('aspirate')
            dispense = step.get('dispense')

            if aspirate:
                # *always* record on aspirates so we can test has_disposal_vol on subsequent dispenses
                have_disposal_vol = self.has_disposal_vol(plan, i, step_info_map, **kwargs)

                # we might have overspill from a previous transfer.
                if self.current_volume > 0:
                    info(pretty.format('carried over {0:n} uL from prev operation', self.current_volume))

                if not seen_aspirate:
                    assert i == 0

                    if kwargs.get('pre_wet', None) and kwargs.get('mix_before', None):
                        warn("simultaneous use of 'pre_wet' and 'mix_before' is not tested")

                    if (kwargs.get('allow_overspill', self.config.allow_overspill_default) and self.config.enable_enhancements) and zeroify(self.current_volume) > 0:
                        this_aspirated_location, __ = unpack_location(aspirate['location'])
                        if self.prev_aspirated_location is this_aspirated_location:
                            if have_disposal_vol:
                                # try to remove current volume from this aspirate
                                new_aspirate_vol = zeroify(aspirate.get('volume') - self.current_volume)
                                if new_aspirate_vol == 0 or new_aspirate_vol >= self.min_volume:
                                    aspirate['volume'] = new_aspirate_vol
                                    info(pretty.format('reduced this aspirate by {0:n} uL', self.current_volume))
                                    extra = 0  # can't blow out since we're relying on its presence in pipette
                                else:
                                    extra = self.current_volume - aspirate['volume']
                                    assert zeroify(extra) > 0
                            else:
                                info(pretty.format("overspill of {0:n} uL isn't for disposal", self.current_volume))
                                extra = self.current_volume
                        else:
                            # different locations; can't re-use
                            info('this aspirate is from location different than current pipette contents')
                            extra = self.current_volume
                        if zeroify(extra) > 0:
                            # quiet_log('blowing out overspill of %s uL' % format_number(self.current_volume))
                            self._blowout_during_transfer(loc=None, **kwargs)  # loc isn't actually used

                    elif zeroify(self.current_volume) > 0:
                        info(pretty.format('blowing out unexpected overspill of {0:n} uL', self.current_volume))
                        self._blowout_during_transfer(loc=None, **kwargs)  # loc isn't actually used

                seen_aspirate = True

                self._add_tip_during_transfer(tips, **kwargs)
                # Previous blow-out elisions that don't reduce their adjacent aspirates (because they're not
                # carrying disposal_vols) might eventually catch up with us in the form of blowing the capacity
                # of the pipette. When they do, we give in, and carry out the blow-out. This still can be a net
                # win, in that we reduce the overall number of blow-outs. We might be tempted here to reduce
                # the capacity of the overflowing aspirate, but that would reduce precision (we still *could*
                # do that if it has disposal_vol, but that doesn't seem worth it).
                if self.current_volume + aspirate['volume'] > self._working_volume:
                    info(pretty.format('current {0:n} uL with aspirate(has_disposal={1}) of {2:n} uL would overflow capacity',
                          self.current_volume,
                          self.has_disposal_vol(plan, i, step_info_map, **kwargs),
                          aspirate['volume']))
                    self._blowout_during_transfer(loc=None, **kwargs)  # loc isn't actually used
                kwargs[self.aspirate_params_hack.pre_wet_during_transfer_kw] = not not kwargs.get('pre_wet', self.config.aspirate.pre_wet.default)
                kwargs[self.aspirate_params_hack.ms_pause_during_transfer_kw] = kwargs.get('pause', self.config.aspirate.pause.ms_default)
                self._aspirate_during_transfer(aspirate['volume'], aspirate['location'], **kwargs)

            if dispense:
                if self.current_volume < dispense['volume']:
                    warn(pretty.format('current {0:n} uL will truncate dispense of {1:n} uL', self.current_volume, dispense['volume']))

                can_full_dispense = self.current_volume - dispense['volume'] <= 0
                kwargs[self.dispense_params_hack.full_dispense_during_transfer_kw] = not not (kwargs.get('full_dispense', self.config.dispense.full_dispense.default) and can_full_dispense)
                self._dispense_during_transfer(dispense['volume'], dispense['location'], **kwargs)

                do_touch = touch_tip or touch_tip is 0
                is_last_step = step is plan[-1]
                if is_last_step or plan[i + 1].get('aspirate'):
                    do_drop = not is_last_step or not (kwargs.get('keep_last_tip', False) and self.config.enable_enhancements)
                    # original always blew here. there are several reasons we could still be forced to blow
                    do_blow = not is_distribute  # other modes (are there any?) we're not sure about
                    do_blow = do_blow or kwargs.get('blow_out', False)  # for compatibility
                    do_blow = do_blow or do_touch  # for compatibility
                    do_blow = do_blow or not (kwargs.get('allow_blow_elision', self.config.allow_blow_elision_default) and self.config.enable_enhancements)
                    if not do_blow:
                        if is_last_step:
                            if self.current_volume > 0:
                                if not (kwargs.get('allow_overspill', self.config.allow_overspill_default) and self.config.enable_enhancements):
                                    do_blow = True
                                elif self.current_volume > kwargs.get('disposal_vol', 0):
                                    warn(pretty.format('carried over {0:n} uL to next operation', self.current_volume))
                                else:
                                    info(pretty.format('carried over {0:n} uL to next operation', self.current_volume))
                        else:
                            # if we can, account for any overspill in the next aspirate
                            if self.current_volume > 0:
                                if self.has_disposal_vol(plan, i + 1, step_info_map, **kwargs):
                                    next_aspirate = plan[i + 1].get('aspirate'); assert next_aspirate
                                    next_aspirated_location, __ = unpack_location(next_aspirate['location'])
                                    if self.prev_aspirated_location is next_aspirated_location:
                                        new_aspirate_vol = zeroify(next_aspirate.get('volume') - self.current_volume)
                                        if new_aspirate_vol == 0 or new_aspirate_vol >= self.min_volume:
                                            next_aspirate['volume'] = new_aspirate_vol
                                            info(pretty.format('reduced next aspirate by {0:n} uL', self.current_volume))
                                        else:
                                            do_blow = True
                                    else:
                                        do_blow = True  # different aspirate locations
                                else:
                                    # Next aspirate doesn't *want* our overspill, so we don't reduce his
                                    # volume. But it's harmless to just leave the overspill present; might
                                    # be useful down the line
                                    pass
                            else:
                                pass  # currently empty
                    if do_touch:
                        self.touch_tip(touch_tip)
                    if do_blow:
                        self._blowout_during_transfer(dispense['location'], **kwargs)
                    if do_drop:
                        tips = self._drop_tip_during_transfer(tips, i, total_transfers, **kwargs)
                else:
                    if air_gap:
                        self.air_gap(air_gap)
                    if do_touch:
                        self.touch_tip(touch_tip)

    #-------------------------------------------------------------------------------------------------------------------
    # Aspirate and dispense
    #-------------------------------------------------------------------------------------------------------------------

    def _pre_wet(self, well, volume, location, rate, pre_wet):
        if pre_wet is None:
            pre_wet = self.aspirate_params_hack.pre_wet_during_transfer
        if pre_wet is None:
            pre_wet = self.config.aspirate.pre_wet.default
        if pre_wet and self.config.enable_enhancements:
            if self.tip_wetness is TipWetness.DRY:
                pre_wet_volume = min(
                    self.max_volume * self.config.aspirate.pre_wet.max_volume_fraction,
                    max(volume, self.well_volume(well).available_volume_min))
                pre_wet_rate = self.config.aspirate.pre_wet.rate_func(rate)
                self.tip_wetness = TipWetness.WETTING
                def do_pre_wet():
                    for i in range(self.config.aspirate.pre_wet.count):
                        self.aspirate(volume=pre_wet_volume, location=location, rate=pre_wet_rate, pre_wet=False, ms_pause=0)
                        self.dispense(volume=pre_wet_volume, location=location, rate=pre_wet_rate, full_dispense=(i+1 == self.config.aspirate.pre_wet.count))
                info_while(pretty.format('prewetting tip in well {0} vol={1:n}', well.get_name(), pre_wet_volume), do_pre_wet)
                self.tip_wetness = TipWetness.WET

    def aspirate(self, volume=None, location=None, rate=1.0, pre_wet=None, ms_pause=None):
        if not helpers.is_number(volume):  # recapitulate super
            if volume and not location:
                location = volume
            volume = self._working_volume - self.current_volume
        location = location if location else self.previous_placeable
        well, _ = unpack_location(location)

        current_well_volume = self.well_volume(well).current_volume_min
        needed_well_volume = self.well_geometry(well).min_aspiratable_volume + volume;
        if current_well_volume < needed_well_volume:
            msg = pretty.format('aspirating too much from well={0} have={1:n} need={2:n}', well.get_name(), current_well_volume, needed_well_volume)
            warn(msg)

        self._pre_wet(well, volume, location, rate, pre_wet)
        location = self._adjust_location_to_liquid_top(location=location, aspirate_volume=volume,
                                                       clearances=self.config.aspirate,
                                                       extra_clearance=getattr(well, self.config.aspirate.extra_top_clearance_name, 0))
        super().aspirate(volume=volume, location=location, rate=rate)

        # if we're asked to, pause after aspiration to let liquid rise
        if ms_pause is None:
            ms_pause = self.aspirate_params_hack.ms_pause_during_transfer
        if ms_pause is None:
            ms_pause = self.config.aspirate.pause.ms_default
        if self.config.enable_enhancements and ms_pause > 0 and not self.is_mix_in_progress():
            self.delay(ms_pause / 1000.0)

        # track volume todo: what if we're doing an air gap
        well, __ = unpack_location(location)
        self.well_volume(well).aspirate(volume)
        if volume != 0:
            self.prev_aspirated_location = well

    def _aspirate_during_transfer(self, vol, loc, **kwargs):
        assert kwargs.get(self.aspirate_params_hack.pre_wet_during_transfer_kw) is not None
        assert kwargs.get(self.aspirate_params_hack.ms_pause_during_transfer_kw) is not None
        self.aspirate_params_hack.pre_wet_during_transfer = kwargs.get(self.aspirate_params_hack.pre_wet_during_transfer_kw)
        self.aspirate_params_hack.ms_pause_during_transfer = kwargs.get(self.aspirate_params_hack.ms_pause_during_transfer_kw)
        super()._aspirate_during_transfer(vol, loc, **kwargs)  # might 'mix_before' todo: is that ok? seems like it is...
        self.aspirate_params_hack.pre_wet_during_transfer = None
        self.aspirate_params_hack.ms_pause_during_transfer = None

    def dispense(self, volume=None, location=None, rate=1.0, full_dispense: bool = False):
        if not helpers.is_number(volume):  # recapitulate super
            if volume and not location:
                location = volume
            volume = self._working_volume - self.current_volume
        location = location if location else self.previous_placeable
        well, _ = unpack_location(location)
        if is_close(volume, self.current_volume):  # avoid finicky floating-point precision issues
            volume = self.current_volume
        location = self._adjust_location_to_liquid_top(location=location, aspirate_volume=None,
                                                       clearances=self.config.dispense,
                                                       extra_clearance=getattr(well, self.config.dispense.extra_top_clearance_name, 0))
        self.dispense_params_hack.full_dispense_from_dispense = full_dispense
        super().dispense(volume=volume, location=location, rate=rate)
        self.dispense_params_hack.full_dispense_from_dispense = False
        if self.dispense_params_hack.fully_dispensed:
            assert self.current_volume == 0
            if self.current_volume == 0:
                pass  # nothing to do: the next self._position_for_aspirate will reposition for us: 'if pipette is currently empty, ensure the plunger is at "bottom"'
            else:
                raise NotImplementedError
            self.dispense_params_hack.fully_dispensed = False
        # track volume
        well, __ = unpack_location(location)
        self.well_volume(well).dispense(volume)

    def _dispense_during_transfer(self, vol, loc, **kwargs):
        assert kwargs.get(self.dispense_params_hack.full_dispense_during_transfer_kw) is not None
        self.dispense_params_hack.full_dispense_during_transfer = kwargs.get(self.dispense_params_hack.full_dispense_during_transfer_kw)
        super()._dispense_during_transfer(vol, loc, **kwargs)  # might 'mix_after' todo: is that ok? probably: we'd just do full_dispense on all of those too?
        self.dispense_params_hack.full_dispense_during_transfer = False

    def _dispense_plunger_position(self, ul):
        mm_from_vol = super()._dispense_plunger_position(ul)  # retrieve position historically used
        if self.config.enable_enhancements and (self.dispense_params_hack.full_dispense_from_dispense or self.dispense_params_hack.full_dispense_during_transfer):
            mm_from_blow = self._get_plunger_position('blow_out')
            info(pretty.format('dispensing to mm={0:n} instead of mm={1:n}', mm_from_blow, mm_from_vol))
            self.dispense_params_hack.fully_dispensed = True
            return mm_from_blow
        else:
            self.dispense_params_hack.fully_dispensed = False
            return mm_from_vol

    def _adjust_location_to_liquid_top(self, location=None, aspirate_volume=None, clearances=None, extra_clearance=0, allow_above=False):
        if isinstance(location, Placeable):
            well = location; assert is_well(well)
            current_well_volume = self.well_volume(well).current_volume_min
            liquid_depth = self.well_geometry(well).depth_from_volume_min(current_well_volume if aspirate_volume is None else current_well_volume - aspirate_volume)
            z = self._top_clearance(liquid_depth=liquid_depth, clearance=(0 if clearances is None else clearances.top_clearance) + extra_clearance)
            if clearances is not None:
                z = max(z, clearances.bottom_clearance)
            if not allow_above:
                z = min(z, well.z_size())
            result = well.bottom(z)
        else:
            result = location  # we already had a displacement baked in to the location, don't adjust (when does this happen?)
        assert isinstance(result, tuple)
        return result

    def well_volume(self, well):
        return self.config.well_volume(well)

    def well_geometry(self, well):
        return self.config.well_geometry(well)

    #-------------------------------------------------------------------------------------------------------------------
    # Tip Management
    #-------------------------------------------------------------------------------------------------------------------

    @property
    def current_tip_overlap(self):  # tip_overlap with respect to the current tip
        assert self.has_tip
        d = self.pipette_self.config.tip_overlap
        try:
            return d[self.current_tip_tiprack.uri]
        except (KeyError, AttributeError):
            return d['default']

    @property
    def full_tip_length(self):
        return self.current_tip_tiprack.tip_length

    @property
    def nominal_available_tip_length(self):
        return self.full_tip_length - self.current_tip_overlap

    @property
    def calibrated_available_tip_length(self):
        return self._tip_length  # tiprack is implicit (2019.10.19), as only one kind of tip is calibrated

    @property
    def available_tip_length(self):
        return self.calibrated_available_tip_length

    @property
    def current_tip_tiprack(self):  # tiprack of the current tip
        assert self.has_tip
        return self.current_tip().parent

    #-------------------------------------------------------------------------------------------------------------------

    def tip_coords_absolute(self):
        xyz = pose_tracker.absolute(self.robot.poses, self)
        return Vector(xyz)

    def pick_up_tip(self, location=None, presses=None, increment=None):
        result = super().pick_up_tip(location, presses, increment)
        self.tip_wetness = TipWetness.DRY
        return result

    def drop_tip(self, location=None, home_after=True):
        result = super().drop_tip(location, home_after)
        self.tip_wetness = TipWetness.NONE
        return result

    def done_tip(self):  # a handy little utility that looks at self.config.trash_control
        if self.has_tip:
            if self.current_volume > 0:
                info(pretty.format('{0} has {1:n} uL remaining', self.name, self.current_volume))
            if self.config.trash_control:
                self.drop_tip()
            else:
                self.return_tip()

    #-------------------------------------------------------------------------------------------------------------------
    # Blow outs
    #-------------------------------------------------------------------------------------------------------------------

    def blow_out(self, location=None):
        super().blow_out(location)
        self._shake_tip(location)  # try to get rid of pesky retentive drops

    def _shake_tip(self, location):
        # Modelled after Pipette._shake_off_tips()
        shake_off_distance = SHAKE_OFF_TIPS_DROP_DISTANCE / 2  # / 2 == less distance than shaking off tips
        if location:
            placeable, _ = unpack_location(location)
            # ensure the distance is not >25% the diameter of placeable
            x = placeable.x_size()
            if x != 0:  # trash well has size zero
                shake_off_distance = max(min(shake_off_distance, x / 4), 1.0)
        self.robot.gantry.push_speed()
        self.robot.gantry.set_speed(SHAKE_OFF_TIPS_SPEED)  # tip is fully on, we can handle it
        self.robot.poses = self._jog(self.robot.poses, 'x', -shake_off_distance)  # move left
        self.robot.poses = self._jog(self.robot.poses, 'x', shake_off_distance * 2)  # move right
        self.robot.poses = self._jog(self.robot.poses, 'x', -shake_off_distance * 2)  # move left
        self.robot.poses = self._jog(self.robot.poses, 'x', shake_off_distance * 2)  # move right
        self.robot.poses = self._jog(self.robot.poses, 'x', -shake_off_distance)  # move left
        self.robot.gantry.pop_speed()

    #-------------------------------------------------------------------------------------------------------------------
    # Mixing
    #-------------------------------------------------------------------------------------------------------------------

    def _mix_during_transfer(self, mix, loc, **kwargs):
        self.begin_internal_mix(True)
        super()._mix_during_transfer(mix, loc, **kwargs)
        self.end_internal_mix(True)

    def mix(self, repetitions=1, volume=None, location=None, rate=1.0):
        self.begin_internal_mix(False)
        result = super().mix(repetitions, volume, location, rate)
        self.end_internal_mix(False)
        return result

    def begin_internal_mix(self, during_transfer: bool):
        self.mixes_in_progress.append('internal_transfer_mix' if bool else 'internal_mix')

    def end_internal_mix(self, during_transfer: bool):
        top = self.mixes_in_progress.pop()
        assert top == ('internal_transfer_mix' if bool else 'internal_mix')

    def begin_layered_mix(self):
        self.mixes_in_progress.append('layered_mix')

    def end_layered_mix(self):
        top = self.mixes_in_progress.pop()
        assert top == 'layered_mix'

    def is_mix_in_progress(self):
        return len(self.mixes_in_progress) > 0

    # If count is provided, we do (at most) that many asp/disp cycles, clamped to an increment of min_incr
    def layered_mix(self, wells, msg='Mixing',
                    count=None,
                    min_incr=None,
                    incr=None,
                    count_per_incr=None,
                    volume=None,
                    keep_last_tip=None,  # todo: add ability to control tip changing per well
                    delay=None,
                    aspirate_rate=None,
                    dispense_rate=None,
                    initial_turnover=None,
                    max_tip_cycles=None):

        def do_layered_mix():
            self.begin_layered_mix()
            local_keep_last_tip = keep_last_tip if keep_last_tip is not None else self.config.layered_mix.keep_last_tip

            for well in wells:
                self._layered_mix_one(well, msg=msg,
                                      count=count,
                                      min_incr=min_incr,
                                      incr=incr,
                                      count_per_incr=count_per_incr,
                                      volume=volume,
                                      delay=delay,
                                      apirate_rate=aspirate_rate,
                                      dispense_rate=dispense_rate,
                                      initial_turnover=initial_turnover,
                                      max_tip_cycles=max_tip_cycles)
            if not local_keep_last_tip:
                self.done_tip()
            self.end_layered_mix()

        log_while(f'{msg} {[well.get_name() for well in wells]}', do_layered_mix)

    def _top_clearance(self, liquid_depth, clearance):
        assert liquid_depth >= 0
        if clearance > 0:
            return liquid_depth + clearance  # going up
        else:
            return liquid_depth + clearance  # going down. we used to clamp to at least a fraction of the current liquid depth, but not worthwhile as tube modelling accuracy has improved

    def _layered_mix_one(self, well, msg, **kwargs):
        def fetch(name, default=None):
            if default is None:
                default = getattr(self.config.layered_mix, name)
            result = kwargs.get(name, default)
            if result is None:
                result = default
            return result
        volume = fetch('volume', self.max_volume)
        incr = fetch('incr')
        count_per_incr = fetch('count_per_incr')
        count = fetch('count')
        min_incr = fetch('min_incr')
        ms_pause = fetch('ms_pause')
        initial_turnover = fetch('initial_turnover')
        max_tip_cycles = fetch('max_tip_cycles', fpu.infinity)
        pre_wet = fetch('pre_wet', False)  # not much point in pre-wetting during mixing; save some time, simpler. but we do so if asked

        current_well_volume = self.well_volume(well).current_volume_min
        liquid_depth = self.well_geometry(well).depth_from_volume(current_well_volume)
        liquid_depth_after_asp = self.well_geometry(well).depth_from_volume(current_well_volume - volume)
        msg = pretty.format("{0:s} well='{1:s}' cur_vol={2:n} well_depth={3:n} after_aspirate={4:n}", msg, well.get_name(), current_well_volume, liquid_depth, liquid_depth_after_asp)

        def do_one():
            count_ = count
            y_min = y = self.config.layered_mix.aspirate_bottom_clearance
            y_max = self._top_clearance(liquid_depth=liquid_depth_after_asp, clearance=self.config.layered_mix.top_clearance)
            if count_ is not None:
                if count_ <= 1:
                    y_max = y_min
                    y_incr = 1  # just so we only go one time through the loop
                else:
                    y_incr = (y_max - y_min) / (count_-1)
                    y_incr = max(y_incr, min_incr)
            else:
                assert incr is not None
                y_incr = incr

            def do_layer(y_layer):
                return y_layer <= y_max or is_close(y_layer, y_max)
            first = True
            tip_cycles = 0
            while do_layer(y):
                if not first:
                    self.delay(ms_pause / 1000.0)  # pause to let dispensed liquid disperse
                #
                if first and initial_turnover is not None:
                    count_ = int(0.5 + (initial_turnover / volume))
                    count_ = max(count_, count_per_incr)
                else:
                    count_ = count_per_incr
                if not self.has_tip:
                    self.pick_up_tip()

                radial_clearance_func = self.radial_clearance_manager.get_clearance_function(self, well)
                radial_clearance = 0 if radial_clearance_func is None or not self.config.layered_mix.enable_radial_randomness else radial_clearance_func(y_max)
                radial_clearance = max(0, radial_clearance - max(self.well_geometry(well).radial_clearance_tolerance, self.config.layered_mix.radial_clearance_tolerance))

                for i in range(count_):
                    tip_cycles += 1
                    need_new_tip = tip_cycles >= max_tip_cycles
                    full_dispense = need_new_tip or (not do_layer(y + y_incr) and i == count_ - 1)

                    theta = random.random() * (2 * math.pi)
                    _, dispense_coordinates = well.bottom(y_max)
                    random_offset = (radial_clearance * math.cos(theta), radial_clearance * math.sin(theta), 0)
                    dispense_location = (well, dispense_coordinates + random_offset)

                    self.aspirate(volume, well.bottom(y), rate=fetch('aspirate_rate', self.config.layered_mix.aspirate_rate_factor), pre_wet=pre_wet)
                    self.dispense(volume, dispense_location, rate=fetch('dispense_rate', self.config.layered_mix.dispense_rate_factor), full_dispense=full_dispense)
                    if need_new_tip:
                        self.done_tip()
                        tip_cycles = 0
                #
                y += y_incr
                first = False

        info_while(msg, do_one)


########################################################################################################################
# Custom Labware
########################################################################################################################

# region Custom Labware

class Point(object):
    def __init__(self, x=0, y=0):
        if is_indexable(x):
            y = x[1]
            x = x[0]
        self.x = x
        self.y = y

    def __add__(self, p):
        return Point(self.x + p.x, self.y + p.y)

    def __radd__(self, other):
        return self + other

    def __sub__(self, p):
        return Point(self.x - p.x, self.y - p.y)

    def __mul__(self, scalar):
        return Point(self.x * scalar, self.y * scalar)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, scalar):
        return Point(self.x / scalar, self.y / scalar)

    def __getitem__(self, index):
        if index == 0:
            return self.x
        if index == 1:
            return self.y
        raise IndexError

class PointF(Point):
    def __init__(self, x: float = 0.0, y: float = 0.0):
        super().__init__(x,y)


class WellGrid(object):
    def __init__(self, config, grid_size: Point, incr: PointF, offset=PointF(), origin_name='A1', origin=None, well_geometry=None):
        self.config = config
        self.grid_size = grid_size
        self.origin = self.well_name_to_indices(origin_name) if origin is None else origin
        self.max = self.origin + self.grid_size
        self.incr = incr
        self.offset = offset
        self.wells_matrix = self._create_wells_matrix(well_geometry)
        self.wells_by_name = dict()
        for row in self.wells_matrix:
            for well_dict in row:
                self.wells_by_name[well_dict['name']] = well_dict

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.wells_by_name[item]
        if is_indexable(item):
            return self.wells_matrix[item[1]][item[0]]  # item is (x,y), which is (col, row)
        raise IndexError

    @staticmethod
    def well_name_to_indices(name):  # zero-based
        return Point(ord(name[1]) - ord('1'), ord(name[0]) - ord('A'))

    def contains_indices(self, indices):
        return self.origin.x <= indices[0] < self.max.x and self.origin.y <= indices[1] < self.max.y

    def definition_map(self, rack, z_reference, hangable_tube_height):
        result = collections.OrderedDict()
        for col in range(self.grid_size.x):
            for row in range(self.grid_size.y):
                rc = self.wells_matrix[row][col]
                if rc['geometry'] is not None:
                    d = dict()
                    geometry: WellGeometry = rc['geometry']
                    d['depth'] = geometry.well_depth
                    d['totalLiquidVolume'] = geometry.well_capacity
                    d['shape'] = 'circular'
                    d['diameter'] = geometry.well_diameter_at_top
                    d['x'] = rc['x']
                    d['y'] = rc['y']
                    d['z'] = geometry.height_above_reference_plane(hangable_tube_height, rack) + z_reference - d['depth']
                    result[rc['name']] = d
        return result

    @property
    def well_ordering_names(self):  # values are well name, but in a 2D matrix
        result = []
        for col in range(self.grid_size.x):
            col_result = []
            for row in range(self.grid_size.y):
                col_result.append(self.wells_matrix[row][col]['name'])
            result.append(col_result)
        return result

    @property
    def well_names(self):
        return [well_name for column in self.well_ordering_names for well_name in column]

    @property
    def well_geometries(self):
        return [well_dict['geometry'] for row in self.wells_matrix for well_dict in row if well_dict['geometry'] is not None]

    def _create_wells_matrix(self, well_geometry=None):
        result = [None] * self.grid_size.y
        for row in range(self.grid_size.y):
            result[row] = [None] * self.grid_size.x
            for col in range(self.grid_size.x):
                d = {  # coordinate system origin is in lower left
                    'x': self.offset.x + self.incr.x * col,
                    'y': self.offset.y + self.incr.y * (self.grid_size.y - 1 - row),
                    'name': chr(ord('A')+row+self.origin.x) + chr(ord('1')+col+self.origin.y),
                    'geometry': None
                }
                if well_geometry is not None:
                    d['geometry'] = well_geometry(config=self.config, well=None)
                result[row][col] = d
        return result


class CustomTubeRack(object):
    def __init__(self, config, name,
                 dimensions=None,  # is either to reference plane (dimensions_measurement_geometry is None) or to the top of rack measure with some tube in place (otherwise)
                 dimensions_measurement_geometry=None,  # geometry used, if any, when 'dimensions' were measured
                 hangable_tube_height=None,  # tubes taller than this don't hang. this value is conservative, in that tubes slightly larger than this might still hang, depending on geometries of the tube and rack indentations
                 brand=None,
                 brandIds=None,
                 well_grids=None
                 ):
        assert name is not None
        self.config = config
        self.name = name
        self.reference_dimensions = dimensions if dimensions is not None else Vector(x=0, y=0, z=0)
        if dimensions_measurement_geometry is not None:
            # find the z height of the reference plane of the labware
            self.reference_dimensions = self.reference_dimensions - (0, 0, dimensions_measurement_geometry(well=None, config=self.config).rim_lip_height)
        self.hangable_tube_height = hangable_tube_height if hangable_tube_height is not None else fpu.infinity
        self.brand = {
            'brand': brand if brand is not None else 'Atkinson Labs'
        }
        if brandIds is not None:
            self.brand['brandId'] = brandIds
        self.metadata = {
            'displayName': name,
            'displayCategory': 'tubeRack',
            'displayVolumeUnits': 'µL',
            'tags': []
        }
        self.well_grids = [] if well_grids is None else well_grids
        self.load_result = None

    def __getitem__(self, item_name):
        if isinstance(item_name, str):
            return self.__getitem__(WellGrid.well_name_to_indices(item_name))
        if is_indexable(item_name):
            for well_grid in self.well_grids:
                if well_grid.contains_indices(item_name):
                    return well_grid.__getitem__(Point(item_name) - well_grid.origin)
        raise IndexError

    @property
    def max_rim_lip_height(self):
        result = 0
        for geometry in self.well_geometries:
            result = max(result, geometry.rim_lip_height)
        return result

    @property
    def max_tube_height_above_reference_plane(self):
        result = 0
        for geometry in self.well_geometries:
            result = max(result, geometry.height_above_reference_plane(self.hangable_tube_height, self))
        return result

    @property
    def dimensions(self):
        return self.reference_dimensions + (0, 0, self.max_tube_height_above_reference_plane)

    @property
    def well_names(self):
        result = []
        for well_grid in self.well_grids:
            result.extend(well_grid.well_names)
        return result

    @property
    def well_geometries(self):
        result = []
        for well_grid in self.well_grids:
            result.extend(well_grid.well_geometries)
        return result

    @property
    def _definition_map(self):
        dimensions = self.dimensions
        result = collections.OrderedDict()
        result['ordering'] = []
        for well_grid in self.well_grids:
            result['ordering'].extend(well_grid.well_ordering_names)
        result['brand'] = self.brand
        result['metadata'] = self.metadata
        result['dimensions'] = {
            'xDimension': dimensions.coordinates.x,
            'yDimension': dimensions.coordinates.y,
            'zDimension': dimensions.coordinates.z
        }
        result['wells'] = collections.OrderedDict()
        for well_grid in self.well_grids:
            for name, definition in well_grid.definition_map(self, self.reference_dimensions.coordinates.z, self.hangable_tube_height).items():
                result['wells'][name] = definition
        # todo: add 'groups', if that's still significant / worthwhile
        result['parameters'] = {
            'format': 'irregular',  # not 'regular': per correspondence from Opentrons, this field is obsolete, and 'irregular' is best for back-compat
            'quirks': [],
            'isTiprack': False,
            'isMagneticModuleCompatible': False,
            'loadName': self.name
        }
        result['namespace'] = 'custom_beta'
        result['version'] = 1
        result['schemaVersion'] = 2
        result['cornerOffsetFromSlot'] = {'x': 0, 'y': 0, 'z': 0}
        return result

    def load(self, slot=None, label=None):
        slot = str(slot)
        if self.load_result is None:
            def_map = self._definition_map
            if label is None:
                label = self.name
            self.load_result = robot.add_container_by_definition(def_map, slot, label=label)
            for well_name in self.well_names:
                well = self.load_result.wells(well_name)
                geometry = self[well_name].get('geometry', None)
                if geometry is not None:
                    assert geometry.well is None or geometry.well is well
                    assert getattr(well, 'geometry', None) is None or well.geometry is geometry
                    geometry.well = well
        return self.load_result

class Opentrons15Rack(CustomTubeRack):
    def __init__(self, config, name, brand=None, default_well_geometry=None):
        super().__init__(
            config,
            dimensions=Vector(127.76, 85.48, 80.83),
            dimensions_measurement_geometry=Eppendorf5point0mlTubeGeometry,
            hangable_tube_height=71.40,
            name=name,
            brand=brand,
            well_grids=[WellGrid(config,
                grid_size=Point(5, 3),
                incr=PointF(25.0, 25.0),
                offset=PointF(13.88, 17.74),
                well_geometry=default_well_geometry)
            ])

# an enhanced version of labware.load(tiprack_type, slot) that grabs more metadata
def load_tiprack(tiprack_type, slot, label=None):
    from opentrons.protocol_api import labware as new_labware
    from opentrons.legacy_api.robot.robot import _setup_container
    from opentrons.legacy_api.containers import load_new_labware_def
    slot = str(slot)
    share = False
    definition = new_labware.get_labware_definition(load_name=tiprack_type)
    container = load_new_labware_def(definition)
    container = _setup_container(container)
    #
    container.uri = new_labware.uri_from_definition(definition)
    container.tip_length = definition['parameters']['tipLength']
    container.tip_overlap = definition['parameters']['tipOverlap']
    #
    robot._add_container_obj(container, tiprack_type, slot, label, share)
    return container

# endregion

########################################################################################################################
# Logging
########################################################################################################################

# region Logging
# region Commands

# Enhance well name to include any label that might be present

def Well_get_name(self):
    result = super(Well, self).get_name()
    label = getattr(self, 'label', None)
    if label is not None:
        result += ' (' + label + ')'
    return result

def Well_top_coords_absolute(self):
    xyz = pose_tracker.absolute(robot.poses, self)
    return Vector(xyz)

Well.has_labelled_well_name = True
Well.get_name = Well_get_name
Well.top_coords_absolute = Well_top_coords_absolute


# Hook commands to provide more informative text

def z_from_bottom(location, clearance):
    if isinstance(location, Placeable):
        return min(location.z_size(), clearance)
    elif isinstance(location, tuple):
        well, vector = location
        _, vector_bottom = well.bottom(0)
        return vector.coordinates.z - vector_bottom.coordinates.z
    else:
        raise ValueError('Location should be (Placeable, (x, y, z)) or Placeable')

def command_aspirate(instrument: EnhancedPipette, volume, location, rate):
    local_config = instrument.config if hasattr(instrument, 'config') else config
    z = z_from_bottom(location, local_config.aspirate.bottom_clearance)
    location_text = stringify_location(location)
    text = pretty.format('Aspirating {volume:n} uL z={z:n} rate={rate:n} at {location}', volume=volume, location=location_text, rate=rate, z=z)
    return make_command(
        name=command_types.ASPIRATE,
        payload={
            'instrument': instrument,
            'volume': volume,
            'location': location,
            'rate': rate,
            'text': text
        }
    )

def command_dispense(instrument: EnhancedPipette, volume, location, rate):
    local_config = instrument.config if hasattr(instrument, 'config') else config
    z = z_from_bottom(location, local_config.dispense.bottom_clearance)
    location_text = stringify_location(location)
    text = pretty.format('Dispensing {volume:n} uL z={z:n} rate={rate:n} at {location}', volume=volume, location=location_text, rate=rate, z=z)
    return make_command(
        name=command_types.DISPENSE,
        payload={
            'instrument': instrument,
            'volume': volume,
            'location': location,
            'rate': rate,
            'text': text
        }
    )


opentrons.commands.aspirate = command_aspirate
opentrons.commands.dispense = command_dispense
# endregion

def format_log_msg(msg: str, prefix="***********", suffix=' ***********'):
    return "%s%s%s%s" % (prefix, '' if len(prefix) == 0 else ' ', msg, suffix)

def log(msg: str, prefix="***********", suffix=' ***********'):
    robot.comment(format_log_msg(msg, prefix=prefix, suffix=suffix))

def log_while(msg: str, func, prefix="***********", suffix=' ***********'):
    msg = format_log_msg(msg, prefix, suffix)
    opentrons.commands.do_publish(robot.broker, opentrons.commands.comment, f=log_while, when='before', res=None, meta=None, msg=msg)
    if func is not None:
        func()
    opentrons.commands.do_publish(robot.broker, opentrons.commands.comment, f=log_while, when='after', res=None, meta=None, msg=msg)

def info(msg):
    log(msg, prefix='info:', suffix='')

def info_while(msg, func):
    log_while(msg, func, prefix='info:', suffix='')

def warn(msg: str, prefix="***********", suffix=' ***********'):
    log(msg, prefix=prefix + " WARNING:", suffix=suffix)

def fatal(msg: str, prefix="***********", suffix=' ***********'):
    formatted = format_log_msg(msg, prefix=prefix + " FATAL ERROR:", suffix=suffix)
    warnings.warn(formatted, stacklevel=2)
    log(formatted, prefix='', suffix='')
    raise RuntimeError  # could do better

def silent_log(msg):
    pass
# endregion

########################################################################################################################
# Utilities
########################################################################################################################

# Returns a unique name for the given location. Must track in Opentrons-Analyze.
def get_location_path(location):
    result = getattr(location, 'location_path', None)
    if result is None:
        result = '/'.join(list(reversed([str(item)
                                       for item in location.get_trace(None)
                                       if str(item) is not None])))
        location.location_path = result
    return result

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

class Pretty(string.Formatter):
    def format_field(self, value, spec):
        if spec.endswith('n'):  # 'n' for number
            precision = 2
            if spec.startswith('.', 0, -1):
                precision = int(spec[1:-1])
            if is_finite_scalar(value):
                factor = 1
                for i in range(precision):
                    if is_close(value * factor, int(value * factor)):
                        precision = i
                        break
                    factor *= 10
                return "{:.{}f}".format(value, precision)
            elif hasattr(value, 'format'):
                return value.format(format_spec="{0:%s}" % spec, formatter=self)
            else:
                return str(value)
        return super().format_field(value, spec)

pretty = Pretty()

def verify_well_locations(well_list: List[Well], pipette: EnhancedPipette):
    picked_tip = False
    if not pipette.has_tip:
        pipette.pick_up_tip()
        picked_tip = True

    for well in well_list:
        move_to_loc = well.top()
        pipette.move_to(move_to_loc)
        #
        well_top_coords_absolute = well.top_coords_absolute()
        _, top_coords = unpack_location(well.top())
        _, move_to_coords = unpack_location(move_to_loc)
        intended_coords = well_top_coords_absolute + (move_to_coords - top_coords)
        tip_coords = pipette.tip_coords_absolute()
        #
        robot.pause(pretty.format('verify location: {0} in {1} loc={2} tip={3}', well.get_name(), well.parent.get_name(), intended_coords, tip_coords))

    if picked_tip:
        pipette.return_tip()  # we didn't dirty it, we can always re-use it todo: enhance return_tip() to adjust iterator so that next pick can pick up again

# endregion Other Enhancements Stuff

# endregion Enhancements
