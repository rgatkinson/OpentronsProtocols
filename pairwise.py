"""
@author Robert Atkinson
"""

from opentrons.commands.commands import stringify_location

metadata = {
    'protocolName': 'Pairwise Interaction: Dilute & Master & Plate',
    'author': 'Robert Atkinson <bob@theatkinsons.org>',
    'description': 'Study the interaction of two DNA strands'
}

# region Enhancements

import enum
import json
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
from opentrons.legacy_api.containers.placeable import unpack_location, Well, Placeable
from opentrons.trackers import pose_tracker
from opentrons.util.vector import Vector

########################################################################################################################
# Enhancements Configuration
########################################################################################################################

class Config(object):
    pass

config = Config()
config.enable_enhancements = True
config.trash_control = True
config.blow_out_rate_factor = 3.0
config.min_aspirate_factor_hack = 15.0
config.allow_blow_elision_default = True
config.allow_carryover_default = True

config.aspirate = Config()
config.aspirate.bottom_clearance = 1.0  # see Pipette._position_for_aspirate
config.aspirate.top_clearance = 3.5
config.aspirate.top_clearance_factor = 10.0
config.aspirate.extra_top_clearance_name = 'extra_aspirate_top_clearance'
config.aspirate.pre_wet = Config()
config.aspirate.pre_wet.default = True
config.aspirate.pre_wet.count = 3
config.aspirate.pre_wet.max_volume_fraction = 1  # https://github.com/Opentrons/opentrons/issues/2901 would pre-wet only 2/3, but why not everything?
config.aspirate.pre_wet.rate_func = lambda aspirate_rate: 1  # could instead just use the aspirate
config.aspirate.pause = Config()
config.aspirate.pause.ms_default = 500

config.dispense = Config()
config.dispense.bottom_clearance = 0.5  # see Pipette._position_for_dispense
config.dispense.top_clearance = 2.0
config.dispense.top_clearance_factor = 10.0
config.dispense.extra_top_clearance_name = 'extra_dispense_top_clearance'
config.dispense.full_dispense = Config()
config.dispense.full_dispense.default = False  # todo: should this be 'True'?

config.layered_mix = Config()
config.layered_mix.top_clearance = 1.0
config.layered_mix.top_clearance_factor = 10
config.layered_mix.aspirate_bottom_clearance = 1.0
config.layered_mix.aspirate_rate_factor = 4.0
config.layered_mix.dispense_rate_factor = 4.0
config.layered_mix.incr = 1.0
config.layered_mix.count = None  # so we default to using incr, not count
config.layered_mix.min_incr = 0.5
config.layered_mix.count_per_incr = 2
config.layered_mix.ms_pause = 750
config.layered_mix.keep_last_tip = False
config.layered_mix.initial_turnover = None
config.layered_mix.max_tip_cycles = None
config.layered_mix.max_tip_cycles_large = None

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
def note_liquid(location, name=None, initial_volume=None, min_volume=None, concentration=None):
    well, __ = unpack_location(location)
    assert isWell(well)
    if name is None:
        name = well.label
    else:
        well.label = name
    d = {'name': name, 'location': get_location_path(well)}
    if initial_volume is None and min_volume is not None:
        initial_volume = interval([min_volume, get_well_geometry(well).well_capacity])
    if initial_volume is not None:
        d['initial_volume'] = initial_volume
        get_well_volume(well).set_initial_volume(initial_volume)
    if concentration is not None:
        d['concentration'] = str(Concentration(concentration))
    serialized = json.dumps(d).replace("{", "{{").replace("}", "}}")  # runtime calls comment.format(...) on our comment; avoid issues therewith
    robot.comment('Liquid: %s' % serialized)

########################################################################################################################
# Well enhancements
########################################################################################################################

class WellVolume(object):

    #-------------------------------------------------------------------------------------------------------------------
    # Construction
    #-------------------------------------------------------------------------------------------------------------------

    def __init__(self, well=None):
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
            return get_well_geometry(self.well).min_aspiratable_volume

    #-------------------------------------------------------------------------------------------------------------------
    # Actions
    #-------------------------------------------------------------------------------------------------------------------

    def aspirate(self, volume):
        assert volume >= 0
        if not self.initial_volume_known:
            self.set_initial_volume(interval([volume,
                                              fpu.infinity if self.well is None else get_well_volume(self.well).capacity]))
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


def isWell(location):
    return isinstance(location, Well)

def get_well_volume(well):
    assert isWell(well)
    try:
        return well.contents
    except AttributeError:
        well.contents = WellVolume(well)
        return well.contents

# region Well Geometry
class WellGeometry(object):

    #-------------------------------------------------------------------------------------------------------------------
    # Construction
    #-------------------------------------------------------------------------------------------------------------------

    def __init__(self, well):
        self.well: Well = well

    #-------------------------------------------------------------------------------------------------------------------
    # Accessing
    #-------------------------------------------------------------------------------------------------------------------

    @property
    @abstractmethod
    def well_capacity(self):
        pass

    @property
    def min_aspiratable_volume(self):  # minimum volume we can aspirate from (i.e.: we leave at least this much behind)
        return 0

    @property
    def well_depth(self):
        return self.well.z_size()  # default to what Opentrons gave us

    #-------------------------------------------------------------------------------------------------------------------
    # Calculations
    #-------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def depth_from_volume(self, volume):  # best calc'n of depth from the given volume. may be an interval
        pass

    @abstractmethod
    def volume_from_depth(self, depth):
        pass

    def depth_from_volume_min(self, volume):  # lowest possible depth for the given volume
        vol = self.depth_from_volume(volume)
        if is_interval(vol):
            return vol.infimum
        else:
            return vol


class UnknownWellGeometry(WellGeometry):
    def __init__(self, well):
        super().__init__(well)

    def depth_from_volume(self, volume):
        return interval([0, self.well_depth])

    def volume_from_depth(self, depth):
        raise interval([0, self.well_capacity])

    @property
    def well_capacity(self):
        # noinspection PyBroadException
        return self.well.properties.get('total-liquid-volume', fpu.infinity)


# Calculated from Mathematica models. We use fittedIdtTube.
class IdtTubeWellGeometry(WellGeometry):
    def __init__(self, well):
        super().__init__(well)

    def depth_from_volume(self, volume):
        if volume <= 0.0:
            return 0.0
        if volume <= 67.1109:
            return 0.909568 * cube_root(volume)
        return 2.46419 + 0.0183591 * volume

    def volume_from_depth(self, depth):
        if depth <= 0:
            return 0
        if depth <= 3.69629:
            return 1.32891 * cube(depth)
        return -134.222 + 54.4688 * depth

    @property
    def well_capacity(self):
        return 2153.47

    @property
    def well_depth(self):
        return 42

    @property
    def min_aspiratable_volume(self):
        return 75  # a rough estimate, but seems functionally useful


# Calculated from Mathematica models, specifically modelBioRad3[]
class Biorad96WellPlateWellGeometry(WellGeometry):
    def __init__(self, well):
        super().__init__(well)

    def depth_from_volume(self, volume):
        if volume <= 0.0:
            return 0.0
        if volume <= 122.784:
            return -8.57618 + 3.10509 * cube_root(21.0698 + 1.34645 * volume)
        return 3.91689 + 0.0427095 * volume

    def volume_from_depth(self, depth):
        if depth <= 0.0:
            return 0.0
        if depth <= 9.16092:
            return depth * (5.47391 + (0.638269 + 0.0248078 * depth) * depth)
        return -91.7099 + 23.414 * depth

    @property
    def well_capacity(self):
        return 255.051

    @property
    def well_depth(self):
        return 14.81


# Calculated from Mathematica models. We use fittedEppendorf1$5ml
class Eppendorf1point5mlTubeGeometry(WellGeometry):
    def __init__(self, well):
        super().__init__(well)

    def depth_from_volume(self, volume):
        if volume <= 0:
            return 0
        if volume <= 0.550217:
            raise NotImplementedError
        if volume <= 575.88:
            return -13.8495 + 2.9248 * cube_root(157.009 + 2.14521 * volume)
        return -216.767 + 20.2694 * cube_root(1376.83 + 0.33533 * volume)

    def volume_from_depth(self, depth):
        if depth <= 0:
            return 0
        if depth < 1.96886:
            raise NotImplementedError
        if depth <= 18.8106:
            return -23.6979 + depth * (10.7209 + (0.774099 + 0.0186313 * depth) * depth)
        return -458.451 + depth * (50.4795 + (0.232875 + 0.000358103 * depth) * depth)

    @property
    def well_capacity(self):
        return 1801.76

    @property
    def well_depth(self):
        return 37.8


# Calculated from Mathematica models fitted to empirical depth vs volume measurements
class FalconTube15mlGeometry(WellGeometry):
    def __init__(self, well):
        super().__init__(well)

    def depth_from_volume(self, volume):
        if volume <= 0.0:
            return 0.0
        if volume <= 1232.34:
            return -4.60531 + 1.42955 * cube_root(33.4335 + 5.25971 * volume)
        return -803.774 + 27.1004 * cube_root(27390.9 + 0.738644 * volume)

    def volume_from_depth(self, depth):
        if depth <= 0:
            return 0.0
        if depth <= 22.0945:
            return depth * (4.14078 + (0.899131 + 0.0650793 * depth) * depth)
        return -1761.24 + depth * (131.833 + (0.164018 + 0.0000680198 * depth) * depth)

    @property
    def well_capacity(self):
        return 16202.8  # compare to 15000 in opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical

    @property
    def well_depth(self):
        return 118.07  # compare to 117.5 in opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical


def get_well_geometry(well):
    assert isWell(well)
    try:
        return well.geometry
    except AttributeError:
        well.geometry = UnknownWellGeometry(well)
        return well.geometry

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

    def __new__(cls, parentInst):
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
    def __init__(self, parentInst):
        self.prev_aspirated_location = None
        self.aspirate_params_hack = EnhancedPipette.AspirateParamsHack()
        self.dispense_params_hack = EnhancedPipette.DispenseParamsHack()
        self.tip_wetness = TipWetness.NONE
        self.mixes_in_progress = list()

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
    #   'allow_carryover': if true, we allow carry over of disposal_vol from one run of (asp, disp+) to the next
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
            # print('cur=%s index=%s step=%s' % (format_number(self.current_volume), i, step))

            aspirate = step.get('aspirate')
            dispense = step.get('dispense')

            if aspirate:
                # *always* record on aspirates so we can test has_disposal_vol on subsequent dispenses
                have_disposal_vol = self.has_disposal_vol(plan, i, step_info_map, **kwargs)

                # we might have carryover from a previous transfer.
                if self.current_volume > 0:
                    info(pretty.format('carried over {0:n} uL from prev operation', self.current_volume))

                if not seen_aspirate:
                    assert i == 0

                    if kwargs.get('pre_wet', None) and kwargs.get('mix_before', None):
                        warn("simultaneous use of 'pre_wet' and 'mix_before' is not tested")

                    if (kwargs.get('allow_carryover', config.allow_carryover_default) and config.enable_enhancements) and zeroify(self.current_volume) > 0:
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
                                info(pretty.format("carryover of {0:n} uL isn't for disposal", self.current_volume))
                                extra = self.current_volume
                        else:
                            # different locations; can't re-use
                            info('this aspirate is from location different than current pipette contents')
                            extra = self.current_volume
                        if zeroify(extra) > 0:
                            # quiet_log('blowing out carryover of %s uL' % format_number(self.current_volume))
                            self._blowout_during_transfer(loc=None, **kwargs)  # loc isn't actually used

                    elif zeroify(self.current_volume) > 0:
                        info(pretty.format('blowing out unexpected carryover of {0:n} uL', self.current_volume))
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
                kwargs[self.aspirate_params_hack.pre_wet_during_transfer_kw] = not not kwargs.get('pre_wet', config.aspirate.pre_wet.default)
                kwargs[self.aspirate_params_hack.ms_pause_during_transfer_kw] = kwargs.get('pause', config.aspirate.pause.ms_default)
                self._aspirate_during_transfer(aspirate['volume'], aspirate['location'], **kwargs)

            if dispense:
                if self.current_volume < dispense['volume']:
                    warn(pretty.format('current {0:n} uL will truncate dispense of {1:n} uL', self.current_volume, dispense['volume']))

                can_full_dispense = self.current_volume - dispense['volume'] <= 0
                kwargs[self.dispense_params_hack.full_dispense_during_transfer_kw] = not not (kwargs.get('full_dispense', config.dispense.full_dispense.default) and can_full_dispense)
                self._dispense_during_transfer(dispense['volume'], dispense['location'], **kwargs)

                do_touch = touch_tip or touch_tip is 0
                is_last_step = step is plan[-1]
                if is_last_step or plan[i + 1].get('aspirate'):
                    do_drop = not is_last_step or not (kwargs.get('keep_last_tip', False) and config.enable_enhancements)
                    # original always blew here. there are several reasons we could still be forced to blow
                    do_blow = not is_distribute  # other modes (are there any?) we're not sure about
                    do_blow = do_blow or kwargs.get('blow_out', False)  # for compatibility
                    do_blow = do_blow or do_touch  # for compatibility
                    do_blow = do_blow or not (kwargs.get('allow_blow_elision', config.allow_blow_elision_default) and config.enable_enhancements)
                    if not do_blow:
                        if is_last_step:
                            if self.current_volume > 0:
                                if not (kwargs.get('allow_carryover', config.allow_carryover_default) and config.enable_enhancements):
                                    do_blow = True
                                elif self.current_volume > kwargs.get('disposal_vol', 0):
                                    warn(pretty.format('carried over {0:n} uL to next operation', self.current_volume))
                                else:
                                    info(pretty.format('carried over {0:n} uL to next operation', self.current_volume))
                        else:
                            # if we can, account for any carryover in the next aspirate
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
                                    # Next aspirate doesn't *want* our carryover, so we don't reduce his
                                    # volume. But it's harmless to just leave the carryover present; might
                                    # be useful down the line
                                    pass
                            else:
                                pass  # currently empty
                    if do_blow:
                        self._blowout_during_transfer(dispense['location'], **kwargs)
                    if do_touch:
                        self.touch_tip(touch_tip)
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

    def aspirate(self, volume=None, location=None, rate=1.0, pre_wet=None, pause=None):
        if not helpers.is_number(volume):  # recapitulate super
            if volume and not location:
                location = volume
            volume = self._working_volume - self.current_volume
        location = location if location else self.previous_placeable
        well, _ = unpack_location(location)

        current_well_volume = get_well_volume(well).current_volume_min
        needed_well_volume = get_well_geometry(well).min_aspiratable_volume + volume;
        if current_well_volume < needed_well_volume:
            msg = pretty.format('aspirating too much from well={0} have={1:n} need={2:n}', well.get_name(), current_well_volume, needed_well_volume)
            warn(msg)

        if pre_wet is None:
            pre_wet = self.aspirate_params_hack.pre_wet_during_transfer
        if pre_wet is None:
            pre_wet = config.aspirate.pre_wet.default
        if pre_wet and config.enable_enhancements:
            if self.tip_wetness is TipWetness.DRY:
                pre_wet_volume = min(
                    self.max_volume * config.aspirate.pre_wet.max_volume_fraction,
                    max(volume, get_well_volume(well).available_volume_min))
                pre_wet_rate = config.aspirate.pre_wet.rate_func(rate)
                self.tip_wetness = TipWetness.WETTING
                def do_pre_wet():
                    for i in range(config.aspirate.pre_wet.count):
                        self.aspirate(volume=pre_wet_volume, location=location, rate=pre_wet_rate, pre_wet=False)
                        self.dispense(volume=pre_wet_volume, location=location, rate=pre_wet_rate, full_dispense=(i+1 == config.aspirate.pre_wet.count))
                info_while(pretty.format('prewetting tip in well {0} vol={1:n}', well.get_name(), pre_wet_volume), do_pre_wet)
                self.tip_wetness = TipWetness.WET

        location = self._adjust_location_to_liquid_top(location=location, aspirate_volume=volume,
                                                       clearances=config.aspirate,
                                                       extra_clearance=getattr(well, config.aspirate.extra_top_clearance_name, 0))
        super().aspirate(volume=volume, location=location, rate=rate)

        # if we're asked to, pause after aspiration to let liquid rise
        if pause is None:
            pause = self.aspirate_params_hack.ms_pause_during_transfer
        if pause is None:
            pause = config.aspirate.pause.ms_default
        if config.enable_enhancements and pause and not self.is_mix_in_progress():
            self.delay(pause / 1000.0)

        # track volume todo: what if we're doing an air gap
        well, __ = unpack_location(location)
        get_well_volume(well).aspirate(volume)
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
                                                       clearances=config.dispense,
                                                       extra_clearance=getattr(well, config.dispense.extra_top_clearance_name, 0))
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
        get_well_volume(well).dispense(volume)

    def _dispense_during_transfer(self, vol, loc, **kwargs):
        assert kwargs.get(self.dispense_params_hack.full_dispense_during_transfer_kw) is not None
        self.dispense_params_hack.full_dispense_during_transfer = kwargs.get(self.dispense_params_hack.full_dispense_during_transfer_kw)
        super()._dispense_during_transfer(vol, loc, **kwargs)  # might 'mix_after' todo: is that ok? probably: we'd just do full_dispense on all of those too?
        self.dispense_params_hack.full_dispense_during_transfer = False

    def _dispense_plunger_position(self, ul):
        mm_from_vol = super()._dispense_plunger_position(ul)  # retrieve position historically used
        if config.enable_enhancements and (self.dispense_params_hack.full_dispense_from_dispense or self.dispense_params_hack.full_dispense_during_transfer):
            mm_from_blow = self._get_plunger_position('blow_out')
            info(pretty.format('dispensing to mm={0:n} instead of mm={1:n}', mm_from_blow, mm_from_vol))
            self.dispense_params_hack.fully_dispensed = True
            return mm_from_blow
        else:
            self.dispense_params_hack.fully_dispensed = False
            return mm_from_vol

    def _adjust_location_to_liquid_top(self, location=None, aspirate_volume=None, clearances=None, extra_clearance=0, allow_above=False):
        if isinstance(location, Placeable):
            well = location; assert isWell(well)
            well_vol = get_well_volume(well).current_volume_min
            well_depth = get_well_geometry(well).depth_from_volume_min(well_vol if aspirate_volume is None else well_vol - aspirate_volume)
            z = well_depth - self._top_clearance(well, well_depth,
                                                 clearance=(0 if clearances is None else clearances.top_clearance) + extra_clearance,
                                                 factor=1 if clearances is None else clearances.top_clearance_factor)
            if clearances is not None:
                z = max(z, clearances.bottom_clearance)
            if not allow_above:
                z = min(z, well.z_size())
            result = well.bottom(z)
        else:
            result = location
        assert isinstance(result, tuple)
        return result

    #-------------------------------------------------------------------------------------------------------------------
    # Tip Management
    #-------------------------------------------------------------------------------------------------------------------

    def pick_up_tip(self, location=None, presses=None, increment=None):
        result = super().pick_up_tip(location, presses, increment)
        self.tip_wetness = TipWetness.DRY
        return result

    def drop_tip(self, location=None, home_after=True):
        result = super().drop_tip(location, home_after)
        self.tip_wetness = TipWetness.NONE
        return result

    def done_tip(self):  # a handy little utility that looks at config.trash_control
        if self.has_tip:
            if self.current_volume > 0:
                info(pretty.format('{0} has {1:n} uL remaining', self.name, self.current_volume))
            if config.trash_control:
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
            local_keep_last_tip = keep_last_tip if keep_last_tip is not None else config.layered_mix.keep_last_tip

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

    def _top_clearance(self, well, depth, clearance, factor):
        if clearance < 0:
            return clearance
        else:
            return max(clearance, depth / factor)

    def _layered_mix_one(self, well, msg, **kwargs):
        def fetch(name, default=None):
            if default is None:
                default = getattr(config.layered_mix, name)
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

        well_vol = get_well_volume(well).current_volume_min
        well_depth = get_well_geometry(well).depth_from_volume(well_vol)
        well_depth_after_asp = get_well_geometry(well).depth_from_volume(well_vol - volume)
        msg = pretty.format("{0:s} well='{1:s}' cur_vol={2:n} well_depth={3:n} after_aspirate={4:n}", msg, well.get_name(), well_vol, well_depth, well_depth_after_asp)

        def do_one():
            count_ = count
            y_min = y = config.layered_mix.aspirate_bottom_clearance
            y_max = well_depth_after_asp - self._top_clearance(well, well_depth_after_asp, clearance=config.layered_mix.top_clearance, factor=config.layered_mix.top_clearance_factor)
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
                for i in range(count_):
                    tip_cycles += 1
                    need_new_tip = tip_cycles >= max_tip_cycles
                    full_dispense = need_new_tip or (not do_layer(y + y_incr) and i == count_ - 1)
                    self.aspirate(volume, well.bottom(y), rate=fetch('aspirate_rate', config.layered_mix.aspirate_rate_factor), pre_wet=pre_wet)
                    self.dispense(volume, well.bottom(y_max), rate=fetch('dispense_rate', config.layered_mix.dispense_rate_factor), full_dispense=full_dispense)
                    if need_new_tip:
                        self.done_tip()
                        tip_cycles = 0
                #
                y += y_incr
                first = False

        info_while(msg, do_one)

    #-------------------------------------------------------------------------------------------------------------------
    # Movement
    #-------------------------------------------------------------------------------------------------------------------

    def tip_coords_absolute(self):
        xyz = pose_tracker.absolute(self.robot.poses, self)
        return Vector(xyz)

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

def command_aspirate(instrument, volume, location, rate):
    z = z_from_bottom(location, config.aspirate.bottom_clearance)
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

def command_dispense(instrument, volume, location, rate):
    z = z_from_bottom(location, config.dispense.bottom_clearance)
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

########################################################################################################################
# Logging
########################################################################################################################

def _format_log_msg(msg: str, prefix="***********", suffix=' ***********'):
    return "%s%s%s%s" % (prefix, '' if len(prefix) == 0 else ' ', msg, suffix)

def log(msg: str, prefix="***********", suffix=' ***********'):
    robot.comment(_format_log_msg(msg, prefix=prefix, suffix=suffix))

def log_while(msg: str, func, prefix="***********", suffix=' ***********'):
    msg = _format_log_msg(msg, prefix, suffix)
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
    formatted = _format_log_msg(msg, prefix=prefix + " FATAL ERROR:", suffix=suffix)
    warnings.warn(formatted, stacklevel=2)
    log(formatted, prefix='', suffix='')
    raise RuntimeError  # could do better

def silent_log(msg):
    pass

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

def cube_root(value):
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

########################################################################################################################
# Configurable protocol parameters
########################################################################################################################

# Volumes of master mix ingredients. These are minimums in each tube.
buffer_volumes = [1000, 1000]       # A1, A2, etc in screwcap rack
evagreen_volumes = [600]            # B1, B2, etc in screwcap rack

strand_a_conc = '10uM'  # '8.820 uM'  # Note: we'll use more Strand A than Strand B because of disposal_volumes
strand_b_conc = '10uM'  # '9.117 uM'
strand_a_min_vol = 600  # set as best one can
strand_b_min_vol = 600  # set as best one can

# Tip usage
p10_start_tip = 'A9'
p50_start_tip = 'A9'
config.trash_control = True

# Diluting each strand
strand_dilution_factor = 25.0 / 9.0  # per Excel worksheet
strand_dilution_vol = 1225

# Master mix, values per Excel worksheet
mm_overhead_factor = 1.0375
master_mix_buffer_vol = 1612.8 * mm_overhead_factor
master_mix_evagreen_vol = 403.2 * mm_overhead_factor
master_mix_common_water_vol = 672 * mm_overhead_factor
master_mix_vol = master_mix_buffer_vol + master_mix_evagreen_vol + master_mix_common_water_vol

# Define the volumes of diluted strands we will use
strand_volumes = [0, 2, 5, 8, 12, 16, 20, 28]
num_replicates = 3
columns_per_plate = 12
rows_per_plate = 8
per_well_water_volumes = [
    [56, 54, 51, 48],
    [54, 52, 49, 46],
    [51, 49, 46, 43],
    [48, 46, 43, 40],
    [32, 28, 24, 16],
    [28, 24, 20, 12],
    [24, 20, 16, 8],
    [16, 12, 8, 0]]
assert len(per_well_water_volumes) == rows_per_plate
assert len(per_well_water_volumes[0]) * num_replicates == columns_per_plate


########################################################################################################################
## Protocol
########################################################################################################################


# compute derived constants
strand_dilution_source_vol = strand_dilution_vol / strand_dilution_factor
strand_dilution_water_vol = strand_dilution_vol - strand_dilution_source_vol

########################################################################################################################
# Labware
########################################################################################################################

# Configure the tips
tips300a = labware.load('opentrons_96_tiprack_300ul', 1)
tips10 = labware.load('opentrons_96_tiprack_10ul', 4)
tips300b = labware.load('opentrons_96_tiprack_300ul', 7)

# Configure the pipettes.
p10 = EnhancedPipette(instruments.P10_Single(mount='left', tip_racks=[tips10]))
p50 = EnhancedPipette(instruments.P50_Single(mount='right', tip_racks=[tips300a, tips300b]))

# Blow out faster than default in an attempt to avoid hanging droplets on the pipettes after blowout
p10.set_flow_rate(blow_out=p10.get_flow_rates()['blow_out'] * config.blow_out_rate_factor)
p50.set_flow_rate(blow_out=p50.get_flow_rates()['blow_out'] * config.blow_out_rate_factor)

# Control tip usage
p10.start_at_tip(tips10[p10_start_tip])
p50.start_at_tip(tips300a[p50_start_tip])

# All the labware containers
temp_slot = 10
temp_module = modules.load('tempdeck', temp_slot)
screwcap_rack = labware.load('opentrons_24_aluminumblock_generic_2ml_screwcap', temp_slot, label='screwcap_rack', share=True)
eppendorf_1_5_rack = labware.load('opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap', 2, label='eppendorf_1_5_rack')
falcon_rack = labware.load('opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical', 5, label='falcon_rack')
plate = labware.load('biorad_96_wellplate_200ul_pcr', 3, label='plate')
trough = labware.load('usascientific_12_reservoir_22ml', 6, label='trough')

# Name specific places in the labware containers
water = trough['A1']
buffers = list(zip(screwcap_rack.rows(0), buffer_volumes))
evagreens = list(zip(screwcap_rack.rows(1), evagreen_volumes))
strand_a = eppendorf_1_5_rack['A1']
strand_b = eppendorf_1_5_rack['B1']
diluted_strand_a = eppendorf_1_5_rack['A6']
diluted_strand_b = eppendorf_1_5_rack['B6']
master_mix = falcon_rack['A1']  # note: this needs tape around it's mid-section to keep it in the holder!

# Define geometries
for well, __ in buffers:
    well.geometry = IdtTubeWellGeometry(well)
for well, __ in evagreens:
    well.geometry = IdtTubeWellGeometry(well)
strand_a.geometry = Eppendorf1point5mlTubeGeometry(strand_a)
strand_b.geometry = Eppendorf1point5mlTubeGeometry(strand_b)
diluted_strand_a.geometry = Eppendorf1point5mlTubeGeometry(diluted_strand_a)
diluted_strand_b.geometry = Eppendorf1point5mlTubeGeometry(diluted_strand_b)
master_mix.geometry = FalconTube15mlGeometry(master_mix)
for well in plate.wells():
    well.geometry = Biorad96WellPlateWellGeometry(well)

# Remember initial liquid names and volumes
log('Liquid Names')
note_liquid(location=water, name='Water', min_volume=7000)  # volume is rough guess
assert strand_a_min_vol >= strand_dilution_source_vol + get_well_geometry(strand_a).min_aspiratable_volume
assert strand_b_min_vol >= strand_dilution_source_vol + get_well_geometry(strand_b).min_aspiratable_volume
note_liquid(location=strand_a, name='StrandA', concentration=strand_a_conc, min_volume=strand_a_min_vol)  # i.e.: we have enough, just not specified how much
note_liquid(location=strand_b, name='StrandB', concentration=strand_b_conc, min_volume=strand_b_min_vol)  # ditto
note_liquid(location=diluted_strand_a, name='Diluted StrandA')
note_liquid(location=diluted_strand_b, name='Diluted StrandB')
note_liquid(location=master_mix, name='Master Mix')
for buffer in buffers:
    note_liquid(location=buffer[0], name='Buffer', initial_volume=buffer[1], concentration='5x')
for evagreen in evagreens:
    note_liquid(location=evagreen[0], name='Evagreen', initial_volume=evagreen[1], concentration='20x')

# Clean up namespace
del well

########################################################################################################################
# Well & Pipettes
########################################################################################################################

num_samples_per_row = columns_per_plate // num_replicates

# Into which wells should we place the n'th sample size of strand A
def calculateStrandAWells(iSample: int) -> List[types.Location]:
    row_first = 0 if iSample < num_samples_per_row else num_samples_per_row
    col_first = (num_replicates * iSample) % columns_per_plate
    result = []
    for row in range(row_first, row_first + min(num_samples_per_row, len(strand_volumes))):
        for col in range(col_first, col_first + num_replicates):
            result.append(plate.rows(row).wells(col))
    return result


# Into which wells should we place the n'th sample size of strand B
def calculateStrandBWells(iSample: int) -> List[types.Location]:
    if iSample < num_samples_per_row:
        col_max = num_replicates * (len(strand_volumes) if len(strand_volumes) < num_samples_per_row else num_samples_per_row)
    else:
        col_max = num_replicates * (0 if len(strand_volumes) < num_samples_per_row else len(strand_volumes) - num_samples_per_row)
    result = []
    for col in range(0, col_max):
        result.append(plate.rows(iSample).wells(col))
    return result


# What wells are at all used here?
def usedWells() -> List[types.Location]:
    result = []
    for n in range(0, len(strand_volumes)):
        result.extend(calculateStrandAWells(n))
    return result


# Figuring out what pipettes should pipette what volumes
p10_max_vol = 10
p50_min_vol = 5
def usesP10(queriedVol, count, allow_zero):
    return (allow_zero or 0 < queriedVol) and (queriedVol < p50_min_vol or queriedVol * count <= p10_max_vol)


########################################################################################################################
# Making master mix and diluting strands
########################################################################################################################

def diluteStrands():
    p50.layered_mix([strand_a])
    p50.layered_mix([strand_b])

    # Create dilutions of strands
    log('Moving water for diluting Strands A and B')
    p50.transfer(strand_dilution_water_vol, water, [diluted_strand_a, diluted_strand_b],
                 new_tip='once',  # can reuse for all diluent dispensing since dest tubes are initially empty
                 trash=config.trash_control
                 )
    log('Diluting Strand A')
    p50.transfer(strand_dilution_source_vol, strand_a, diluted_strand_a, trash=config.trash_control, keep_last_tip=True)
    p50.layered_mix([diluted_strand_a])

    log('Diluting Strand B')
    p50.transfer(strand_dilution_source_vol, strand_b, diluted_strand_b, trash=config.trash_control, keep_last_tip=True)
    p50.layered_mix([diluted_strand_b])


def createMasterMix():
    # Buffer was just unfrozen. Mix to ensure uniformity. EvaGreen doesn't freeze, no need to mix
    p50.layered_mix([buffer for buffer, __ in buffers], incr=2)

    # transfer from multiple source wells, each with a current defined volume
    def transfer_multiple(msg, xfer_vol_remaining, tubes, dest, new_tip, *args, **kwargs):
        tube_index = 0
        cur_well = None
        cur_vol = 0
        min_vol = 0
        while xfer_vol_remaining > 0:
            if xfer_vol_remaining < p50_min_vol:
                warn("remaining transfer volume of %f too small; ignored" % xfer_vol_remaining)
                return
            # advance to next tube if there's not enough in this tube
            while cur_well is None or cur_vol <= min_vol:
                if tube_index >= len(tubes):
                    fatal('%s: more reagent needed' % msg)
                cur_well = tubes[tube_index][0]
                cur_vol = tubes[tube_index][1]
                min_vol = max(p50_min_vol,
                              cur_vol / config.min_aspirate_factor_hack,  # tolerance is proportional to specification of volume. can probably make better guess
                              get_well_geometry(cur_well).min_aspiratable_volume)
                tube_index = tube_index + 1
            this_vol = min(xfer_vol_remaining, cur_vol - min_vol)
            assert this_vol >= p50_min_vol  # TODO: is this always the case?
            log('%s: xfer %f from %s in %s to %s in %s' % (msg, this_vol, cur_well, cur_well.parent, dest, dest.parent))
            p50.transfer(this_vol, cur_well, dest, trash=config.trash_control, new_tip=new_tip, **kwargs)
            xfer_vol_remaining -= this_vol
            cur_vol -= this_vol

    def mix_master_mix():
        log('Mixing Master Mix')
        p50.layered_mix([master_mix], incr=2, initial_turnover=master_mix_evagreen_vol * 1.2, max_tip_cycles=config.layered_mix.max_tip_cycles_large)

    log('Creating Master Mix: Water')
    p50.transfer(master_mix_common_water_vol, water, master_mix, trash=config.trash_control)

    log('Creating Master Mix: Buffer')
    transfer_multiple('Creating Master Mix: Buffer', master_mix_buffer_vol, buffers, master_mix, new_tip='once', keep_last_tip=True)  # 'once' because we've only got water & buffer in context
    p50.done_tip()  # EvaGreen needs a new tip

    log('Creating Master Mix: EvaGreen')
    transfer_multiple('Creating Master Mix: EvaGreen', master_mix_evagreen_vol, evagreens, master_mix, new_tip='always', keep_last_tip=True)  # 'always' to avoid contaminating the Evagreen source w/ buffer

    mix_master_mix()


########################################################################################################################
# Plating
########################################################################################################################

def plateMasterMix():
    log('Plating Master Mix')
    master_mix_per_well = 28
    p50.transfer(master_mix_per_well, master_mix, usedWells(),
                 new_tip='once',
                 trash=config.trash_control,
                 full_dispense=True)

def platePerWellWater():
    log('Plating per-well water')
    # Plate per-well water. We save tips by being happy to pollute our water trough with a bit of master mix.
    # We begin by flattening per_well_water_volumes into a column-major array
    water_volumes = [0] * (columns_per_plate * rows_per_plate)
    for iRow in range(rows_per_plate):
        for iCol in range(len(per_well_water_volumes[iRow])):
            volume = per_well_water_volumes[iRow][iCol]
            for iReplicate in range(num_replicates):
                index = (iCol * num_replicates + iReplicate) * rows_per_plate + iRow
                water_volumes[index] = volume

    p50.transfer(water_volumes, water, plate.wells(),
                 new_tip='once',
                 trash=config.trash_control,
                 full_dispense=True)

def plateStrandA():
    # Plate strand A
    # All plate wells at this point only have water and master mix, so we can't get cross-plate-well
    # contamination. We only need to worry about contaminating the Strand A source, which we accomplish
    # by using new_tip='always'. Update: we don't worry about that pollution, that source is disposable.
    # So we can minimize tip usage.
    log('Plating Strand A')
    p10.pick_up_tip()
    p50.pick_up_tip()
    for iVolume in range(0, len(strand_volumes)):
        dest_wells = calculateStrandAWells(iVolume)
        volume = strand_volumes[iVolume]
        if volume == 0: continue
        if usesP10(volume, len(dest_wells), allow_zero=False):
            p = p10
        else:
            p = p50
        log('Plating Strand A: volume %d with %s' % (volume, p.name))
        volumes = [volume] * len(dest_wells)
        p.transfer(volumes, diluted_strand_a, dest_wells,
                     new_tip='never',
                     trash=config.trash_control,
                     full_dispense=True)
    p10.done_tip()
    p50.done_tip()

def mix_plate_well(well, keep_last_tip=False):
    p50.layered_mix([well], incr=0.75, keep_last_tip=keep_last_tip)

def plateStrandBAndMix():
    # Plate strand B and mix
    # Mixing always needs the p50, but plating may need either; optimize tip usage
    log('Plating Strand B')
    mixed_wells = set()
    for iVolume in range(0, len(strand_volumes)):
        dest_wells = calculateStrandBWells(iVolume)
        volume = strand_volumes[iVolume]
        if usesP10(volume, len(dest_wells), allow_zero=True):
            p = p10
        else:
            p = p50

        # We can't use distribute here as we need to avoid cross contamination from plate well to plate well
        for well in dest_wells:
            if volume != 0:
                log("Plating Strand B: well='%s' vol=%d pipette=%s" % (well.get_name(), volume, p.name))
                p.pick_up_tip()
                p.transfer(volume, diluted_strand_b, well, new_tip='never', full_dispense=True)
            if p is p50:
                mix_plate_well(well, keep_last_tip=True)
                mixed_wells.add(well)
            p.done_tip()

    for well in plate.wells():
        if well not in mixed_wells:
            mix_plate_well(well)

def plateEverythingAndMix():
    plateMasterMix()
    platePerWellWater()
    plateStrandA()
    plateStrandBAndMix()


########################################################################################################################
# Off to the races
########################################################################################################################

wells_to_verify = [master_mix, strand_a, strand_b, diluted_strand_a, diluted_strand_b, plate.wells('A1'), plate.wells('A12'), plate.wells('H1'), plate.wells('H12')]
verify_well_locations(wells_to_verify, p50)
verify_well_locations(wells_to_verify, p10)

diluteStrands()
createMasterMix()
plateEverythingAndMix()