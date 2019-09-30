"""
@author Robert Atkinson
"""

import cmath
import json
import numpy
import string
import warnings
from abc import abstractmethod
from functools import wraps
from numbers import Number
from typing import List

import opentrons
from opentrons import labware, instruments, robot, modules, types
from opentrons.commands.commands import stringify_location, make_command, command_types
from opentrons.helpers import helpers
from opentrons.legacy_api.instruments import Pipette
from opentrons.legacy_api.instruments.pipette import SHAKE_OFF_TIPS_DISTANCE, SHAKE_OFF_TIPS_SPEED
from opentrons.legacy_api.containers.placeable import unpack_location, Well, Placeable

metadata = {
    'protocolName': 'Pairwise Interaction: Dilute & Master',
    'author': 'Robert Atkinson <bob@theatkinsons.org>',
    'description': 'Study the interaction of two DNA strands'
}

########################################################################################################################
# Configurable parameters
########################################################################################################################

# Volumes of master mix ingredients
buffer_volumes = [1000, 1000]       # A1, A2, etc in screwcap rack
evagreen_volumes = [720]           # B1, B2, etc in screwcap rack

# Tip usage
p10_start_tip = 'A4'
p50_start_tip = 'A6'
trash_control = False

# Diluting each strand
strand_dilution_factor = 25.0 / 9.0  # per Excel worksheet
strand_dilution_vol = 1225

# Master mix
master_mix_buffer_vol = 1693.44
master_mix_evagreen_vol = 423.36
master_mix_common_water_vol = 705.6
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


# Optimization Control (needs cleaning up)
allow_blow_elision = True
allow_carryover = allow_blow_elision


class Config(object):
    pass

config = Config()
config.blow_out_rate_factor = 3.0
config.aspirate = Config()
config.aspirate.bottom_clearance = 1.0  # see Pipette._position_for_aspirate
config.aspirate.top_clearance = 3.5
config.aspirate.top_clearance_factor = 10.0
config.dispense = Config()
config.dispense.bottom_clearance = 0.5  # see Pipette._position_for_dispense
config.dispense.top_clearance = 3.5  # was 1.0, but we were missing top of master mix
config.dispense.top_clearance_factor = 10.0
config.simple_mix = Config()
config.simple_mix.count = 6
config.layered_mix = Config()
config.layered_mix.top_clearance = 1.0
config.layered_mix.top_clearance_factor = 10
config.layered_mix.aspirate_bottom_clearance = 1.0
config.layered_mix.aspirate_rate_factor = 3.0
config.layered_mix.dispense_rate_factor = 3.0
config.layered_mix.incr = 1.0
config.layered_mix.count = None  # so we default to using incr, not count
config.layered_mix.min_incr = 0.5
config.layered_mix.count_per_incr = 2
config.layered_mix.delay = 750
config.layered_mix.drop_tip = True
config.layered_mix.initial_turnover = None
config.layered_mix.max_tip_cycles = None
config.layered_mix.max_tip_cycles_large = 12


########################################################################################################################
########################################################################################################################
##                                                                                                                    ##
## Extensions : this section can be reused across protocols                                                           ##
##                                                                                                                    ##
########################################################################################################################
########################################################################################################################

# region Extensions

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

    @staticmethod
    def isnan(x):
        return x != x

    def is_infinite(self, x):
        return x == self.infinity or x == -self.infinity

    def is_finite(self, x):
        return not self.isnan(x) and not self.is_infinite(x)

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
        if self.isnan(x):
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

    @staticmethod
    def isinteger(n):
        return isinstance(n, int)

    def power_rn(self, x, n):
        # Raise x to the n-th power (with n positive integer), rounded to nearest.
        assert self.isinteger(n) and n >= 0
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
# Well enhancements
########################################################################################################################

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
    def current_volume_min(self):  # (scalar) minimum known to be currently occupied
        vol = self.current_volume
        if isinstance(vol, interval):
            return vol.infimum
        else:
            return vol

    @property
    def min_volume(self):  # minimum historically seen
        return self.initial_volume + self.min_delta

    @property
    def max_volume(self):
        return self.initial_volume + self.max_delta

    def aspirate(self, volume):
        assert volume >= 0
        if not self.initial_volume_known:
            self.set_initial_volume(interval([volume, get_well_volume(well).capacity]))
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

# Must keep in sync with Opentrons-Analyze controller.note_liquid_name
def note_liquid(location, name=None, initial_volume=None, min_volume=None):
    well, __ = unpack_location(location)
    assert isWell(well)
    if name is None:
        name = well.label
    else:
        well.label = name
    d = {'name': name, 'location': get_location_path(well)}
    if initial_volume is None and min_volume is not None:
        initial_volume = interval([min_volume, get_well_geometry(well).capacity])
    if initial_volume is not None:
        d['initial_volume'] = initial_volume
        get_well_volume(well).set_initial_volume(initial_volume)
    serialized = json.dumps(d).replace("{", "{{").replace("}", "}}")  # runtime calls comment.format(...) on our comment; avoid issues therewith
    robot.comment('Liquid: %s' % serialized)

########################################################################################################################

# region Well Geometry
class WellGeometry(object):
    def __init__(self, well):
        self.well = well

    @abstractmethod
    def depth_from_volume(self, volume):
        pass

    @property
    @abstractmethod
    def capacity(self):
        pass

    def min_depth_from_volume(self, volume):
        vol = self.depth_from_volume(volume)
        if isinstance(vol, interval):
            return vol.infimum
        else:
            return vol


class UnknownWellGeometry(WellGeometry):
    def __init__(self, well):
        super().__init__(well)

    def depth_from_volume(self, volume):
        return interval([0, self.capacity])

    @property
    def capacity(self):
        # noinspection PyBroadException
        return self.well.properties.get('total-liquid-volume', fpu.infinity)


class IdtTubeWellGeometry(WellGeometry):
    def __init__(self, well):
        super().__init__(well)

    def depth_from_volume(self, volume):
        # Calculated from Mathematica models
        if volume <= 0.0:
            return 0.0
        if volume <= 57.8523:
            return 0.827389 * cube_root(volume)
        return 3.2 - 0.0184378 * (57.8523 - volume)

    @property
    def capacity(self):
        return 2266.91


class Biorad96WellPlateWellGeometry(WellGeometry):
    def __init__(self, well):
        super().__init__(well)

    def depth_from_volume(self, volume):
        # Calculated from Mathematica models
        if volume <= 0.0:
            return 0.0
        if volume <= 60.7779:
            return -13.7243 + 4.24819 * cube_root(33.7175 + 1.34645 * volume)
        return 14.66 - 0.0427095 * (196.488 - volume)

    @property
    def capacity(self):
        return 200.0


class Eppendorf1point5mlTubeGeometry(WellGeometry):
    def __init__(self, well):
        super().__init__(well)

    def depth_from_volume(self, volume):
        # Calculated from Mathematica models
        if volume <= 12.2145:
            i = complex(0, 1)
            term = cube_root(36.6435 - 3. * volume + 1.73205 * cmath.sqrt(-73.2871 * volume + 3. * volume * volume))
            result = 1.8 - (2.98934 - 5.17768 * i) / term - (0.270963 + 0.469322 * i) * term
            assert isinstance(result, complex)
            return result.real
        if volume <= 445.995:
            return -8.22353 + 2.2996 * cube_root(53.0712 + 2.43507 * volume)
        return -564. + 49.1204 * cube_root(1580.62 + 0.143239 * volume)

    @property
    def capacity(self):
        return 1688.61


class FalconTube15mlGeometry(WellGeometry):
    def __init__(self, well):
        super().__init__(well)

    def depth_from_volume(self, volume):
        # Calculated from Mathematica models
        if volume <= 0.0686291:
            return 0.0  # not correct, but not worth it right now to do correct value
        if volume <= 874.146:
            return -0.758658 + 1.23996 * cube_root(0.267715 + 5.69138 * volume)
        return -360.788 + 13.8562 * cube_root(19665.7 + 1.32258 * volume)

    @property
    def capacity(self):
        return 13756.5


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

class MyPipette(Pipette):
    def __new__(cls, parentInst):
        parentInst.__class__ = MyPipette
        return parentInst

    # noinspection PyMissingConstructor
    def __init__(self, parentInst):
        self.prev_aspirated_location = None
        pass

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

    def _get_next_ops(self, plan, step_index, max_count):
        result = []
        while step_index < len(plan) and len(result) < max_count:
            step = plan[step_index]
            if step.get('aspirate'):
                result.append('aspirate')
            if step.get('dispense'):
                result.append('dispense')
            step_index += 1
        return result

    def has_disposal_vol(self, plan, step_index, **kwargs):
        if kwargs.get('mode', 'transfer') != 'distribute':
            return False
        if kwargs.get('disposal_vol', 0) <= 0:
            return False
        check_has_disposal_vol = False
        next_steps = self._get_next_ops(plan, step_index, 3)
        assert next_steps[0] == 'aspirate'
        if len(next_steps) >= 2:
            if next_steps[1] == 'dispense':
                if len(next_steps) >= 3:
                    if next_steps[2] == 'dispense':
                        check_has_disposal_vol = True
                    else:
                        silent_log('aspirate-dispense-aspirate')
                else:
                    info('aspirate-dispense is entire remaining plan')
            else:
                info('unexpected aspirate-aspirate sequence')
        return check_has_disposal_vol

    # Copied and overridden
    # New kw args:
    #   'retain_tip': if true, then tip is not dropped at end of transfer
    #   'allow_carryover'
    #   'allow_blow_elision'
    def _run_transfer_plan(self, tips, plan, **kwargs):
        air_gap = kwargs.get('air_gap', 0)
        touch_tip = kwargs.get('touch_tip', False)
        is_distribute = kwargs.get('mode', 'transfer') == 'distribute'

        total_transfers = len(plan)
        seen_aspirate = False
        assert len(plan) == 0 or plan[0].get('aspirate')  # first step must be an aspirate

        for step_index, step in enumerate(plan):
            # print('cur=%s index=%s step=%s' % (format_number(self.current_volume), step_index, step))

            aspirate = step.get('aspirate')
            dispense = step.get('dispense')

            if aspirate:
                # we might have carryover from a previous transfer.
                if self.current_volume > 0:
                    info(Pretty().format('carried over {0:n} uL from prev operation', self.current_volume))

                if not seen_aspirate:
                    assert step_index == 0

                    if kwargs.get('allow_carryover', False) and zeroify(self.current_volume) > 0:
                        this_aspirated_location, __ = unpack_location(aspirate['location'])
                        if self.prev_aspirated_location is this_aspirated_location:
                            if self.has_disposal_vol(plan, step_index, **kwargs):
                                # try to remove current volume from this aspirate
                                new_aspirate_vol = zeroify(aspirate.get('volume') - self.current_volume)
                                if new_aspirate_vol == 0 or new_aspirate_vol >= self.min_volume:
                                    aspirate['volume'] = new_aspirate_vol
                                    info(Pretty().format('reduced this aspirate by {0:n} uL', self.current_volume))
                                    extra = 0  # can't blow out since we're relying on its presence in pipette
                                else:
                                    extra = self.current_volume - aspirate['volume']
                                    assert zeroify(extra) > 0
                            else:
                                info(Pretty().format("carryover of {0:n} uL isn't for disposal", self.current_volume))
                                extra = self.current_volume
                        else:
                            # different locations; can't re-use
                            info('this aspirate is from location different than current pipette contents')
                            extra = self.current_volume
                        if zeroify(extra) > 0:
                            # quiet_log('blowing out carryover of %s uL' % format_number(self.current_volume))
                            self._blowout_during_transfer(loc=None, **kwargs)  # loc isn't actually used

                    elif zeroify(self.current_volume) > 0:
                        info(Pretty().format('blowing out unexpected carryover of {0:n} uL', self.current_volume))
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
                    info(Pretty().format('current {0:n} uL with aspirate(has_disposal={1}) of {2:n} uL would overflow capacity',
                          self.current_volume,
                          self.has_disposal_vol(plan, step_index, **kwargs),
                          aspirate['volume']))
                    self._blowout_during_transfer(loc=None, **kwargs)  # loc isn't actually used
                self._aspirate_during_transfer(aspirate['volume'], aspirate['location'], **kwargs)

            if dispense:
                if self.current_volume < dispense['volume']:
                    warn(Pretty().format('current {0:n} uL will truncate dispense of {1:n} uL', self.current_volume, dispense['volume']))
                self._dispense_during_transfer(dispense['volume'], dispense['location'], **kwargs)

                do_touch = touch_tip or touch_tip is 0
                is_last_step = step is plan[-1]
                if is_last_step or plan[step_index + 1].get('aspirate'):
                    do_drop = not is_last_step or not kwargs.get('retain_tip', False)
                    # original always blew here. there are several reasons we could still be forced to blow
                    do_blow = not is_distribute  # other modes (are there any?) we're not sure about
                    do_blow = do_blow or kwargs.get('blow_out', False)  # for compatibility
                    do_blow = do_blow or do_touch  # for compatibility
                    do_blow = do_blow or not kwargs.get('allow_blow_elision', False)
                    if not do_blow:
                        if is_last_step:
                            if self.current_volume > 0:
                                if not kwargs.get('allow_carryover', False):
                                    do_blow = True
                                elif self.current_volume > kwargs.get('disposal_vol', 0):
                                    warn(Pretty().format('carried over {0:n} uL to next operation', self.current_volume))
                                else:
                                    info(Pretty().format('carried over {0:n} uL to next operation', self.current_volume))
                        else:
                            # if we can, account for any carryover in the next aspirate
                            if self.current_volume > 0:
                                if self.has_disposal_vol(plan, step_index + 1, **kwargs):
                                    next_aspirate = plan[step_index + 1].get('aspirate'); assert next_aspirate
                                    next_aspirated_location, __ = unpack_location(next_aspirate['location'])
                                    if self.prev_aspirated_location is next_aspirated_location:
                                        new_aspirate_vol = zeroify(next_aspirate.get('volume') - self.current_volume)
                                        if new_aspirate_vol == 0 or new_aspirate_vol >= self.min_volume:
                                            next_aspirate['volume'] = new_aspirate_vol
                                            info(Pretty().format('reduced next aspirate by {0:n} uL', self.current_volume))
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
                        tips = self._drop_tip_during_transfer(tips, step_index, total_transfers, **kwargs)
                else:
                    if air_gap:
                        self.air_gap(air_gap)
                    if do_touch:
                        self.touch_tip(touch_tip)

    def aspirate(self, volume=None, location=None, rate=1.0):
        # recapitulate super
        if not helpers.is_number(volume):
            if volume and not location:
                location = volume
            volume = self._working_volume - self.current_volume
        location = location if location else self.previous_placeable
        location = self._adjust_location_to_liquid_top(location=location, aspirate_volume=volume, clearances=config.aspirate)
        super().aspirate(volume=volume, location=location, rate=rate)
        # track volume todo: what if we're doing an air gap
        well, __ = unpack_location(location)
        get_well_volume(well).aspirate(volume)
        if volume != 0:
            self.prev_aspirated_location = well

    def dispense(self, volume=None, location=None, rate=1.0):
        # recapitulate super
        if not helpers.is_number(volume):
            if volume and not location:
                location = volume
            volume = self._working_volume - self.current_volume
        location = location if location else self.previous_placeable
        location = self._adjust_location_to_liquid_top(location=location, aspirate_volume=None, clearances=config.dispense)
        super().dispense(volume=volume, location=location, rate=rate)
        # track volume
        well, __ = unpack_location(location)
        get_well_volume(well).dispense(volume)

    def _adjust_location_to_liquid_top(self, location=None, aspirate_volume=None, clearances=None):
        if isinstance(location, Placeable):
            well = location; assert isWell(well)
            well_vol = get_well_volume(well).current_volume_min
            well_depth = get_well_geometry(well).min_depth_from_volume(well_vol if aspirate_volume is None else well_vol - aspirate_volume)
            z = well_depth - self._top_clearance(well, well_depth, clearance=clearances.top_clearance, factor=clearances.top_clearance_factor)
            z = max(z, clearances.bottom_clearance)
            z = min(z, well.z_size())
            return well.bottom(z)
        else:
            assert isinstance(location, tuple)
        return location

    def blow_out(self, location=None):
        super().blow_out(location)
        self._shake_tip(location)  # try to get rid of pesky retentive drops

    def _shake_tip(self, location):
        # Modelled after Pipette._shake_off_tips()
        shake_off_distance = SHAKE_OFF_TIPS_DISTANCE / 2  # / 2 == less distance than shaking off tips
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

    def done_tip(self):
        if self.has_tip:
            if self.current_volume > 0:
                info(Pretty().format('{0} has {1:n} uL remaining', self.name, self.current_volume))
            if trash_control:
                self.drop_tip()
            else:
                self.return_tip()

    def simple_mix(self, wells, msg=None, count=None, volume=None, drop_tip=True):
        if count is None:
            count = config.simple_mix.count
        if msg is not None:
            log(msg)
        if volume is None:
            volume = self.max_volume
        if not self.has_tip:
            self.pick_up_tip()
        for well in wells:
            self.mix(count, volume, well)
        if drop_tip:
            self.done_tip()

    # If count is provided, we do (at most) that many asp/disp cycles, clamped to an increment of min_incr
    def layered_mix(self, wells, msg='Mixing',
                    count=None,
                    min_incr=None,
                    incr=None,
                    count_per_incr=None,
                    volume=None,
                    drop_tip=None,
                    delay=None,
                    aspirate_rate=None,
                    dispense_rate=None,
                    initial_turnover=None,
                    max_tip_cycles=None):
        if drop_tip is None:
            drop_tip = config.layered_mix.drop_tip

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
        if drop_tip:
            self.done_tip()

    def _top_clearance(self, well, depth, clearance, factor):
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
        delay = fetch('delay')
        initial_turnover = fetch('initial_turnover')
        max_tip_cycles = fetch('max_tip_cycles', fpu.infinity)

        well_vol = get_well_volume(well).current_volume_min
        well_depth = get_well_geometry(well).depth_from_volume(well_vol)
        well_depth_after_asp = get_well_geometry(well).depth_from_volume(well_vol - volume)
        msg = Pretty().format("{0:s} well='{1:s}' cur_vol={2:n} well_depth={3:n} after_aspirate={4:n}", msg, well.get_name(), well_vol, well_depth, well_depth_after_asp)
        if msg is not None:
            log(msg)
        y_min = y = config.layered_mix.aspirate_bottom_clearance
        y_max = well_depth_after_asp - self._top_clearance(well, well_depth_after_asp, clearance=config.layered_mix.top_clearance, factor=config.layered_mix.top_clearance_factor)
        if count is not None:
            if count <= 1:
                y_max = y_min
                y_incr = 1  # just so we only go one time through the loop
            else:
                y_incr = (y_max - y_min) / (count-1)
                y_incr = max(y_incr, min_incr)
        else:
            assert incr is not None
            y_incr = incr

        first = True
        tip_cycles = 0
        while y <= y_max or numpy.isclose(y, y_max):
            if not first:
                self.delay(delay / 1000.0)
            #
            if first and initial_turnover is not None:
                count = int(0.5 + (initial_turnover / volume))
                count = max(count, count_per_incr)
            else:
                count = count_per_incr
            if not self.has_tip:
                self.pick_up_tip()
            for i in range(count):
                self.aspirate(volume, well.bottom(y), rate=fetch('aspirate_rate', config.layered_mix.aspirate_rate_factor))
                self.dispense(volume, well.bottom(y_max), rate=fetch('dispense_rate', config.layered_mix.dispense_rate_factor))
                tip_cycles += 1
                if tip_cycles >= max_tip_cycles:
                    self.done_tip()
                    tip_cycles = 0
            #
            y += y_incr
            first = False

# region Commands

# Enhance well name to include any label that might be present

def get_labelled_well_name(self):
    result = super(Well, self).get_name()
    label = getattr(self, 'label', None)
    if label is not None:
        result += ' (' + label + ')'
    return result


Well.get_name = get_labelled_well_name


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
    text = Pretty().format('Aspirating {volume:n} uL z={z:n} rate={rate:n} at {location}', volume=volume, location=location_text, rate=rate, z=z)
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
    text = Pretty().format('Dispensing {volume:n} uL z={z:n} rate={rate:n} at {location}', volume=volume, location=location_text, rate=rate, z=z)
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
# Utilities
########################################################################################################################

# Returns a unique name for the given location. Must track in Opentrons-Analyze.
def get_location_path(location):
    return '/'.join(list(reversed([str(item)
                                   for item in location.get_trace(None)
                                   if str(item) is not None])))


def log(msg: str, prefix="***********", suffix=' ***********'):
    robot.comment("%s%s%s%s" % (prefix, '' if len(prefix) == 0 else ' ', msg, suffix))

def info(msg):
    log(msg, prefix='info:', suffix='')

def warn(msg: str, prefix="***********", suffix=' ***********'):
    log(msg, prefix=prefix + " WARNING:", suffix=suffix)

def silent_log(msg):
    pass


def cube_root(value):
    return pow(value, 1.0/3.0)

def zeroify(value, digits=2):  # clamps small values to zero, leaves others alone
    rounded = round(value, digits)
    return rounded if rounded == 0 else value

class Pretty(string.Formatter):
    def format_field(self, value, spec):
        if spec.endswith('n'):  # 'n' for number
            precision = 2
            if spec.startswith('.', 0, -1):
                precision = int(spec[1:-1])
            if isinstance(value, Number) and fpu.is_finite(value):
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

# endregion

########################################################################################################################
########################################################################################################################
##                                                                                                                    ##
## Protocol                                                                                                           ##
##                                                                                                                    ##
########################################################################################################################
########################################################################################################################


# compute derived constants
strand_dilution_source_vol = strand_dilution_vol / strand_dilution_factor
strand_dilution_water_vol = strand_dilution_vol - strand_dilution_source_vol

########################################################################################################################
# Labware
########################################################################################################################

# Configure the tips
tips300a = labware.load('opentrons_96_tiprack_300ul', 1)
tips300b = labware.load('opentrons_96_tiprack_300ul', 4)
tips10 = labware.load('opentrons_96_tiprack_10ul', 7)

# Configure the pipettes.
p10 = MyPipette(instruments.P10_Single(mount='left', tip_racks=[tips10]))
p50 = MyPipette(instruments.P50_Single(mount='right', tip_racks=[tips300a, tips300b]))

# Blow out faster than default in an attempt to avoid hanging droplets on the pipettes after blowout
p10.set_flow_rate(blow_out=p10.get_flow_rates()['blow_out'] * config.blow_out_rate_factor)
p50.set_flow_rate(blow_out=p50.get_flow_rates()['blow_out'] * config.blow_out_rate_factor)

# Control tip usage
p10.start_at_tip(tips10[p10_start_tip])
p50.start_at_tip(tips300a[p50_start_tip])

# Custom disposal volumes to minimize reagent usage
p50_disposal_vol = 5
p10_disposal_vol = 1

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
note_liquid(location=water, name='Water', min_volume=6000)  # 6000 is a rough guess
note_liquid(location=strand_a, name='Strand A', min_volume=strand_dilution_source_vol)
note_liquid(location=strand_b, name='Strand B', min_volume=strand_dilution_source_vol)
note_liquid(location=diluted_strand_a, name='Diluted Strand A')
note_liquid(location=diluted_strand_b, name='Diluted Strand B')
note_liquid(location=master_mix, name='Master Mix')
for buffer in buffers:
    note_liquid(location=buffer[0], name='Buffer', initial_volume=buffer[1])
for evagreen in evagreens:
    note_liquid(location=evagreen[0], name='Evagreen', initial_volume=evagreen[1])

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
    p50.layered_mix([strand_a], 'Mixing Strand A')
    p50.layered_mix([strand_b], 'Mixing Strand B')

    # Create dilutions of strands
    log('Moving water for diluting Strands A and B')
    p50.transfer(strand_dilution_water_vol, water, [diluted_strand_a, diluted_strand_b],
                 new_tip='once',  # can reuse for all diluent dispensing since dest tubes are initially empty
                 trash=trash_control
                 )
    log('Diluting Strand A')
    p50.transfer(strand_dilution_source_vol, strand_a, diluted_strand_a, trash=trash_control, retain_tip=True)
    p50.layered_mix([diluted_strand_a], 'Mixing Diluted Strand A', incr=2)

    log('Diluting Strand B')
    p50.transfer(strand_dilution_source_vol, strand_b, diluted_strand_b, trash=trash_control, retain_tip=True)
    p50.layered_mix([diluted_strand_b], 'Mixing Diluted Strand B', incr=2)


def createMasterMix():
    # Buffer was just unfrozen. Mix to ensure uniformity. EvaGreen doesn't freeze, no need to mix
    p50.layered_mix([buffer for buffer, __ in buffers], "Mixing Buffers", incr=4)

    # transfer from multiple source wells, each with a current defined volume
    def transfer_multiple(ctx, xfer_vol_remaining, tubes, dest, new_tip, *args, **kwargs):
        tube_index = 0
        cur_loc = None
        cur_vol = 0
        min_vol = 0
        while xfer_vol_remaining > 0:
            if xfer_vol_remaining < p50_min_vol:
                warn("remaining transfer volume of %f too small; ignored" % xfer_vol_remaining)
                return
            # advance to next tube if there's not enough in this tube
            while cur_loc is None or cur_vol <= min_vol:
                cur_loc = tubes[tube_index][0]
                cur_vol = tubes[tube_index][1]
                min_vol = max(p50_min_vol, cur_vol / 15.0)  # tolerance is proportional to specification of volume. can probably make better guess
                tube_index = tube_index + 1
            this_vol = min(xfer_vol_remaining, cur_vol - min_vol)
            assert this_vol >= p50_min_vol  # TODO: is this always the case?
            log('%s: xfer %f from %s in %s to %s in %s' % (ctx, this_vol, cur_loc, cur_loc.parent, dest, dest.parent))
            p50.transfer(this_vol, cur_loc, dest, trash=trash_control, new_tip=new_tip, **kwargs)
            xfer_vol_remaining -= this_vol
            cur_vol -= this_vol

    def mix_master_mix():
        log('Mixing Master Mix')
        p50.layered_mix([master_mix], incr=2, initial_turnover=master_mix_evagreen_vol * 1.2, max_tip_cycles=config.layered_mix.max_tip_cycles_large)

    log('Creating Master Mix: Water')
    p50.transfer(master_mix_common_water_vol, water, master_mix, trash=trash_control)

    log('Creating Master Mix: Buffer')
    transfer_multiple('Creating Master Mix: Buffer', master_mix_buffer_vol, buffers, master_mix, new_tip='once', retain_tip=True)  # 'once' because we've only got water & buffer in context
    p50.done_tip()  # EvaGreen needs a new tip

    log('Creating Master Mix: EvaGreen')
    transfer_multiple('Creating Master Mix: EvaGreen', master_mix_evagreen_vol, evagreens, master_mix, new_tip='always', retain_tip=True)  # 'always' to avoid contaminating the Evagreen source w/ buffer

    mix_master_mix()


########################################################################################################################
# Plating
########################################################################################################################

def plateEverythingAndMix():
    # Plate master mix
    log('Plating Master Mix')
    master_mix_per_well = 28
    p50.distribute(master_mix_per_well, master_mix, usedWells(),
                   new_tip='once',
                   disposal_vol=p50_disposal_vol,
                   trash=trash_control)

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

    p50.distribute(water_volumes, water, plate.wells(),
                   new_tip='once',
                   disposal_vol=p50_disposal_vol,
                   trash=trash_control,
                   allow_blow_elision=allow_blow_elision,
                   allow_carryover=allow_carryover)

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
            disposal_vol = p10_disposal_vol
        else:
            p = p50
            disposal_vol = p50_disposal_vol
        log('Plating Strand A: volume %d with %s' % (volume, p.name))
        volumes = [volume] * len(dest_wells)
        p.distribute(volumes, diluted_strand_a, dest_wells,
                     new_tip='never',
                     disposal_vol=disposal_vol,
                     trash=trash_control,
                     allow_blow_elision=allow_blow_elision,
                     allow_carryover=allow_carryover)
    p10.done_tip()
    p50.done_tip()

    # Plate strand B and mix
    # Mixing always needs the p50, but plating may need either; optimize tip usage
    log('Plating Strand B')
    mixed_wells = set()
    for iVolume in range(0, len(strand_volumes)):
        dest_wells = calculateStrandBWells(iVolume)
        volume = strand_volumes[iVolume]
        # if strand_volumes[index] == 0: continue  # don't skip: we want to mix
        if usesP10(volume, len(dest_wells), allow_zero=True):
            p = p10
        else:
            p = p50

        # We can't use distribute here as we need to avoid cross contamination from plate well to plate well
        for well in dest_wells:
            if volume != 0:
                log("Plating Strand B: well='%s' vol=%d pipette=%s" % (well.get_name(), volume, p.name))
                p.pick_up_tip()
                p.transfer(volume, diluted_strand_b, well, new_tip='never')
            if p is p50:
                mix_plate_well(well, drop_tip=False)
                mixed_wells.add(well)
            p.done_tip()

    for well in plate.wells():
        if well not in mixed_wells:
            mix_plate_well(well)


def mix_plate_well(well, drop_tip=True):
    p50.layered_mix([well], incr=0.75, drop_tip=drop_tip)


def debug_mix_plate():
    wells = plate.cols(0)[0:2]
    for well in wells:
        get_well_volume(well).set_initial_volume(84)
    for well in wells:
        mix_plate_well(well)
    p50.done_tip()

def debug_test_blow():
    p50.pick_up_tip()
    p50.aspirate(5, location=plate.wells('A2'))
    p50.blow_out(p50.trash_container)
    p50.done_tip()


########################################################################################################################
# Off to the races
########################################################################################################################

master_and_dilutions_made = False

if not master_and_dilutions_made:
    diluteStrands()
    createMasterMix()
else:
    note_liquid(location=diluted_strand_a, name=None, initial_volume=strand_dilution_vol)
    note_liquid(location=diluted_strand_b, name=None, initial_volume=strand_dilution_vol)
    note_liquid(location=master_mix, name=None, initial_volume=master_mix_vol)

plateEverythingAndMix()
