#
# Liquid.py
#
import json
from enum import Enum

from opentrons import robot
from opentrons.legacy_api.containers import unpack_location

import rgatkinson
from rgatkinson.interval import supremum, Interval, fpu, is_interval, infimum
from rgatkinson.logging import pretty, get_location_path
from rgatkinson.util import first, is_scalar, is_close

class Liquid:
    def __init__(self, name):
        self.name = name
        self.concentration = Concentration('dc')  # dc==dont' care

    def __str__(self) -> str:
        if self.concentration.flavor == Concentration.Flavor.DontCare:
            return f'Liquid({self.name})'
        else:
            return f'Liquid([{self.name}]={self.concentration})'


class Liquids(object):
    def __init__(self):
        self._liquids = dict()

    def get_liquid(self, liquid_name):
        try:
            return self._liquids[liquid_name]
        except KeyError:
            self._liquids[liquid_name] = Liquid(liquid_name)
            return self._liquids[liquid_name]


class Concentration(object):

    class Flavor(Enum):
        Molar = 0
        X = 1
        DontCare = 2

    def __init__(self, value, unit=None, flavor=None):
        if isinstance(value, Concentration):
            self.value = value.value
            self.flavor = value.flavor
            return

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
        units = [('mM', 0.001), ('uM', 0.000001), ('nM', 1e-9), ('pM', 1e-12), ('fM', 1e-15), ('M', 1)]
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

    @property
    def x(self):
        assert self.flavor == Concentration.Flavor.X
        return self.value

    @property
    def uM(self):
        assert self.flavor == Concentration.Flavor.Molar
        return self.value * 1e6

    @property
    def nM(self):
        assert self.flavor == Concentration.Flavor.Molar
        return self.value * 1e9

    @property
    def M(self):
        assert self.flavor == Concentration.Flavor.Molar
        return self.value

    @property
    def molar(self):
        assert self.flavor == Concentration.Flavor.Molar
        return self.value

    def __mul__(self, scale):
        return Concentration(self.value * scale, flavor=self.flavor)

    def __rmul__(self, scale):
        return self * scale

    def __str__(self) -> str:
        def test(v, scale):
            return not is_close(int(v * scale + 0.5), 0)

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
            elif test(self.value, 1e9):
                return emit(1e9, 'nM')
            elif test(self.value, 1e12):
                return emit(1e12, 'pM')
            else:
                return emit(1e15, 'fM')
        elif self.flavor == Concentration.Flavor.X:
            return pretty.format('{0:.3n}x', self.value)
        else:
            return 'DC'


class Mixture(object):
    def __init__(self, liquid=None, initially=0):
        self.liquids = dict()  # map from liquid to volume
        if liquid is not None:
            self.set_initial_liquid(liquid=liquid, initially=initially)

    def set_initial_liquid(self, liquid, initially):
        assert len(self.liquids) == 0
        if liquid is not None:
            self._adjust_liquid(liquid, LiquidVolume.fix_initially(initially))

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


class LiquidVolume(object):

    #-------------------------------------------------------------------------------------------------------------------
    # Construction
    #-------------------------------------------------------------------------------------------------------------------

    def __init__(self, well):
        self.__well = None
        self.well = well
        self.initially_set = False
        self.initially = Interval([0, fpu.infinity])
        self.cum_delta = 0
        self.min_delta = 0
        self.max_delta = 0

    @property
    def well(self):
        return self.__well

    @well.setter
    def well(self, value):
        if self.__well is not value:
            old_well = self.__well
            self.__well = None
            if old_well is not None:
                old_well.liquid_volume = None

            self.__well = value
            if self.__well is not None:
                self.__well.liquid_volume = self

    def set_initially(self, initially):  # idempotent
        if self.initially_set:
            assert self.initially == initially
        else:
            assert not self.initially_set
            assert self.cum_delta == 0
            self.initially_set = True
            self.initially = self.fix_initially(initially)

    @classmethod
    def fix_initially(cls, initially):
        if isinstance(initially, list):  # work around json inability to parse serialized Intervals
            initially = Interval(*initially)
        return initially

    #-------------------------------------------------------------------------------------------------------------------
    # Accessing
    #-------------------------------------------------------------------------------------------------------------------

    @property
    def made_by_machine(self):
        """Is (at least the current min of) this volume of liquid created by machine, as opposed to a human"""
        return not self.initially_set or infimum(self.initially) == 0

    @property
    def current_volume(self):  # may be interval
        return self.initially + self.cum_delta

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
    def lo_volume(self):  # minimum historically seen
        return self.initially + self.min_delta

    @property
    def hi_volume(self):  # maximum historically seen
        return self.initially + self.max_delta

    @property
    def _min_aspiratable_volume(self):
        return 0 if self.well is None else self.well.geometry.min_aspiratable_volume

    #-------------------------------------------------------------------------------------------------------------------
    # Actions
    #-------------------------------------------------------------------------------------------------------------------

    def aspirate(self, volume):
        assert volume >= 0
        if not self.initially_set:
            self.set_initially(Interval([volume, fpu.infinity if self.well is None else self.well.geometry.well_capacity]))
        self._track_volume(-volume)

    def dispense(self, volume):
        assert volume >= 0
        if not self.initially_set:
            self.set_initially(0)
        self._track_volume(volume)

    def _track_volume(self, delta):
        self.cum_delta = self.cum_delta + delta
        self.min_delta = min(self.min_delta, self.cum_delta)
        self.max_delta = max(self.max_delta, self.cum_delta)


def note_liquid(location, name=None, initially=None, initially_at_least=None, concentration=None, local_config=None):
    # Must keep in sync with Opentrons-Analyze analyze_liquid_name

    if local_config is None:
        local_config = rgatkinson.configuration.config

    well, __ = unpack_location(location)
    if name is None:
        name = well.label
    else:
        well.label = name

    liquid = local_config.execution_context.liquids.get_liquid(name)

    d = {'name': name, 'location': get_location_path(well)}

    if initially is not None and initially_at_least is not None:
        raise ValueError  # can't use both at once

    initially = LiquidVolume.fix_initially(initially)

    if initially is None and initially_at_least is not None:
        initially = Interval([initially_at_least, well.geometry.well_capacity])

    if initially is not None:
        d['initially'] = initially
        well.liquid_volume.set_initially(initially)

    if concentration is not None:
        concentration = Concentration(concentration)
        liquid.concentration = concentration
        d['concentration'] = str(concentration)

    serialized = json.dumps(d).replace("{", "{{").replace("}", "}}")  # runtime calls comment.format(...) on our comment; avoid issues therewith
    robot.comment('Liquid: %s' % serialized)
