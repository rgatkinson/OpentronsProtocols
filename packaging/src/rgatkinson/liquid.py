#
# Liquid.py
#

import json
from enum import Enum

from opentrons import robot
from opentrons.legacy_api.containers import unpack_location

import rgatkinson
from rgatkinson.interval import is_scalar, supremum, Interval, fpu, is_interval
from rgatkinson.logging import pretty, get_location_path
from rgatkinson.util import first
from rgatkinson.well import is_well

class Liquid:
    def __init__(self, name):
        self.name = name
        self.concentration = Concentration('dc')  # dc==dont' care

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


class LiquidVolume(object):

    #-------------------------------------------------------------------------------------------------------------------
    # Construction
    #-------------------------------------------------------------------------------------------------------------------

    def __init__(self, well, config):
        self.well = well
        self.config = config
        self.initial_volume_known = False
        self.initial_volume = Interval([0, fpu.infinity])
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
            self.set_initial_volume(Interval([volume, fpu.infinity if self.well is None else self.config.well_geometry(self.well).well_capacity]))
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


def note_liquid(location, name=None, initial_volume=None, min_volume=None, concentration=None, local_config=None):
    # Must keep in sync with Opentrons-Analyze controller.note_liquid_name
    if local_config is None:
        local_config = rgatkinson.configuration.config
    well, __ = unpack_location(location)
    assert is_well(well)
    if name is None:
        name = well.label
    else:
        well.label = name
    d = {'name': name, 'location': get_location_path(well)}
    if initial_volume is None and min_volume is not None:
        initial_volume = Interval([min_volume, local_config.well_geometry(well).well_capacity])
    if initial_volume is not None:
        d['initial_volume'] = initial_volume
        local_config.well_volume(well).set_initial_volume(initial_volume)
    if concentration is not None:
        d['concentration'] = str(Concentration(concentration))
    serialized = json.dumps(d).replace("{", "{{").replace("}", "}}")  # runtime calls comment.format(...) on our comment; avoid issues therewith
    robot.comment('Liquid: %s' % serialized)