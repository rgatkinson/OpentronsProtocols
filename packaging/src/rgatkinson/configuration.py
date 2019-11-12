#
# configuration.py
#

from rgatkinson.liquid import LiquidVolume
from rgatkinson.well import is_well, UnknownWellGeometry


class ConfigurationContext(object):
    pass

class PauseConfigurationContext(ConfigurationContext):
    def __init__(self, ms_default):
        self.ms_default = ms_default

class ClearanceConfigurationContext(ConfigurationContext):
    pass

class AspirateConfigurationContext(ClearanceConfigurationContext):
    def __init__(self):
        self.bottom_clearance = 1.0  # see Pipette._position_for_aspirate
        self.top_clearance = -3.5
        self.extra_top_clearance_name = 'extra_aspirate_top_clearance'
        self.pre_wet = ConfigurationContext()
        self.pre_wet.default = True
        self.pre_wet.count = 2  # save some time vs 3
        self.pre_wet.max_volume_fraction = 1  # https://github.com/Opentrons/opentrons/issues/2901 would pre-wet only 2/3, but why not everything?
        self.pre_wet.rate_func = lambda aspirate_rate: 1  # could instead just use the aspirate
        self.pause = PauseConfigurationContext(750)

class DispenseConfigurationContext(ClearanceConfigurationContext):
    def __init__(self):
        self.bottom_clearance = 0.5  # see Pipette._position_for_dispense
        self.top_clearance = -2.0
        self.extra_top_clearance_name = 'extra_dispense_top_clearance'  # todo: is this worth it?
        self.full_dispense = ConfigurationContext()
        self.full_dispense.default = True

class LayeredMixConfigurationContext(ConfigurationContext):
    def __init__(self):
        self.top_clearance = -1.5  # close, so we mix top layers too
        self.aspirate_bottom_clearance = 1.0
        self.aspirate_rate_factor = 4.0
        self.dispense_rate_factor = 4.0
        self.incr = 1.0
        self.count = None  # so we default to using incr, not count
        self.min_incr = 0.5
        self.count_per_incr = 2
        self.ms_pause = 0           # pause between layers
        self.ms_final_pause = 750   # pause after last layer
        self.keep_last_tip = False
        self.initial_turnover = None
        self.max_tip_cycles = None
        self.max_tip_cycles_large = None
        self.enable_radial_randomness = True
        self.radial_clearance_tolerance = 0.5

class WellsMixConfigurationContext(ConfigurationContext):
    def __init__(self):
        self.radial_clearance_tolerance = 0.5

class TopConfigurationContext(ConfigurationContext):
    def __init__(self):
        self.enable_enhancements = True
        self.trash_control = True
        self.blow_out_rate_factor = 3.0
        self.min_aspirate_factor_hack = 15.0
        self.allow_blow_elision_default = True
        self.allow_overspill_default = True

        self.aspirate = AspirateConfigurationContext()
        self.dispense = DispenseConfigurationContext()
        self.layered_mix = LayeredMixConfigurationContext()
        self.wells = WellsMixConfigurationContext()

    def well_geometry(self, well):
        assert is_well(well)
        try:
            result = well.geometry
            assert result.config is self
            return result
        except AttributeError:
            return self.set_well_geometry(well, UnknownWellGeometry)

    def set_well_geometry(self, well, geometry_class):
        result = geometry_class(well, self)
        assert well.geometry is result
        return result

    def liquid_volume(self, well):
        assert is_well(well)
        try:
            result = well.liquid_volume
            assert result.config is self
            return result
        except AttributeError:
            well.liquid_volume = LiquidVolume(well, self)
            return well.liquid_volume

config = TopConfigurationContext()
