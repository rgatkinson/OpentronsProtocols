#
# configuration.py
#

from rgatkinson.liquid import LiquidVolume
from rgatkinson.well import is_well, UnknownWellGeometry


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
            well.contents = LiquidVolume(well, self)
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
