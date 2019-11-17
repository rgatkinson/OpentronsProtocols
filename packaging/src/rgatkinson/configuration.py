#
# configuration.py
#
from rgatkinson.liquid import LiquidVolume
from rgatkinson.util import thread_local_storage

#-----------------------------------------------------------------------------------------------------------------------

class ProtocolExecutionContext(object):
    """
    One instance of this across all configuration contexts
    """
    def __init__(self):
        from rgatkinson.liquid import Liquids
        self.liquids = Liquids()

#-----------------------------------------------------------------------------------------------------------------------

class AbstractConfigurationContext(object):
    def __init__(self, execution_context: ProtocolExecutionContext):
        self.execution_context = execution_context

class SimpleConfigurationContext(AbstractConfigurationContext):
    def __init__(self, execution_context: ProtocolExecutionContext):
        super().__init__(execution_context)

class PauseConfigurationContext(AbstractConfigurationContext):
    def __init__(self, execution_context: ProtocolExecutionContext, ms_default):
        super().__init__(execution_context)
        self.ms_default = ms_default

class ClearanceConfigurationContext(AbstractConfigurationContext):
    def __init__(self, execution_context: ProtocolExecutionContext, top, bottom):
        super().__init__(execution_context)
        self.top_clearance = top
        self.bottom_clearance = bottom

class AspirateConfigurationContext(ClearanceConfigurationContext):
    def __init__(self, execution_context: ProtocolExecutionContext):
        super().__init__(execution_context, -3.5, 1.0)
        self.pre_wet = SimpleConfigurationContext(execution_context)
        self.pre_wet.default = True
        self.pre_wet.count = 2  # save some time vs 3
        self.pre_wet.max_volume_fraction = 1  # https://github.com/Opentrons/opentrons/issues/2901 would pre-wet only 2/3, but why not everything?
        self.pre_wet.rate_func = lambda aspirate_rate: 1  # could instead just use the aspirate
        self.pause = PauseConfigurationContext(execution_context, 750)

class DispenseConfigurationContext(ClearanceConfigurationContext):
    def __init__(self, execution_context: ProtocolExecutionContext):
        super().__init__(execution_context, -2.0, 0.5)
        self.full_dispense = SimpleConfigurationContext(execution_context)
        self.full_dispense.default = True

class LayeredMixConfigurationContext(ClearanceConfigurationContext):
    def __init__(self, execution_context: ProtocolExecutionContext):
        super().__init__(execution_context, -1.5, 1.0)  # close, so we mix top layers too
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

class WellGeometryConfigurationContext(AbstractConfigurationContext):
    def __init__(self, execution_context: ProtocolExecutionContext):
        super().__init__(execution_context)
        self.radial_clearance_tolerance = 0.5

class TopConfigurationContext(AbstractConfigurationContext):
    def __init__(self, execution_context: ProtocolExecutionContext):
        super().__init__(execution_context)
        self.enable_enhancements = True
        self.trash_control = True
        self.blow_out_rate_factor = 3.0
        self.min_aspirate_factor_hack = 15.0
        self.allow_blow_elision_default = True
        self.allow_overspill_default = True

        self.aspirate: AspirateConfigurationContext = AspirateConfigurationContext(execution_context)
        self.dispense: DispenseConfigurationContext = DispenseConfigurationContext(execution_context)
        self.layered_mix: LayeredMixConfigurationContext = LayeredMixConfigurationContext(execution_context)
        self.wells = WellGeometryConfigurationContext(execution_context)

    def liquid_volume(self, well):
        try:
            result = well.liquid_volume
            assert result.config is self
            return result
        except AttributeError:
            well.liquid_volume = LiquidVolume(well, self)
            return well.liquid_volume

config = TopConfigurationContext(ProtocolExecutionContext())

# ambiently remember the default top configuration context
thread_local_storage.config = config
