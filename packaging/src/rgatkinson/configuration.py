#
# configuration.py
#

from opentrons import protocol_api

#-----------------------------------------------------------------------------------------------------------------------

class ExecutionContext(object):
    """
    One instance of this across all configuration contexts
    """
    def __init__(self):
        self.__protocol_context: protocol_api.ProtocolContext = None
        self.__liquids = None

    @property
    def protocol_context(self) -> protocol_api.ProtocolContext:
        return self.__protocol_context

    @protocol_context.setter
    def protocol_context(self, protocol_context: protocol_api.ProtocolContext):
        # Hook the HardwareManager with our extensions
        if protocol_context:  # will be None in v1
            from rgatkinson.hardware import EnhancedHardwareManager
            if not isinstance(protocol_context._hw_manager, EnhancedHardwareManager):
                EnhancedHardwareManager.hook(protocol_context._hw_manager)
        self.__protocol_context = protocol_context

    @property
    def liquids(self):  # deferred creation to avoid initialization cycles between modules
        if self.__liquids is None:
            from rgatkinson.liquid import Liquids
            self.__liquids = Liquids()
        return self.__liquids

    @property
    def isApiV1(self):
        return self.protocol_context is None

#-----------------------------------------------------------------------------------------------------------------------

class AbstractConfigurationContext(object):
    def __init__(self, execution_context: ExecutionContext):
        self.execution_context = execution_context

    @property
    def isApiV1(self):
        return self.execution_context.isApiV1

    @property
    def protocol_context(self):
        return self.execution_context.protocol_context

    @protocol_context.setter
    def protocol_context(self, value):
        self.execution_context.protocol_context = value

class SimpleConfigurationContext(AbstractConfigurationContext):
    def __init__(self, execution_context: ExecutionContext):
        super().__init__(execution_context)

class PauseConfigurationContext(AbstractConfigurationContext):
    def __init__(self, execution_context: ExecutionContext, ms_default):
        super().__init__(execution_context)
        self.ms_default = ms_default

class ClearanceConfigurationContext(AbstractConfigurationContext):
    def __init__(self, execution_context: ExecutionContext, top, bottom, manual_liquid_volume_allowance):
        super().__init__(execution_context)
        self.top_clearance = top
        self.bottom_clearance = bottom
        self.manual_liquid_volume_allowance = manual_liquid_volume_allowance

class AspirateConfigurationContext(ClearanceConfigurationContext):
    def __init__(self, execution_context: ExecutionContext):
        super().__init__(execution_context, -5.0, 1.0, 0.05)  # -5.0 was -3.5; temporary; fix when 5ml geometry fixed; ie: allow for 5% slop in volume of manual pipetting when p
        self.pre_wet = SimpleConfigurationContext(execution_context)
        self.pre_wet.default = True
        self.pre_wet.count = 2  # save some time vs 3
        self.pre_wet.max_volume_fraction = 1  # https://github.com/Opentrons/opentrons/issues/2901 would pre-wet only 2/3, but why not everything?
        self.pre_wet.rate_func = lambda aspirate_rate: 1  # could instead just use the aspirate
        self.pause = PauseConfigurationContext(execution_context, 750)

class DispenseConfigurationContext(ClearanceConfigurationContext):
    def __init__(self, execution_context: ExecutionContext):
        super().__init__(execution_context, -2.0, 0.5, 0)  # dispensing at top is less critical than aspirating
        self.full_dispense = SimpleConfigurationContext(execution_context)
        self.full_dispense.default = True

class LayeredMixConfigurationContext(ClearanceConfigurationContext):
    def __init__(self, execution_context: ExecutionContext):
        super().__init__(execution_context, -1.5, 1.0, 0)  # close, so we mix top layers too
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
    def __init__(self, execution_context: ExecutionContext):
        super().__init__(execution_context)
        self.radial_clearance_tolerance = 0.5

class TopConfigurationContext(AbstractConfigurationContext):
    def __init__(self, execution_context: ExecutionContext):
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

config: TopConfigurationContext = TopConfigurationContext(ExecutionContext())
