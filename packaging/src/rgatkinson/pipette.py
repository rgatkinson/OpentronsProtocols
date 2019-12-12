#
# pipette.py
#

from abc import abstractmethod

from rgatkinson.configuration import TopConfigurationContext, AspirateConfigurationContext, DispenseConfigurationContext
from rgatkinson.logging import pretty, info_while
from rgatkinson.types import TipWetness
from rgatkinson.util import sqrt, infinity, tls
from rgatkinson.well import FalconTube15MlGeometry, FalconTube50MlGeometry, Eppendorf5Point0MlTubeGeometry, \
    Eppendorf1Point5MlTubeGeometry, IdtTubeWellGeometry, Biorad96WellPlateWellGeometry, EnhancedWellV1, \
    EnhancedWell, EnhancedWellV2


########################################################################################################################
# Radial Clearance
########################################################################################################################

class RadialClearanceManager(object):

    def __init__(self, config):
        self.config = config
        self._functions = {
            ('p50_single_v1.4', 'opentrons/opentrons_96_tiprack_300ul/1', FalconTube15MlGeometry): self.p50_single_v1_4_opentrons_96_tiprack_300ul_falcon15ml,
            ('p50_single_v1.4', 'opentrons/opentrons_96_tiprack_300ul/1', FalconTube50MlGeometry): self.p50_single_v1_4_opentrons_96_tiprack_300ul_falcon50ml,
            ('p50_single_v1.4', 'opentrons/opentrons_96_tiprack_300ul/1', Eppendorf1Point5MlTubeGeometry): self.p50_single_v1_4_opentrons_96_tiprack_300ul_eppendorf1_5ml,
            ('p50_single_v1.4', 'opentrons/opentrons_96_tiprack_300ul/1', Eppendorf5Point0MlTubeGeometry): self.p50_single_v1_4_opentrons_96_tiprack_300ul_eppendorf5_0ml,
            ('p50_single_v1.4', 'opentrons/opentrons_96_tiprack_300ul/1', IdtTubeWellGeometry): self.p50_single_v1_4_opentrons_96_tiprack_300ul_idt_tube,
            ('p50_single_v1.4', 'opentrons/opentrons_96_tiprack_300ul/1', Biorad96WellPlateWellGeometry): self.p50_single_v1_4_opentrons_96_tiprack_300ul_biorad_plate_well,
        }

    def get_clearance_function(self, pipette, well):
        key = (pipette.model, pipette.current_tip_tiprack.uri, well.geometry.__class__)
        return self._functions.get(key, None)

    def _free_sailing(self):
        return infinity

    def p50_single_v1_4_opentrons_96_tiprack_300ul_falcon15ml(self, depth):
        if depth < 0:
            return 0
        if depth < 4.42012:
            return 0.3181014675267553 + 0.2492912278496944*depth
        if depth < 52.59:
            return 1.3843756405067387 + 0.008059412406212692*depth
        if depth < 59.9064:
            return -14.830911008591517 + 0.3163934426229509*depth
        if depth <= 118.07:
            return 3.640273194181542 + 0.008059412406212692*depth
        return self._free_sailing()

    def p50_single_v1_4_opentrons_96_tiprack_300ul_falcon50ml(self, depth):
        if depth < 0:
            return 0
        if depth < 6.69969:
            return 3.036768510671534 + 0.7002075382097096*depth
        if depth < 47.19:
            return 7.678249659109892 + 0.0074166666666666305*depth
        if depth < 54.9388:
            return -6.90236439826716 + 0.3163934426229509*depth
        if depth <= 112.67:
            return 10.164032546012043 + 0.0057498667891316*depth
        return self._free_sailing()

    def p50_single_v1_4_opentrons_96_tiprack_300ul_eppendorf1_5ml(self, depth):
        if depth <= 0.0220787:
            return 0
        if depth < 0.114975:
            return -0.505 + 2.2698281410529737 * sqrt((2.2462247020231834 - 0.19409486595347666 * depth) * depth)
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
            return -0.505 + 0.9281232870726978 * sqrt((3.624697354653752 - 1.1608835603563077 * depth) * depth)
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
        if depth <= 0:
            return 0
        if depth < 3.72084:
            return 0.6610777502824954 + 0.1789907029367993*depth
        if depth <= 6.28:
            return 1.1722035122811951 + 0.04162219850586984*depth
        if depth <= 14.81:
            return 0.9329578591090766 + 0.07971864009378668*depth
        return self._free_sailing()


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

class AspirateParamsTransfer(object):
    def __init__(self, config: AspirateConfigurationContext) -> None:
        self.config = config
        self.pre_wet_during_transfer_kw = '_pre_wet_transfer'
        self.ms_pause_during_transfer_kw = '_ms_pause_transfer'
        self.top_clearance_transfer_kw = '_top_clearance_transfer'
        self.bottom_clearance_transfer_kw = '_bottom_clearance_transfer'
        self.manual_manufacture_tolerance_transfer_kw = '_manual_manufacture_tolerance_transfer'
        self.pre_wet_transfer = None
        self.ms_pause_transfer = None
        self.top_clearance_transfer = None
        self.bottom_clearance_transfer = None
        self.manual_manufacture_tolerance_transfer = None

    def clear(self):
        self.pre_wet_transfer = None
        self.ms_pause_transfer = None
        self.top_clearance_transfer = None
        self.bottom_clearance_transfer = None
        self.manual_manufacture_tolerance_transfer = None

    def sequester(self, kwargs):
        kwargs[self.pre_wet_during_transfer_kw] = not not kwargs.get('pre_wet', self.config.pre_wet.default)
        kwargs[self.ms_pause_during_transfer_kw] = kwargs.get('ms_pause', self.config.pause.ms_default)
        kwargs[self.top_clearance_transfer_kw] = kwargs.get('aspirate_top_clearance', None)
        kwargs[self.bottom_clearance_transfer_kw] = kwargs.get('aspirate_bottom_clearance', None)
        kwargs[self.manual_manufacture_tolerance_transfer_kw] = kwargs.get('manual_liquid_volume_allowance', None)

    def unsequester(self, kwargs):
        self.pre_wet_transfer = kwargs.get(self.pre_wet_during_transfer_kw)
        self.ms_pause_transfer = kwargs.get(self.ms_pause_during_transfer_kw)
        self.top_clearance_transfer = kwargs.get(self.top_clearance_transfer_kw)
        self.bottom_clearance_transfer = kwargs.get(self.bottom_clearance_transfer_kw)
        self.manual_manufacture_tolerance_transfer = kwargs.get(self.manual_manufacture_tolerance_transfer_kw)

    def __enter__(self):
        assert tls.aspirate_params_transfer is None
        tls.aspirate_params_transfer = self

    def __exit__(self, exception_type, exception_value, traceback):
        assert tls.aspirate_params_transfer is self
        tls.aspirate_params_transfer = None


class DispenseParamsTransfer(object):
    def __init__(self, config: DispenseConfigurationContext) -> None:
        self.config = config
        self.full_dispense_transfer_kw = '_full_dispense_transfer'
        self.top_clearance_transfer_kw = '_top_clearance_transfer'
        self.bottom_clearance_transfer_kw = '_bottom_clearance_transfer'
        self.manual_manufacture_tolerance_transfer_kw = '_manual_manufacture_tolerance_transfer'

        self.full_dispense_transfer = False
        self.top_clearance_transfer = None
        self.bottom_clearance_transfer = None
        self.manual_manufacture_tolerance_transfer = None

    def clear(self):
        self.full_dispense_transfer = False
        self.top_clearance_transfer = None
        self.bottom_clearance_transfer = None
        self.manual_manufacture_tolerance_transfer = None

    def sequester(self, kwargs, can_full_dispense):
        kwargs[self.full_dispense_transfer_kw] = not not (kwargs.get('full_dispense', self.config.full_dispense.default) and can_full_dispense)
        kwargs[self.top_clearance_transfer_kw] = kwargs.get('dispense_top_clearance', None)
        kwargs[self.bottom_clearance_transfer_kw] = kwargs.get('dispense_bottom_clearance', None)
        kwargs[self.manual_manufacture_tolerance_transfer_kw] = kwargs.get('manual_liquid_volume_allowance', None)

    def unsequester(self, kwargs):
        assert kwargs.get(self.full_dispense_transfer_kw, None) is not None
        self.full_dispense_transfer = kwargs.get(self.full_dispense_transfer_kw)
        self.top_clearance_transfer = kwargs.get(self.top_clearance_transfer_kw)
        self.bottom_clearance_transfer = kwargs.get(self.bottom_clearance_transfer_kw)
        self.manual_manufacture_tolerance_transfer = kwargs.get(self.manual_manufacture_tolerance_transfer_kw)

    def __enter__(self):
        assert tls.dispense_params_transfer is None
        tls.dispense_params_transfer = self

    def __exit__(self, exception_type, exception_value, traceback):
        assert tls.dispense_params_transfer is self
        tls.dispense_params_transfer = None


class DispenseParams(object):
    def __init__(self) -> None:
        self.full_dispense_from_dispense = False
        self.fully_dispensed = False

    def __enter__(self):
        assert tls.dispense_params is None
        tls.dispense_params = self

    def __exit__(self, exception_type, exception_value, traceback):
        assert tls.dispense_params is self
        tls.dispense_params = None


class EnhancedPipette(object):

    #-------------------------------------------------------------------------------------------------------------------
    # Construction
    #-------------------------------------------------------------------------------------------------------------------

    def __init__(self, config: TopConfigurationContext):
        self.config = config
        self.tip_wetness = TipWetness.NONE
        self.mixes_in_progress = list()
        self.radial_clearance_manager = RadialClearanceManager(self.config)
        self.prev_aspirated_well = None

    #-------------------------------------------------------------------------------------------------------------------
    # Accessing
    #-------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def get_speeds(self):
        pass

    @abstractmethod
    def get_flow_rates(self):
        pass

    @abstractmethod
    def get_max_volume(self):
        pass

    @abstractmethod
    def aspirate(self,
                 volume: float = None,
                 location=None,
                 rate: float = 1.0,
                 pre_wet: bool = None,
                 ms_pause: float = None,
                 top_clearance=None,
                 bottom_clearance=None,
                 manual_liquid_volume_allowance=None):
        pass

    @abstractmethod
    def dispense(self,
                 volume: float = None,
                 location=None,
                 rate: float = 1.0,
                 full_dispense: bool = False,
                 top_clearance=None,
                 bottom_clearance=None,
                 manual_liquid_volume_allowance=None):
        pass

    @abstractmethod
    def is_mix_in_progress(self):
        pass

    @abstractmethod
    def dwell(self, seconds=0, minutes=0):
        pass

    #-------------------------------------------------------------------------------------------------------------------
    # Utility
    #-------------------------------------------------------------------------------------------------------------------

    def _pre_wet(self, well: EnhancedWell, volume, location, rate, pre_wet: bool):
        if pre_wet is None:
            if tls.aspirate_params_transfer:
                pre_wet = tls.aspirate_params_transfer.pre_wet_transfer
        if pre_wet is None:
            pre_wet = self.config.aspirate.pre_wet.default
        if pre_wet and self.config.enable_enhancements:
            if self.tip_wetness is TipWetness.DRY:
                pre_wet_volume = min(
                    self.get_max_volume() * self.config.aspirate.pre_wet.max_volume_fraction,
                    max(volume, well.liquid_volume.available_volume_min))
                pre_wet_rate = self.config.aspirate.pre_wet.rate_func(rate)
                self.tip_wetness = TipWetness.WETTING
                def do_pre_wet():
                    for i in range(self.config.aspirate.pre_wet.count):
                        self.aspirate(volume=pre_wet_volume, location=location, rate=pre_wet_rate, pre_wet=False, ms_pause=0)
                        self.dispense(volume=pre_wet_volume, location=location, rate=pre_wet_rate, full_dispense=(i+1 == self.config.aspirate.pre_wet.count))
                info_while(pretty.format('prewetting tip in well {0} vol={1:n}', well.get_name(), pre_wet_volume), do_pre_wet)
                self.tip_wetness = TipWetness.WET

    def pause_after_aspirate(self, ms_pause):
        # if we're asked to, pause after aspiration to let liquid rise
        if ms_pause is None:
            if tls.aspirate_params_transfer:
                ms_pause = tls.aspirate_params_transfer.ms_pause_transfer
        if ms_pause is None:
            ms_pause = self.config.aspirate.pause.ms_default
        if self.config.enable_enhancements and ms_pause > 0 and not self.is_mix_in_progress():
            self.dwell(seconds=ms_pause / 1000.0)

    def _top_clearance(self, liquid_depth, clearance):
        assert liquid_depth >= 0
        if clearance > 0:
            return liquid_depth + clearance  # going up
        else:
            return liquid_depth + clearance  # going down. we used to clamp to at least a fraction of the current liquid depth, but not worthwhile as tube modelling accuracy has improved

    def _adjust_location_to_liquid_top(self, location=None, aspirate_volume=None, top_clearance=None, bottom_clearance=None, allow_above=False, manual_liquid_volume_allowance=0):
        if isinstance(location, (EnhancedWellV1, EnhancedWellV2)):
            well = location
            current_liquid_volume = well.liquid_volume.current_volume_min
            # if the well isn't machine made, don't go so close to the top
            if not well.liquid_volume.made_by_machine:
                current_liquid_volume = current_liquid_volume * (1 - manual_liquid_volume_allowance)
            liquid_depth = well.geometry.liquid_depth_from_volume_min(current_liquid_volume if aspirate_volume is None else current_liquid_volume - aspirate_volume)
            z = self._top_clearance(liquid_depth=liquid_depth, clearance=(0 if top_clearance is None else top_clearance))
            if bottom_clearance is not None:
                z = max(z, bottom_clearance)
            if not allow_above:
                z = min(z, well.well_depth)
            result = well.bottom(z)
        else:
            result = location  # we already had a displacement baked in to the location, don't adjust (when does this happen?)
        return result

