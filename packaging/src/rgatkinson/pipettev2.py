#
# pipettev2.py
#
from typing import Union, Sequence, Callable

from opentrons import types
from opentrons.hardware_control import  Pipette as HwPipette, SHAKE_OFF_TIPS_DROP_DISTANCE
from opentrons.protocol_api import InstrumentContext, transfers
from opentrons.protocol_api.contexts import AdvancedLiquidHandling
from opentrons.protocol_api.labware import Well as WellV2, quirks_from_any_parent
from opentrons.protocol_api.util import Clearances

from rgatkinson.configuration import TopConfigurationContext
from rgatkinson.logging import info, log_while_core, pretty, warn
from rgatkinson.pipette import EnhancedPipette, DispenseParams
from rgatkinson.util import tls, is_close
from rgatkinson.well import EnhancedWellV2


class EnhancedPipetteV2(EnhancedPipette, InstrumentContext):

    #-------------------------------------------------------------------------------------------------------------------
    # Construction
    #-------------------------------------------------------------------------------------------------------------------

    @classmethod
    def hook(cls, config: TopConfigurationContext, parentInst: InstrumentContext):
        return cls(config, parentInst)

    def __new__(cls, config: TopConfigurationContext, parentInst: InstrumentContext):
        assert isinstance(parentInst, InstrumentContext)
        parentInst.__class__ = EnhancedPipetteV2
        return parentInst

    # noinspection PyMissingConstructor
    def __init__(self, config: TopConfigurationContext, parentInst: InstrumentContext):
        super(self).__init__(config)
        #
        # more to come
        #

    #-------------------------------------------------------------------------------------------------------------------
    # Accessing
    #-------------------------------------------------------------------------------------------------------------------

    def get_speeds(self):
        return {'aspirate': self.speed.aspirate,
                'dispense': self.speed.dispense,
                'blow_out': self.speed.blow_out}

    def get_flow_rates(self):
        return {'aspirate': self.flow_rate.aspirate,
                'dispense': self.flow_rate.dispense,
                'blow_out': self.flow_rate.blow_out}

    @property
    def well_bottom_clearance(self) -> Clearances:
        # We track changes in the config
        return Clearances(self.config.aspirate.bottom_clearance, self.config.dispense.bottom_clearance)

    @property
    def well_top_clearance(self) -> Clearances:
        return Clearances(self.config.aspirate.top_clearance, self.config.dispense.top_clearance)

    def get_max_volume(self):
        return self.max_volume

    #-------------------------------------------------------------------------------------------------------------------
    # Transfers
    #-------------------------------------------------------------------------------------------------------------------

    class ExtendedTransferArgs(object):
        def __init__(self, config, **kwargs):
            self.keep_last_tip: bool = False
            self.full_dispense: bool = True

    # New kw args:
    #   'keep_last_tip': if true, then tip is not dropped at end of transfer
    #   'full_dispense': if true, and if a dispense empties the pipette, then dispense to blow_out 'position' instead of 'bottom' position
    #   'allow_overspill': if true, we allow over spill (of disposal_vol) from one run of (asp, disp+) to the next
    #   'allow_blow_elision': if true, then blow-outs which are not logically needed (such as before a tip drop) are elided
    def transfer(self,
                 volume: Union[float, Sequence[float]],
                 source: AdvancedLiquidHandling,
                 dest: AdvancedLiquidHandling,
                 trash=True,
                 **kwargs) -> 'InstrumentContext':
        tls.extended_transfer_args = self.ExtendedTransferArgs(self.config, **kwargs)
        result = super().transfer(volume, source, dest, trash, **kwargs)
        tls.extended_transfer_args = None
        return result

    def _execute_transfer(self, plan: transfers.TransferPlan):
        result = super()._execute_transfer(plan)
        return result

    #-------------------------------------------------------------------------------------------------------------------
    # Aspirate and dispense
    #-------------------------------------------------------------------------------------------------------------------

    def aspirate(self,
                 volume: float = None,
                 location: Union[types.Location, EnhancedWellV2] = None,
                 rate: float = 1.0,
                 # remainder are added params
                 pre_wet: bool = None,
                 ms_pause: float = None,
                 top_clearance=None,
                 bottom_clearance=None,
                 manual_liquid_volume_allowance=None):

        # figure out where we're aspirating from
        # recapitulate super
        if isinstance(location, WellV2):
            point, well = location.bottom()
            dest = types.Location(point + types.Point(0, 0, self.well_bottom_clearance.aspirate), well)
        elif isinstance(location, types.Location):
            dest = location
        elif location is not None:
            raise TypeError('location should be a Well or Location, but it is {}'.format(location))
        elif self._ctx.location_cache:
            dest = self._ctx.location_cache
        else:
            raise RuntimeError("If aspirate is called without an explicit location, another method that moves to a  location (such as move_to or dispense) must previously have been called so the robot knows where it is.")

        location = dest  # no need for new variable
        assert isinstance(location, types.Location)
        point, well = location

        if top_clearance is None:
            if tls.aspirate_params_transfer:
                top_clearance = tls.aspirate_params_transfer.top_clearance_transfer
            if top_clearance is None:
                top_clearance = self.well_top_clearance.aspirate
        if bottom_clearance is None:
            if tls.aspirate_params_transfer:
                bottom_clearance = tls.aspirate_params_transfer.bottom_clearance_transfer
            if bottom_clearance is None:
                bottom_clearance = self.well_bottom_clearance.aspirate
        if manual_liquid_volume_allowance is None:
            if tls.aspirate_params_transfer:
                manual_liquid_volume_allowance = tls.aspirate_params_transfer.manual_manufacture_tolerance_transfer
            if manual_liquid_volume_allowance is None:
                manual_liquid_volume_allowance = self.config.aspirate.manual_liquid_volume_allowance

        current_liquid_volume = well.liquid_volume.current_volume_min
        needed_liquid_volume = well.geometry.min_aspiratable_volume + volume
        if current_liquid_volume < needed_liquid_volume:
            msg = pretty.format('aspirating too much from well={0} have={1:n} need={2:n}', well.get_name(), current_liquid_volume, needed_liquid_volume)
            warn(msg)

        self._pre_wet(well, volume, location, rate, pre_wet)
        location = self._adjust_location_to_liquid_top(location=location, aspirate_volume=volume, top_clearance=top_clearance, bottom_clearance=bottom_clearance, manual_liquid_volume_allowance=manual_liquid_volume_allowance)

        def call_super():
            super(EnhancedPipette, self).aspirate(volume=volume, location=location, rate=rate)
        self.use_self_while(call_super)

        self.pause_after_aspirate(ms_pause)

        # finish up todo: what if we're doing an air gap
        well.liquid_volume.aspirate(volume)
        if volume != 0:
            self.prev_aspirated_well = well

    def dispense(self,
                 volume=None,
                 location=None,
                 rate=1.0,
                 # remainder are added params
                 full_dispense: bool = False,
                 top_clearance=None,
                 bottom_clearance=None,
                 manual_liquid_volume_allowance=None):
        # figure out where we're dispensing to
        # recapitulate super
        if isinstance(location, WellV2):
            if 'fixedTrash' in quirks_from_any_parent(location):
                loc = location.top()
            else:
                point, well = location.bottom()
                loc = types.Location(point + types.Point(0, 0, self.well_bottom_clearance.dispense), well)
            self.move_to(loc)
        elif isinstance(location, types.Location):
            loc = location
            self.move_to(location)
        elif location is not None:
            raise TypeError('location should be a Well or Location, but it is {}'.format(location))
        elif self._ctx.location_cache:
            loc = self._ctx.location_cache
        else:
            raise RuntimeError("If dispense is called without an explicit location, another method that moves to a location (such as move_to or aspirate) must previously have been called so the robot knows where it is.")

        location = loc  # no need for new variable
        assert isinstance(location, types.Location)
        point, well = location

        if top_clearance is None:
            if tls.dispense_params_transfer:
                top_clearance = tls.dispense_params_transfer.top_clearance_transfer
            if top_clearance is None:
                top_clearance = self.well_top_clearance.dispense
        if bottom_clearance is None:
            if tls.dispense_params_transfer:
                bottom_clearance = tls.dispense_params_transfer.bottom_clearance_transfer
            if bottom_clearance is None:
                bottom_clearance = self.well_bottom_clearance.dispense
        if manual_liquid_volume_allowance is None:
            if tls.dispense_params_transfer:
                manual_liquid_volume_allowance = tls.dispense_params_transfer.manual_manufacture_tolerance_transfer
            if manual_liquid_volume_allowance is None:
                manual_liquid_volume_allowance = self.config.dispense.manual_liquid_volume_allowance

        if is_close(volume, self.current_volume):  # avoid finicky floating-point precision issues
            volume = self.current_volume
        location = self._adjust_location_to_liquid_top(location=location, aspirate_volume=None, top_clearance=top_clearance, bottom_clearance=bottom_clearance, manual_liquid_volume_allowance=manual_liquid_volume_allowance)

        with DispenseParams():
            tls.dispense_params.full_dispense_from_dispense = full_dispense

            def call_super():
                super(EnhancedPipette, self).dispense(volume=volume, location=location, rate=rate)
            self.use_self_while(call_super)

            if tls.dispense_params.fully_dispensed:
                assert self.current_volume == 0
                if self.current_volume == 0:
                    pass  # nothing to do: the next self._position_for_aspirate will reposition for us: 'if pipette is currently empty, ensure the plunger is at "bottom"'
                else:
                    raise NotImplementedError

            # track volume
            well.liquid_volume.dispense(volume)

    def plunger_position(self, instr: HwPipette, ul: float, action: str, call_super: Callable) -> float:
        mm_from_vol = call_super()
        result = mm_from_vol

        if action == 'dispense':
            assert tls.dispense_params
            if self.config.enable_enhancements and \
                    (tls.dispense_params.full_dispense_from_dispense or
                     (tls.dispense_params_transfer and tls.dispense_params_transfer.full_dispense_transfer)):
                result = mm_from_blow = instr.config.blow_out
                info(pretty.format('full dispensing to mm={0:n} instead of mm={1:n}', mm_from_blow, mm_from_vol))
                tls.dispense_params.fully_dispensed = True
            else:
                result = call_super(action)
                tls.dispense_params.fully_dispensed = False

        return result

    #-------------------------------------------------------------------------------------------------------------------
    # Blow outs
    #-------------------------------------------------------------------------------------------------------------------

    def blow_out(self, location: Union[types.Location, WellV2] = None) -> 'InstrumentContext':
        result = super().blow_out(location)
        self._shake_tip(location)
        return result

    def _shake_tip(self, location):
        shake_off_dist = SHAKE_OFF_TIPS_DROP_DISTANCE / 2  # / 2 == less distance than shaking off tips
        if location:
            _, well = self.point_and_well(location)
            # ensure the distance is not >25% the diameter of well
            x = well.x_size()
            if x != 0:  # trash well has size zero
                shake_off_dist = min(shake_off_dist, x / 4)
        self._hw_manager.enhanced_api().shake_off_tips_drop(self._mount, shake_off_dist * 4)  # *4 because api will div by 4

    #-------------------------------------------------------------------------------------------------------------------
    # Delaying
    #-------------------------------------------------------------------------------------------------------------------

    def dwell(self, seconds=0, minutes=0):
        # like delay() but synchronous with the back-end

        msg = pretty.format('Dwelling for {0:n}m {1:n}s', minutes, seconds)

        minutes += int(seconds / 60)
        seconds = seconds % 60
        seconds += float(minutes * 60)

        def do_dwell():
            self.config.protocol_context.pause()
            self._hw_manager.enhanced_api().dwell(seconds)
            self.config.protocol_context.resume()

        log_while_core(msg, do_dwell)

    #-------------------------------------------------------------------------------------------------------------------
    # Utility
    #-------------------------------------------------------------------------------------------------------------------

    def point_and_well(self, location: Union[types.Location, WellV2]):
        if isinstance(location, WellV2):
            labware, point = location.top()
        elif isinstance(location, types.Location):
            labware, point = location
        else:
            raise ValueError("can't unpack")
        return types.Location(point, labware)

    def use_self_while(self, func):
        prev = tls.enhanced_pipette
        tls.enhanced_pipette = self
        result = func()
        tls.enhanced_pipette = prev
        return result
