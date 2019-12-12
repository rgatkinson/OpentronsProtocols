#
# pipettev1.py
#
import math
import random
from typing import List

from opentrons import instruments, robot
from opentrons.helpers import helpers
from opentrons.legacy_api.instruments import Pipette
from opentrons.legacy_api.instruments.pipette import SHAKE_OFF_TIPS_DROP_DISTANCE, SHAKE_OFF_TIPS_SPEED
from opentrons.trackers import pose_tracker
from opentrons.util.vector import Vector

from rgatkinson.configuration import TopConfigurationContext
from rgatkinson.logging import log_while_core, info, warn, info_while, log_while, pretty
from rgatkinson.pipette import EnhancedPipette, AspirateParamsTransfer, DispenseParamsTransfer, DispenseParams
from rgatkinson.types import TipWetness
from rgatkinson.tls import tls
from rgatkinson.collection_util import well_vector
from rgatkinson.math_util import zeroify, is_close, infinity
from rgatkinson.well import EnhancedWellV1


class EnhancedPipetteV1(EnhancedPipette, Pipette):

    #-------------------------------------------------------------------------------------------------------------------
    # Hack management
    #-------------------------------------------------------------------------------------------------------------------

    perf_hacks_installed = False
    use_perf_hacked_move = False
    @classmethod
    def install_perf_hacks(cls):
        if not cls.perf_hacks_installed:
            cls.use_perf_hacked_move = True
            cls.perf_hacks_installed = True

    #-------------------------------------------------------------------------------------------------------------------
    # Construction
    #-------------------------------------------------------------------------------------------------------------------

    @classmethod
    def hook(cls, config: TopConfigurationContext, parentInst: Pipette):
        return cls(config, parentInst)

    def __new__(cls, config: TopConfigurationContext, parentInst: Pipette):
        assert isinstance(parentInst, Pipette)
        parentInst.__class__ = EnhancedPipetteV1
        return parentInst

    # noinspection PyMissingConstructor
    def __init__(self, config: TopConfigurationContext, parentInst: Pipette):
        super(EnhancedPipetteV1, self).__init__(config)

        # load the config (again) in order to extract some more data later
        from opentrons.config import pipette_config
        from opentrons.config.pipette_config import configs
        pipette_model_version, pip_id = instruments._pipette_details(self.mount, self.name)
        self.pipette_config = pipette_config.load(pipette_model_version, pip_id)
        if not hasattr(self.pipette_config, 'drop_tip_min'):  # future-proof
            cfg = configs[pipette_model_version]  # ignores the id-based overrides done by pipette_config.load, but we can live with that
            self.pipette_config_drop_tip_min = cfg['dropTip']['min']  # hack: can't add field to pipette_config, so we do it this way
        else:
            self.pipette_config_drop_tip_min = self.pipette_config.drop_tip_min

        # try to mitigate effects of static electricity on small pipettes: they can cling to the tip on drop, causing disasters when next tips are picked up
        if self.config.enable_enhancements and self.name == 'p10_single':
            # dropping twice probably will help
            # if 'doubleDropTip' not in self.quirks:  # not necessary, in the end, it seems
            #     self.quirks.append('doubleDropTip')
            # plunging lower also helps, clearly
            self.plunger_positions['drop_tip'] = self.pipette_config_drop_tip_min

    #-------------------------------------------------------------------------------------------------------------------
    # Rates and speeds
    #-------------------------------------------------------------------------------------------------------------------

    def _get_speed(self, func: str):
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

    def get_max_volume(self):
        return self.max_volume

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
    #   'allow_overspill': if true, we allow over spill (of disposal_vol) from one run of (asp, disp+) to the next
    #   'allow_blow_elision': if true, then blow-outs which are not logically needed (such as before a tip drop) are elided
    def _run_transfer_plan(self, tips, plan, **kwargs):
        air_gap = kwargs.get('air_gap', 0)
        touch_tip = kwargs.get('touch_tip', False)
        is_distribute = kwargs.get('mode', 'transfer') == 'distribute'

        total_transfers = len(plan)
        seen_aspirate = False
        assert len(plan) == 0 or plan[0].get('aspirate')  # first step must be an aspirate

        step_info_map = dict()

        with AspirateParamsTransfer(self.config.aspirate):
            with DispenseParamsTransfer(self.config.dispense):

                for i, step in enumerate(plan):
                    aspirate = step.get('aspirate')
                    dispense = step.get('dispense')

                    if aspirate:
                        # *always* record on aspirates so we can test has_disposal_vol on subsequent dispenses
                        have_disposal_vol = self.has_disposal_vol(plan, i, step_info_map, **kwargs)

                        # we might have overspill from a previous transfer.
                        if self.current_volume > 0:
                            info(pretty.format('carried over {0:n} uL from prev operation', self.current_volume))

                        if not seen_aspirate:
                            assert i == 0

                            if kwargs.get('pre_wet', None) and kwargs.get('mix_before', None):
                                warn("simultaneous use of 'pre_wet' and 'mix_before' is not tested")

                            if (kwargs.get('allow_overspill', self.config.allow_overspill_default) and self.config.enable_enhancements) and zeroify(self.current_volume) > 0:
                                this_aspirated_well, __ = well_vector(aspirate['location'])
                                if self.prev_aspirated_well is this_aspirated_well:
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
                                        info(pretty.format("overspill of {0:n} uL isn't for disposal", self.current_volume))
                                        extra = self.current_volume
                                else:
                                    # different locations; can't re-use
                                    info('this aspirate is from location different than current pipette contents')
                                    extra = self.current_volume
                                if zeroify(extra) > 0:
                                    # quiet_log('blowing out overspill of %s uL' % format_number(self.current_volume))
                                    self._blowout_during_transfer(loc=None, **kwargs)  # loc isn't actually used

                            elif zeroify(self.current_volume) > 0:
                                info(pretty.format('blowing out unexpected overspill of {0:n} uL', self.current_volume))
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

                        tls.aspirate_params_transfer.sequester(kwargs)
                        self._aspirate_during_transfer(aspirate['volume'], aspirate['location'], **kwargs)

                    if dispense:
                        if self.current_volume < dispense['volume']:
                            warn(pretty.format('current {0:n} uL will truncate dispense of {1:n} uL', self.current_volume, dispense['volume']))

                        can_full_dispense = self.current_volume - dispense['volume'] <= 0

                        tls.dispense_params_transfer.sequester(kwargs, can_full_dispense)
                        self._dispense_during_transfer(dispense['volume'], dispense['location'], **kwargs)

                        do_touch = touch_tip or touch_tip is 0
                        is_last_step = step is plan[-1]
                        if is_last_step or plan[i + 1].get('aspirate'):
                            do_drop = not is_last_step or not (kwargs.get('keep_last_tip', False) and self.config.enable_enhancements)
                            # original always blew here. there are several reasons we could still be forced to blow
                            do_blow = not is_distribute  # other modes (are there any?) we're not sure about
                            do_blow = do_blow or kwargs.get('blow_out', False)  # for compatibility
                            do_blow = do_blow or do_touch  # for compatibility
                            do_blow = do_blow or not (kwargs.get('allow_blow_elision', self.config.allow_blow_elision_default) and self.config.enable_enhancements)
                            if not do_blow:
                                if is_last_step:
                                    if self.current_volume > 0:
                                        if not (kwargs.get('allow_overspill', self.config.allow_overspill_default) and self.config.enable_enhancements):
                                            do_blow = True
                                        elif self.current_volume > kwargs.get('disposal_vol', 0):
                                            warn(pretty.format('carried over {0:n} uL to next operation', self.current_volume))
                                        else:
                                            info(pretty.format('carried over {0:n} uL to next operation', self.current_volume))
                                else:
                                    # if we can, account for any overspill in the next aspirate
                                    if self.current_volume > 0:
                                        if self.has_disposal_vol(plan, i + 1, step_info_map, **kwargs):
                                            next_aspirate = plan[i + 1].get('aspirate'); assert next_aspirate
                                            next_aspirated_well, __ = well_vector(next_aspirate['location'])
                                            if self.prev_aspirated_well is next_aspirated_well:
                                                new_aspirate_vol = zeroify(next_aspirate.get('volume') - self.current_volume)
                                                if new_aspirate_vol == 0 or new_aspirate_vol >= self.min_volume:
                                                    next_aspirate['volume'] = new_aspirate_vol
                                                    info(
                                                        pretty.format('reduced next aspirate by {0:n} uL', self.current_volume))
                                                else:
                                                    do_blow = True
                                            else:
                                                do_blow = True  # different aspirate locations
                                        else:
                                            # Next aspirate doesn't *want* our overspill, so we don't reduce his
                                            # volume. But it's harmless to just leave the overspill present; might
                                            # be useful down the line
                                            pass
                                    else:
                                        pass  # currently empty
                            if do_touch:
                                self.touch_tip(touch_tip)
                            if do_blow:
                                self._blowout_during_transfer(dispense['location'], **kwargs)
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

    def aspirate(self, volume=None, location=None, rate: float = 1.0, pre_wet: bool = None, ms_pause: float = None, top_clearance=None, bottom_clearance=None, manual_liquid_volume_allowance=None):
        if not helpers.is_number(volume):  # recapitulate super
            if volume and not location:
                location = volume
            volume = self._working_volume - self.current_volume
        location = location if location else self.previous_placeable
        well, _ = well_vector(location)

        if top_clearance is None:
            if tls.aspirate_params_transfer:
                top_clearance = tls.aspirate_params_transfer.top_clearance_transfer
            if top_clearance is None:
                top_clearance = self.config.aspirate.top_clearance
        if bottom_clearance is None:
            if tls.aspirate_params_transfer:
                bottom_clearance = tls.aspirate_params_transfer.bottom_clearance_transfer
            if bottom_clearance is None:
                bottom_clearance = self.config.aspirate.bottom_clearance
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
        self._update_pose_tree_in_place(call_super)

        self.pause_after_aspirate(ms_pause)

        # finish up todo: what if we're doing an air gap
        well, __ = well_vector(location)
        well.liquid_volume.aspirate(volume)
        if volume != 0:
            self.prev_aspirated_well = well

    def _aspirate_during_transfer(self, vol, loc, **kwargs):
        assert tls.aspirate_params_transfer
        tls.aspirate_params_transfer.unsequester(kwargs)
        super()._aspirate_during_transfer(vol, loc, **kwargs)  # might 'mix_before' todo: is that ok? seems like it is...
        tls.aspirate_params_transfer.clear()

    def dispense(self, volume=None, location=None, rate=1.0, full_dispense: bool = False, top_clearance=None, bottom_clearance=None, manual_liquid_volume_allowance=None):
        if not helpers.is_number(volume):  # recapitulate super
            if volume and not location:
                location = volume
            volume = self._working_volume - self.current_volume
        location = location if location else self.previous_placeable
        well, _ = well_vector(location)

        if top_clearance is None:
            if tls.dispense_params_transfer:
                top_clearance = tls.dispense_params_transfer.top_clearance_transfer
            if top_clearance is None:
                top_clearance = self.config.dispense.top_clearance
        if bottom_clearance is None:
            if tls.dispense_params_transfer:
                bottom_clearance = tls.dispense_params_transfer.bottom_clearance_transfer
            if bottom_clearance is None:
                bottom_clearance = self.config.dispense.bottom_clearance
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
            self._update_pose_tree_in_place(call_super)

            if tls.dispense_params.fully_dispensed:
                assert self.current_volume == 0
                if self.current_volume == 0:
                    pass  # nothing to do: the next self._position_for_aspirate will reposition for us: 'if pipette is currently empty, ensure the plunger is at "bottom"'
                else:
                    raise NotImplementedError

            # track volume
            well, __ = well_vector(location)
            well.liquid_volume.dispense(volume)

    def _dispense_during_transfer(self, vol, loc, **kwargs):
        tls.dispense_params_transfer.unsequester(kwargs)
        super()._dispense_during_transfer(vol, loc, **kwargs)  # might 'mix_after' todo: is that ok? probably: we'd just do full_dispense on all of those too?
        tls.dispense_params_transfer.clear()

    def _dispense_plunger_position(self, ul):
        assert tls.dispense_params
        mm_from_vol = super()._dispense_plunger_position(ul)  # retrieve position historically used
        if self.config.enable_enhancements and \
                (tls.dispense_params.full_dispense_from_dispense or
                 (tls.dispense_params_transfer and tls.dispense_params_transfer.full_dispense_transfer)):
            mm_from_blow = self._get_plunger_position('blow_out')
            info(pretty.format('full dispensing to mm={0:n} instead of mm={1:n}', mm_from_blow, mm_from_vol))
            tls.dispense_params.fully_dispensed = True
            return mm_from_blow
        else:
            tls.dispense_params.fully_dispensed = False
            return mm_from_vol

    #-------------------------------------------------------------------------------------------------------------------
    # Movement
    #-------------------------------------------------------------------------------------------------------------------

    def _update_pose_tree_in_place(self, func):
        if not self.use_perf_hacked_move:
            return func()
        else:
            tls.update_pose_tree_in_place = True
            result = func()
            tls.update_pose_tree_in_place = False
            return result

    def _move(self, pose_tree, x=None, y=None, z=None):
        # In this hacked version, we make the assumption that copying of the pose_tree isn't necessary; we can instead update in place.
        # This is reasonable because all extant callers of _move() are of the stylized form: obj.pose_tree = pip._move(obj.pose_tree, ...)
        def func():
            return super(EnhancedPipetteV1, self)._move(pose_tree, x=x, y=y, z=z)
        return self._update_pose_tree_in_place(func)

    #-------------------------------------------------------------------------------------------------------------------
    # Tip Management
    #-------------------------------------------------------------------------------------------------------------------

    @property
    def current_tip_overlap(self):  # tip_overlap with respect to the current tip
        assert self.tip_attached
        d = self.pipette_self.config.tip_overlap
        try:
            return d[self.current_tip_tiprack.uri]
        except (KeyError, AttributeError):
            return d['default']

    @property
    def full_tip_length(self):
        return self.current_tip_tiprack.tip_length

    @property
    def nominal_available_tip_length(self):
        return self.full_tip_length - self.current_tip_overlap

    @property
    def calibrated_available_tip_length(self):
        return self._tip_length  # tiprack is implicit (2019.10.19), as only one kind of tip is calibrated

    @property
    def available_tip_length(self):
        return self.calibrated_available_tip_length

    @property
    def current_tip_tiprack(self):  # tiprack of the current tip
        assert self.tip_attached
        return self.current_tip().parent

    #-------------------------------------------------------------------------------------------------------------------

    def tip_coords_absolute(self):
        xyz = pose_tracker.absolute(self.robot.poses, self)
        return Vector(xyz)

    def pick_up_tip(self, location=None, presses=None, increment=None):
        if self.tip_attached:
            info('pick_up_tip(): tip is already attached')
        result = super().pick_up_tip(location, presses, increment)
        self.tip_wetness = TipWetness.DRY
        return result

    def drop_tip(self, location=None, home_after=True):
        if not self.tip_attached:
            info('drop_tip(): no tip attached')
        result = super().drop_tip(location, home_after)
        self.tip_wetness = TipWetness.NONE
        return result

    def done_tip(self):  # a handy little utility that looks at self.config.trash_control
        if self.tip_attached:
            if self.current_volume > 0:
                info(pretty.format('{0} has {1:n} uL remaining', self.name, self.current_volume))
            if self.config.trash_control:
                self.drop_tip()
            else:
                self.return_tip()

    #-------------------------------------------------------------------------------------------------------------------
    # Blow outs
    #-------------------------------------------------------------------------------------------------------------------

    def blow_out(self, location=None):
        def func():
            # calls to mover.move() in super() used stylized pose-tree management
            super().blow_out(location)
        self._update_pose_tree_in_place(func)
        self._shake_tip(location)  # try to get rid of pesky retentive drops

    def _shake_tip(self, location):
        # Modelled after Pipette._shake_off_tips()
        shake_off_distance = SHAKE_OFF_TIPS_DROP_DISTANCE / 2  # / 2 == less distance than shaking off tips
        if location:
            placeable, _ = well_vector(location)
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
    # Delaying
    #-------------------------------------------------------------------------------------------------------------------

    def dwell(self, seconds=0, minutes=0):
        # like delay() but synchronous with the back-end

        msg = pretty.format('Dwelling for {0:n}m {1:n}s', minutes, seconds)

        minutes += int(seconds / 60)
        seconds = seconds % 60
        seconds += float(minutes * 60)

        def do_dwell():
            self.robot.pause()
            self.robot._driver.delay(seconds)
            self.robot.resume()

        log_while_core(msg, do_dwell)

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
                    ms_pause=None,
                    ms_final_pause=None,
                    aspirate_rate=None,
                    dispense_rate=None,
                    initial_turnover=None,
                    max_tip_cycles=None,
                    pre_wet=None,
                    top_clearance=None,
                    bottom_clearance=None
                    ):

        def do_layered_mix():
            self.begin_layered_mix()
            local_keep_last_tip = keep_last_tip if keep_last_tip is not None else self.config.layered_mix.keep_last_tip

            for well in wells:
                self._layered_mix_one(well, msg=msg,
                                      count=count,
                                      min_incr=min_incr,
                                      incr=incr,
                                      count_per_incr=count_per_incr,
                                      volume=volume,
                                      ms_pause=ms_pause,
                                      ms_final_pause=ms_final_pause,
                                      apirate_rate=aspirate_rate,
                                      dispense_rate=dispense_rate,
                                      initial_turnover=initial_turnover,
                                      max_tip_cycles=max_tip_cycles,
                                      pre_wet=pre_wet,
                                      top_clearance=top_clearance,
                                      bottom_clearance=bottom_clearance
                                      )
            if not local_keep_last_tip:
                self.done_tip()
            self.end_layered_mix()

        log_while(f'{msg} {[well.get_name() for well in wells]}', do_layered_mix)

    def _layered_mix_one(self, well: EnhancedWellV1, msg, **kwargs):
        def fetch(name, default=None):
            if default is None:
                default = getattr(self.config.layered_mix, name, None)
            result = kwargs.get(name, default)
            if result is None:
                result = default  # replace any explicitly stored 'None' with default
            return result

        count = fetch('count')
        min_incr = fetch('min_incr')
        incr = fetch('incr')
        count_per_incr = fetch('count_per_incr')
        volume = fetch('volume', self.max_volume)
        ms_pause = fetch('ms_pause')
        ms_final_pause = fetch('ms_final_pause')
        aspirate_rate = fetch('aspirate_rate', self.config.layered_mix.aspirate_rate_factor)
        dispense_rate = fetch('dispense_rate', self.config.layered_mix.dispense_rate_factor)
        initial_turnover = fetch('initial_turnover')
        max_tip_cycles = fetch('max_tip_cycles', infinity)
        pre_wet = fetch('pre_wet', False)  # not much point in pre-wetting during mixing; save some time, simpler. but we do so if asked
        top_clearance = fetch('top_clearance')
        bottom_clearance = fetch('bottom_clearance')

        current_liquid_volume = well.liquid_volume.current_volume_min
        liquid_depth = well.geometry.liquid_depth_from_volume(current_liquid_volume)
        liquid_depth_after_asp = well.geometry.liquid_depth_from_volume(current_liquid_volume - volume)
        msg = pretty.format("{0:s} well='{1:s}' cur_vol={2:n} well_depth={3:n} after_aspirate={4:n}", msg, well.get_name(), current_liquid_volume, liquid_depth, liquid_depth_after_asp)

        def do_one():
            count_ = count
            y_min = y = bottom_clearance
            y_max = self._top_clearance(liquid_depth=liquid_depth_after_asp, clearance=top_clearance)
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
            looped = False
            while do_layer(y):
                looped = True
                if not first and ms_pause > 0:
                    self.dwell(seconds=ms_pause / 1000.0)  # pause to let dispensed liquid disperse
                #
                if first and initial_turnover is not None:
                    count_ = int(0.5 + (initial_turnover / volume))
                    count_ = max(count_, count_per_incr)
                else:
                    count_ = count_per_incr
                if not self.tip_attached:
                    self.pick_up_tip()

                radial_clearance_func = self.radial_clearance_manager.get_clearance_function(self, well)
                radial_clearance = 0 if radial_clearance_func is None or not self.config.layered_mix.enable_radial_randomness else radial_clearance_func(y_max)
                radial_clearance = max(0, radial_clearance - max(well.geometry.radial_clearance_tolerance, self.config.layered_mix.radial_clearance_tolerance))

                for i in range(count_):
                    tip_cycles += 1
                    need_new_tip = tip_cycles >= max_tip_cycles
                    full_dispense = need_new_tip or (not do_layer(y + y_incr) and i == count_ - 1)

                    theta = random.random() * (2 * math.pi)
                    _, dispense_coordinates = well.bottom(y_max)
                    random_offset = (radial_clearance * math.cos(theta), radial_clearance * math.sin(theta), 0)
                    dispense_location = (well, dispense_coordinates + random_offset)

                    self.aspirate(volume, well.bottom(y), rate=aspirate_rate, pre_wet=pre_wet)
                    self.move_to(well.bottom(y_max))  # ascend vertically from aspirate location
                    self.dispense(volume, dispense_location, rate=dispense_rate, full_dispense=full_dispense)
                    self.move_to(well.bottom(y_max))  # prepare for vertical descent on a subsequent aspirate

                    if need_new_tip:
                        self.done_tip()
                        tip_cycles = 0
                #
                y += y_incr
                first = False
            if looped and ms_final_pause > 0:
                self.dwell(seconds=ms_final_pause / 1000.0)

        info_while(msg, do_one)


def verify_well_locations(well_list: List[EnhancedWellV1], pipette: EnhancedPipetteV1):
    picked_tip = False
    if not pipette.tip_attached:
        pipette.pick_up_tip()
        picked_tip = True

    for well in well_list:
        move_to_loc = well.top()
        pipette.move_to(move_to_loc)
        #
        well_top_coords_absolute = well.top_coords_absolute()
        _, top_coords = well_vector(well.top())
        _, move_to_coords = well_vector(move_to_loc)
        intended_coords = well_top_coords_absolute + (move_to_coords - top_coords)
        tip_coords = pipette.tip_coords_absolute()
        #
        robot.pause(
            pretty.format('verify location: {0} in {1} loc={2} tip={3}', well.get_name(), well.parent.get_name(), intended_coords, tip_coords))

    if picked_tip:
        pipette.return_tip()  # we didn't dirty it, we can always re-use it todo: enhance return_tip() to adjust iterator so that next pick can pick up again