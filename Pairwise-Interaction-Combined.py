"""
@author Robert Atkinson
"""

import json
import math
import cmath
from abc import abstractmethod
from numbers import Number
from typing import List
from opentrons import labware, instruments, robot, modules, types
from opentrons.helpers import helpers
from opentrons.legacy_api.instruments import Pipette
from opentrons.legacy_api.instruments.pipette import SHAKE_OFF_TIPS_DISTANCE, SHAKE_OFF_TIPS_SPEED
from opentrons.legacy_api.containers.placeable import unpack_location, Well

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
evagreen_volumes = [1000]           # B1, B2, etc in screwcap rack

# Tip usage
p10_start_tip = 'A1'
p50_start_tip = 'A1'
trash_control = True

# Diluting each strand
strand_dilution_factor = 25.0 / 9.0  # per Excel worksheet
strand_dilution_vol = 1225

# Master mix
master_mix_buffer_vol = 1693.44
master_mix_evagreen_vol = 423.36
master_mix_common_water_vol = 705.6

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

# Mixing
simple_mix_vol = 50  # how much to suck and spew for mixing
simple_mix_count = 4

# Optimization Control
allow_blow_elision = True
allow_carryover = allow_blow_elision


########################################################################################################################
########################################################################################################################
##                                                                                                                    ##
## Extensions : this section can be reused across protocols                                                           ##
##                                                                                                                    ##
########################################################################################################################
########################################################################################################################

########################################################################################################################
# Well enhancements
########################################################################################################################

class PrimitiveUnknownNumber(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'unk(%s)' % hash(self)


# This is not fully elaborated for arithmetic
class UnknownNumber(Number):
    def __init__(self, offset=0, unknowns=None):
        if unknowns is None:
            unknowns = [PrimitiveUnknownNumber()]
        self.offset = offset
        self.unknowns = unknowns

    def __copy__(self):
        return UnknownNumber(self.offset, self.unknowns.copy())

    def __add__(self, other):
        result = self.__copy__()
        if isinstance(other, UnknownNumber):
            result.unknowns.extend(other.unknowns)
            result.offset += other.offset
        else:
            result.offset += other
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def formatted(self, *args, **kwargs):
        result = '('
        for i, unknown in enumerate(self.unknowns):
            if i != 0:
                result += ', '
            result += str(unknown)
        result += '+'
        result += format_number(self.offset, *args, **kwargs)
        result += ')'
        return result


# If we aspirate from
class WellVolume(object):
    def __init__(self, well):
        self.well = well
        self.initial_known = False
        self.initial = UnknownNumber()
        self.cum_delta = 0
        self.min_delta = 0
        self.max_delta = 0

    def set_initial(self, initial_volume):
        assert not self.initial_known
        assert self.cum_delta == 0
        self.initial_known = True
        self.initial = initial_volume

    def current(self):
        return self.initial + self.cum_delta

    def aspirate(self, volume):
        if not self.initial_known:
            self.set_initial(UnknownNumber())
        self._track_volume(-volume)

    def dispense(self, volume):
        if not self.initial_known:
            self.set_initial(0)
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


# Must keep in sync with Opentrons-Analyze
def noteLiquid(name, location, initial_volume=None):
    well, __ = unpack_location(location)
    assert isWell(well)
    d = {'name': name, 'location': get_location_path(well)}
    if initial_volume is not None:
        d['volume'] = initial_volume
        get_well_volume(well).set_initial(initial_volume)
    serialized = json.dumps(d).replace("{", "{{").replace("}", "}}")  # runtime calls comment.format(...) on our comment; avoid issues therewith
    robot.comment('Liquid: %s' % serialized)

########################################################################################################################

class WellGeometry(object):
    def __init__(self, well):
        self.well = well

    @abstractmethod
    def depthFromVolume(self, volume):
        pass


class UnknownWellGeometry(WellGeometry):
    def __init__(self, well):
        super().__init__(well)

    def depthFromVolume(self, volume):
        return UnknownNumber()


class IdtTubeWellGeometry(WellGeometry):
    def __init__(self, well):
        super().__init__(well)

    def depthFromVolume(self, volume):
        # Calculated from Mathematica models
        if volume <= 0.0:
            return 0.0
        if volume <= 57.8523:
            return 0.827389 * cube_root(volume)
        return 3.2 - 0.0184378 * (57.8523 - volume)


class Biorad96WellPlateWellGeometry(WellGeometry):
    def __init__(self, well):
        super().__init__(well)

    def depthFromVolume(self, volume):
        # Calculated from Mathematica models
        if volume <= 0.0:
            return 0.0
        if volume <= 60.7779:
            return -13.7243 + 4.24819 * cube_root(33.7175 + 1.34645 * volume)
        return 14.66 - 0.0427095 * (196.488 - volume)


class Eppendorf1point5mlTubeGeometry(WellGeometry):
    def __init__(self, well):
        super().__init__(well)

    def depthFromVolume(self, volume):
        # Calculated from Mathematica models
        if volume <= 12.2145:
            i = complex(0, 1)
            term = cube_root(36.6435 - 3. * volume + 1.73205 * cmath.sqrt(-73.2871 * volume + 3. * volume * volume))
            result = 1.8 - (2.98934 - 5.17768 * i) / term - (0.270963 + 0.469322 * i) * term
            return result.real
        if volume <= 445.995:
            return -8.22353 + 2.2996 * cube_root(53.0712 + 2.43507 * volume)
        return -564. + 49.1204 * cube_root(1580.62 + 0.143239 * volume)


class FalconTube15mlGeometry(WellGeometry):
    def __init__(self, well):
        super().__init__(well)

    def depthFromVolume(self, volume):
        # Calculated from Mathematica models
        if volume <= 0.0686291:
            return 0.0  # not correct, but not worth it right now to do correct value
        if volume <= 874.146:
            return -0.758658 + 1.23996 * cube_root(0.267715 + 5.69138 * volume)
        return -360.788 + 13.8562 * cube_root(19665.7 + 1.32258 * volume)


def get_well_geometry(well):
    assert isWell(well)
    try:
        return well.geometry
    except AttributeError:
        well.geometry = UnknownWellGeometry(well)
        return well.geometry


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
    #   'retain_tip'
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
                    info('carried over %s uL from prev operation' % format_number(self.current_volume))

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
                                    info('reduced this aspirate by %s uL' % format_number(self.current_volume))
                                    extra = 0  # can't blow out since we're relying on its presence in pipette
                                else:
                                    extra = self.current_volume - aspirate['volume']
                                    assert zeroify(extra) > 0
                            else:
                                info("carryover of %s uL isn't for disposal" % format_number(self.current_volume))
                                extra = self.current_volume
                        else:
                            # different locations; can't re-use
                            info('this aspirate is from location different than current pipette contents')
                            extra = self.current_volume
                        if zeroify(extra) > 0:
                            # quiet_log('blowing out carryover of %s uL' % format_number(self.current_volume))
                            self._blowout_during_transfer(loc=None, **kwargs)  # loc isn't actually used

                    elif zeroify(self.current_volume) > 0:
                        info('blowing out unexpected carryover of %s uL' % format_number(self.current_volume))
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
                    info('current %s uL with aspirate(has_disposal=%s) of %s uL would overflow capacity' % (
                          format_number(self.current_volume),
                          self.has_disposal_vol(plan, step_index, **kwargs),
                          format_number(aspirate['volume'])))
                    self._blowout_during_transfer(loc=None, **kwargs)  # loc isn't actually used
                self._aspirate_during_transfer(aspirate['volume'], aspirate['location'], **kwargs)

            if dispense:
                if self.current_volume < dispense['volume']:
                    warn('current %s uL will truncate dispense of %s uL' %(format_number(self.current_volume), format_number(dispense['volume'])))
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
                                    warn('carried over %s uL to next operation' % format_number(self.current_volume))
                                else:
                                    info('carried over %s uL to next operation' % format_number(self.current_volume))
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
                                            info('reduced next aspirate by %s uL' % format_number(self.current_volume))
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
        # save so super sees actual original parameters
        saved_volume = volume
        saved_location = location
        # recapitulate super
        if not helpers.is_number(volume):
            if volume and not location:
                location = volume
            volume = self._working_volume - self.current_volume
        display_location = location if location else self.previous_placeable
        # call super
        super().aspirate(volume=saved_volume, location=saved_location, rate=rate)
        # keep track of where we aspirated from and keep track of volume in wells
        well, __ = unpack_location(display_location)
        get_well_volume(well).aspirate(volume)
        if volume != 0:
            self.prev_aspirated_location = well

    def dispense(self, volume=None, location=None, rate=1.0):
        # save so super sees actual original parameters
        saved_volume = volume
        saved_location = location
        # recapitulate super
        if not helpers.is_number(volume):
            if volume and not location:
                location = volume
            volume = self._working_volume - self.current_volume
        display_location = location if location else self.previous_placeable
        # call super
        super().dispense(volume=saved_volume, location=saved_location, rate=rate)
        # keep track of volume in wells
        well, __ = unpack_location(display_location)
        get_well_volume(well).dispense(volume)

    def blow_out(self, location=None):
        super().blow_out(location)
        self._shake_tip(location)  # try to get rid of pesky retentive drops

    def _shake_tip(self, location):
        # Modelled after Pipette._shake_off_tips()
        shake_off_distance = SHAKE_OFF_TIPS_DISTANCE / 2  # less distance than shaking off tips
        if location:
            placeable, _ = unpack_location(location)
            # ensure the distance is not >25% the diameter of placeable
            x = placeable.x_size()
            if x != 0:  # trash well has size zero
                shake_off_distance = max(min(shake_off_distance, x / 4), 1.0)
        self.robot.gantry.push_speed()
        self.robot.gantry.set_speed(SHAKE_OFF_TIPS_SPEED / 2)  # less fast than shaking off tips
        self.robot.poses = self._jog(self.robot.poses, 'x', -shake_off_distance)  # move left
        self.robot.poses = self._jog(self.robot.poses, 'x', shake_off_distance * 2)  # move right
        self.robot.poses = self._jog(self.robot.poses, 'x', -shake_off_distance)  # move left
        self.robot.gantry.pop_speed()


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


def done_tip(pp):
    if pp.has_tip:
        if pp.current_volume > 0:
            info('%s has %s uL remaining' % (pp.name, format_number(pp.current_volume)))
        if trash_control:
            pp.drop_tip()
        else:
            pp.return_tip()


def format_number(value, precision=2):
    if isinstance(value, UnknownNumber):
        return value.formatted(precision=precision)
    factor = 1
    for i in range(precision):
        if value * factor == int(value * factor):
            precision = i
            break
        factor *= 10
    return "{:.{}f}".format(value, precision)


def cube_root(value):
    return pow(value, 1.0/3.0)

def zeroify(value, digits=2):  # clamps small values to zero, leaves others alone
    rounded = round(value, digits)
    return rounded if rounded == 0 else value


########################################################################################################################
########################################################################################################################
##                                                                                                                    ##
## Protocol                                                                                                           ##
##                                                                                                                    ##
########################################################################################################################
########################################################################################################################


########################################################################################################################
# Labware
########################################################################################################################

# Configure the tips
tips300a = labware.load('opentrons_96_tiprack_300ul', 1)
tips300b = labware.load('opentrons_96_tiprack_300ul', 4)
tips10 = labware.load('opentrons_96_tiprack_10ul', 7)

# Configure the pipettes. Blow out faster than default in an attempt to avoid hanging droplets on the pipettes after blowout
p10 = MyPipette(instruments.P10_Single(mount='left', tip_racks=[tips10]))
p50 = MyPipette(instruments.P50_Single(mount='right', tip_racks=[tips300a, tips300b]))
p10.set_flow_rate(blow_out=p10.get_flow_rates()['blow_out'] * 2)
p50.set_flow_rate(blow_out=p50.get_flow_rates()['blow_out'] * 2)

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
master_mix = falcon_rack['A1']

for well, __ in buffers:
    well.geometry = IdtTubeWellGeometry(well)
for well, __ in evagreens:
    well.geometry = IdtTubeWellGeometry(well)
diluted_strand_a.geometry = Eppendorf1point5mlTubeGeometry(diluted_strand_a)
diluted_strand_b.geometry = Eppendorf1point5mlTubeGeometry(diluted_strand_b)
master_mix.geometry = FalconTube15mlGeometry(master_mix)
for well in plate.wells():
    well.geometry = Biorad96WellPlateWellGeometry(well)


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

strand_dilution_source_vol = strand_dilution_vol / strand_dilution_factor
strand_dilution_water_vol = strand_dilution_vol - strand_dilution_source_vol


def simple_mix(wells, msg=None, count=simple_mix_count, volume=simple_mix_vol, pipette=p50, drop_tip=True):
    if msg is not None:
        log(msg)
    if not pipette.has_tip:
        pipette.pick_up_tip()
    for well in wells:
        pipette.mix(count, volume, well)
    if drop_tip:
        done_tip(pipette)

def layered_mix(wells, msg='Mixing', count=simple_mix_count, volume=simple_mix_vol, pipette=p50, drop_tip=True):
    for well in wells:
        _layered_mix_one(well, msg=msg, count=count, volume=volume, pipette=pipette)
    if drop_tip:
        done_tip(pipette)

def _layered_mix_one(well, msg, count, volume, pipette):
    mix_vol = volume
    mix_count = count
    well_vol = get_well_volume(well).current()
    well_depth = get_well_geometry(well).depthFromVolume(well_vol)
    well_depth_after_asp = get_well_geometry(well).depthFromVolume(well_vol - mix_vol)
    msg = "%s well='%s' cur_vol=%s well_depth=%s after_asp=%s" % (msg, well.get_name(), format_number(well_vol), format_number(well_depth), format_number(well_depth_after_asp))
    if msg is not None:
        log(msg)
    if not pipette.has_tip:
        pipette.pick_up_tip()
    y_min = y = 1.0  # 1.0 is default aspiration position from bottom
    y_max = well_depth_after_asp-max(1.0, well_depth_after_asp/10)  # 1.0 here is
    y_incr = (y_max - y_min) / mix_count
    while y <= y_max:
        log('asp=%s disp=%s' % (format_number(y), format_number(y_max)))
        pipette.aspirate(mix_vol, well.bottom(y))
        pipette.dispense(mix_vol, well.bottom(y_max))
        y += y_incr


def diluteStrands():
    log('Liquid Names')
    noteLiquid('Strand A', location=strand_a)
    noteLiquid('Strand B', location=strand_b)
    noteLiquid('Diluted Strand A', location=diluted_strand_a)
    noteLiquid('Diluted Strand B', location=diluted_strand_b)

    simple_mix([strand_a], 'Mixing Strand A')  # can't used layered_mix as we don't know the volume
    simple_mix([strand_b], 'Mixing Strand B')  # ditto

    # Create dilutions of strands
    log('Moving water for diluting Strands A and B')
    p50.transfer(strand_dilution_water_vol, water, [diluted_strand_a, diluted_strand_b],
                 new_tip='once',  # can reuse for all diluent dispensing since dest tubes are initially empty
                 trash=trash_control
                 )
    log('Diluting Strand A')
    p50.transfer(strand_dilution_source_vol, strand_a, diluted_strand_a, trash=trash_control, retain_tip=True)
    layered_mix([diluted_strand_a], 'Mixing Diluted Strand A', count=10)

    log('Diluting Strand B')
    p50.transfer(strand_dilution_source_vol, strand_b, diluted_strand_b, trash=trash_control, retain_tip=True)
    layered_mix([diluted_strand_b], 'Mixing Diluted Strand B', count=10)


def createMasterMix():
    noteLiquid('Master Mix', location=master_mix)
    for buffer in buffers:
        noteLiquid('Buffer', location=buffer[0], initial_volume=buffer[1])
    for evagreen in evagreens:
        noteLiquid('Evagreen', location=evagreen[0], initial_volume=evagreen[1])

    # Buffer was just unfrozen. Mix to ensure uniformity. EvaGreen doesn't freeze, no need to mix
    layered_mix([buffer for buffer, __ in buffers], "Mixing Buffers", count=10)

    # transfer from multiple source wells, each with a current defined volume
    def transferMultiple(ctx, xfer_vol_remaining, tubes, dest, new_tip, *args, **kwargs):
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

    # Mixes possibly several times, at different levels
    def mix_master_mix(current_volume, pipette=p50):
        log('Mixing Master Mix')
        layered_mix([master_mix], pipette=pipette, count=10, volume=pipette.max_volume)

    log('Creating Master Mix: Water')
    p50.transfer(master_mix_common_water_vol, water, master_mix, trash=trash_control)

    log('Creating Master Mix: Buffer')
    transferMultiple('Creating Master Mix: Buffer', master_mix_buffer_vol, buffers, master_mix, new_tip='once', retain_tip=True)  # 'once' because we've only got water & buffer in context
    mix_master_mix(current_volume=master_mix_common_water_vol + master_mix_buffer_vol)  # help eliminate air bubbles: smaller volume right now

    log('Creating Master Mix: EvaGreen')
    transferMultiple('Creating Master Mix: EvaGreen', master_mix_evagreen_vol, evagreens, master_mix, new_tip='always', retain_tip=True)  # 'always' to avoid contaminating the Evagreen source w/ buffer
    mix_master_mix(current_volume=master_mix_common_water_vol + master_mix_buffer_vol + master_mix_evagreen_vol)


########################################################################################################################
# Plating
########################################################################################################################

def plateEverything():
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
    done_tip(p10)
    done_tip(p50)

    # Plate strand B and mix
    # Mixing always needs the p50, but plating may need either; optimize tip usage
    log('Plating Strand B')
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
            if not p50.has_tip:
                p50.pick_up_tip()
            # total plated volume is some 84uL; we need to use a substantial fraction of that to get good mixing
            layered_mix([well], volume=50, pipette=p50)
            done_tip(p10)
            done_tip(p50)


########################################################################################################################
# Off to the races
########################################################################################################################

noteLiquid('Water', location=water)
diluteStrands()
createMasterMix()
plateEverything()
