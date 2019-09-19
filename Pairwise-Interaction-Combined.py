"""
@author Robert Atkinson
"""

import json
import math
from typing import List
from opentrons import labware, instruments, robot, modules, types
from opentrons.helpers import helpers
from opentrons.legacy_api.instruments import Pipette
from opentrons.legacy_api.containers.placeable import unpack_location

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
strand_dilution_vol = 1175

# Master mix
master_mix_buffer_vol = 1693.44
master_mix_evagreen_vol = 423.36
master_mix_common_water_vol = 705.6

# Define the volumes of diluted strands we will use
strand_volumes = [0, 2, 5, 8, 12, 16, 20, 28]
num_replicates = 3
columns_per_plate = 12
per_well_water_volumes = [
    [56, 54, 51, 48],
    [54, 52, 49, 46],
    [51, 49, 46, 43],
    [48, 46, 43, 40],
    [32, 28, 24, 16],
    [28, 24, 20, 12],
    [24, 20, 16, 8],
    [16, 12, 8, 0]]

# Mixing
simple_mix_vol = 50  # how much to suck and spew for mixing
simple_mix_count = 4

# Optimization control
use_blow_elision = True
use_carryover = True

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
                    quiet_log('aspirate-dispense is entire remaining plan')
            else:
                quiet_log('unexpected aspirate-aspirate sequence')
        return check_has_disposal_vol

    # Copied and overridden
    def _run_transfer_plan(self, tips, plan, **kwargs):
        air_gap = kwargs.get('air_gap', 0)
        touch_tip = kwargs.get('touch_tip', False)
        is_distribute = kwargs.get('mode', 'transfer') == 'distribute'

        total_transfers = len(plan)
        seen_aspirate = False
        assert len(plan) == 0 or plan[0].get('aspirate')  # first step must be an aspirate

        for step_index, step in enumerate(plan):

            aspirate = step.get('aspirate')
            dispense = step.get('dispense')

            if aspirate:
                # we might have carryover from a previous transfer.
                if self.current_volume > 0:
                    quiet_log('carried over %s uL from prev operation' % format_number(self.current_volume))

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
                                    quiet_log('reduced this aspirate by %s uL' % format_number(self.current_volume))
                                    extra = 0  # can't blow out since we're relying on its presence in pipette
                                else:
                                    extra = self.current_volume - aspirate['volume']
                                    assert zeroify(extra) > 0
                            else:
                                quiet_log("carryover of %s uL isn't for disposal" % format_number(self.current_volume))
                                extra = self.current_volume
                        else:
                            # different locations; can't re-use
                            quiet_log('this aspirate is from location different than current pipette contents')
                            extra = self.current_volume
                        if zeroify(extra) > 0:
                            # quiet_log('blowing out carryover of %s uL' % format_number(self.current_volume))
                            self._blowout_during_transfer(loc=None, **kwargs)  # loc isn't actually used

                    elif zeroify(self.current_volume) > 0:
                        warn('blowing out unexpected carryover of %s uL' % format_number(self.current_volume))
                        self._blowout_during_transfer(loc=None, **kwargs)  # loc isn't actually used

                seen_aspirate = True

                self._add_tip_during_transfer(tips, **kwargs)
                self._aspirate_during_transfer(aspirate['volume'], aspirate['location'], **kwargs)

            if dispense:
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
                                    quiet_log('carried over %s uL to next operation' % format_number(self.current_volume))
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
                                            quiet_log('reduced next aspirate by %s uL' % format_number(self.current_volume))
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
        super(MyPipette, self).aspirate(volume=saved_volume, location=saved_location, rate=rate)
        # keep track of where we aspirated from
        if volume != 0:
            self.prev_aspirated_location, __ = unpack_location(display_location)


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


########################################################################################################################
# Utilities
########################################################################################################################

# Returns a unique name for the given location. Must track in Opentrons-Analyze.
def get_location_path(location):
    return '/'.join(list(reversed([str(item)
                                   for item in location.get_trace(None)
                                   if str(item) is not None])))


def log(msg: str):
    robot.comment("*********** %s ***********" % msg)

def quiet_log(msg):
    info(msg, prefix='info:', suffix='')

def silent_log(msg):
    pass

def info(msg: str, prefix="***********", suffix=' ***********'):
    robot.comment("%s%s%s%s" % (prefix, '' if len(prefix) == 0 else ' ', msg, suffix))

def warn(msg: str, prefix="***********", suffix=' ***********'):
    robot.comment("%s%sWARNING: %s%s" % (prefix, '' if len(prefix) == 0 else ' ', msg, suffix))


def noteLiquid(name, location, volume=None):
    d = {'name': name, 'location': get_location_path(location)}
    if volume is not None:
        d['volume'] = volume
    serialized = json.dumps(d).replace("{", "{{").replace("}", "}}")  # runtime calls comment.format(...) on our comment; avoid issues therewith
    robot.comment('Liquid: %s' % serialized)


def done_tip(pp):
    if pp.current_volume > 0:
        quiet_log('%s has %s uL remaining' % (pp.name, format_number(pp.current_volume)))
    if trash_control:
        pp.drop_tip()
    else:
        pp.return_tip()


def format_number(value, precision=2):
    factor = 1
    for i in range(precision):
        if value * factor == int(value * factor):
            precision = i
            break
        factor *= 10
    return "{:.{}f}".format(value, precision)


def zeroify(value, digits=2):  # clamps small values to zero, leaves others alone
    rounded = round(value, digits)
    return rounded if rounded == 0 else value


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

def simple_mix(wells, msg=None, count=simple_mix_count, volume=simple_mix_vol, pipette=p50, pick_tip=True, drop_tip=True):
    if msg is not None:
        log(msg)
    assert pipette.has_tip != pick_tip  # if we have one, don't pick, and visa versa
    if pick_tip:
        pipette.pick_up_tip()
    for well in wells:
        pipette.mix(count, volume, well)
    if drop_tip:
        done_tip(pipette)

def diluteStrands():
    log('Liquid Names')
    noteLiquid('Strand A', location=strand_a)
    noteLiquid('Strand B', location=strand_b)
    noteLiquid('Diluted Strand A', location=diluted_strand_a)
    noteLiquid('Diluted Strand B', location=diluted_strand_b)

    simple_mix([strand_a], 'Mixing Strand A')
    simple_mix([strand_b], 'Mixing Strand B')

    # Create dilutions of strands
    log('Moving water for diluting Strands A and B')
    p50.transfer(strand_dilution_water_vol, water, [diluted_strand_a, diluted_strand_b],
                 new_tip='once',  # can reuse for all diluent dispensing since dest tubes are initially empty
                 trash=trash_control
                 )
    log('Diluting Strand A')
    p50.transfer(strand_dilution_source_vol, strand_a, diluted_strand_a, trash=trash_control, retain_tip=True)
    simple_mix([diluted_strand_a], 'Mixing Diluted Strand A', pick_tip=False)

    log('Diluting Strand B')
    p50.transfer(strand_dilution_source_vol, strand_b, diluted_strand_b, trash=trash_control, retain_tip=True)
    simple_mix([diluted_strand_b], 'Mixing Diluted Strand B', pick_tip=False)

def createMasterMix():
    noteLiquid('Master Mix', location=master_mix)
    for buffer in buffers:
        noteLiquid('Buffer', location=buffer[0], volume=buffer[1])
    for evagreen in evagreens:
        noteLiquid('Evagreen', location=evagreen[0], volume=evagreen[1])

    # Buffer was just unfrozen. Mix to ensure uniformity. EvaGreen doesn't freeze, no need to mix
    simple_mix([buffer for buffer, _ in buffers], "Mixing Buffers")

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
    def mix_master_mix(current_volume, pick_tip=True, pipette=p50):
        radius = master_mix.x_size() / 2.0
        area = math.pi * radius * radius  # area is square mm, current_volume is uL
        # We'll assume tube is cylindrical, which it isn't, but that's conservative
        current_height = current_volume / area  # in mm (that's how the units work out)
        step = 5.0  # mm
        clearance = 1.0  # mm, as in _position_for_aspirate
        height = min(master_mix.z_size(), clearance)  # as in _position_for_aspirate
        while height < current_height - 5.0:
            simple_mix([master_mix.bottom(height)], 'Mixing Master Mix @ %s' % format_number(height), pick_tip=pick_tip, drop_tip=False, pipette=pipette)  # WRONG
            height += step
        if pipette.has_tip:
            done_tip(p50)

    log('Creating Master Mix: Water')
    p50.transfer(master_mix_common_water_vol, water, master_mix, trash=trash_control)

    log('Creating Master Mix: Buffer')
    transferMultiple('Creating Master Mix: Buffer', master_mix_buffer_vol, buffers, master_mix, new_tip='once', retain_tip=True)  # 'once' because we've only got water & buffer in context
    mix_master_mix(current_volume=master_mix_common_water_vol + master_mix_buffer_vol, pick_tip=False)  # help eliminate air bubbles: smaller volume right now

    log('Creating Master Mix: EvaGreen')
    transferMultiple('Creating Master Mix: EvaGreen', master_mix_evagreen_vol, evagreens, master_mix, new_tip='always', retain_tip=True)  # 'always' to avoid contaminating the Evagreen source w/ buffer
    mix_master_mix(current_volume=master_mix_common_water_vol + master_mix_buffer_vol + master_mix_evagreen_vol, pick_tip=False)


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
    p50.pick_up_tip()
    for iRow in range(len(per_well_water_volumes)):
        for iCol in range(len(per_well_water_volumes[iRow])):
            volume = per_well_water_volumes[iRow][iCol]
            if volume != 0:
                p50.distribute(volume, water, plate.rows(iRow).wells(iCol * num_replicates, length=num_replicates),
                               new_tip='never',
                               disposal_vol=p50_disposal_vol,
                               trash=trash_control,
                               allow_blow_elision=use_blow_elision,
                               allow_carryover=use_carryover)
    done_tip(p50)

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
                     allow_blow_elision=use_blow_elision,
                     allow_carryover=use_carryover)
    done_tip(p10)
    done_tip(p50)

    # Plate strand B and mix
    # Mixing always needs the p50, but plating may need either; optimize tip usage
    log('Plating Strand B')
    plate_mix_vol = 50  # total plated volume is some 84uL; we need to use a substantial fraction of that to get good mixing
    plate_mix_count = simple_mix_count
    for iVolume in range(0, len(strand_volumes)):
        dest_wells = calculateStrandBWells(iVolume)
        volume = strand_volumes[iVolume]
        # if strand_volumes[index] == 0: continue  # don't skip: we want to mix
        if usesP10(volume, len(dest_wells), allow_zero=True):
            p = p10
        else:
            p = p50

        if volume != 0 or p is p50:
            log('Plating Strand B: volume %d with %s' % (volume, p.name))
            # We can't use distribute here as we need to avoid cross contamination from plate well to plate well
            p.transfer(volume, diluted_strand_b, dest_wells,
                       new_tip='always',
                       trash=trash_control,
                       mix_after=(plate_mix_count if p is p50 else 0, plate_mix_vol))  # always use p50 to mix

        if p is not p50:  # mix plate wells that we didn't already
            for well in dest_wells:
                simple_mix([well], 'Explicitly Mixing',  pipette=p50, volume=plate_mix_vol, count=plate_mix_count)  # new tip each well


########################################################################################################################
# Off to the races
########################################################################################################################

noteLiquid('Water', location=water)
diluteStrands()
createMasterMix()
plateEverything()
