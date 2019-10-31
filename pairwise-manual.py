"""
@author Robert Atkinson
"""

metadata = {
    'protocolName': 'Pairwise Interaction: Dilute & Master & Plate - Manual Assist',
    'author': 'Robert Atkinson <bob@theatkinsons.org>',
    'description': 'Study the interaction of two DNA strands'
}

from typing import List
from opentrons import instruments, labware, modules, robot, types

from rgatkinson import *
from rgatkinson.custom_labware import Opentrons15Rack, load_tiprack
from rgatkinson.liquid import note_liquid
from rgatkinson.logging import log, fatal
from rgatkinson.pipette import verify_well_locations

########################################################################################################################
# Tweakable protocol parameters
########################################################################################################################

use_eppendorf_for_master_mix = True

# Volumes of master mix ingredients. These are minimums in each tube.
buffer_volumes = [2000, 2000]  # Fresh tubes of B9022S
evagreen_volumes = [1000]      # Fresh tube of EvaGreen

strand_a_conc = '8.820 uM'  # Note: we'll use more Strand A than Strand B because of disposal_volumes
strand_b_conc = '9.117 uM'
strand_a_min_vol = 1100  # records said we had 1200
strand_b_min_vol = 1100  # ditto

# Tip usage
p10_start_tip = 'A1'
p50_start_tip = 'A1'
config.trash_control = True


########################################################################################################################
## Protocol
########################################################################################################################

# Diluting each strand
strand_dilution_factor = 25.0 / 9.0  # per Excel worksheet
strand_dilution_vol = 1225

# Master mix, values per Excel worksheet
mm_overhead_factor = 1.0375
master_mix_buffer_vol = 1612.8 * mm_overhead_factor
master_mix_evagreen_vol = 403.2 * mm_overhead_factor
master_mix_common_water_vol = 672 * mm_overhead_factor
master_mix_vol = master_mix_buffer_vol + master_mix_evagreen_vol + master_mix_common_water_vol

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

# compute derived constants
strand_dilution_source_vol = strand_dilution_vol / strand_dilution_factor
strand_dilution_water_vol = strand_dilution_vol - strand_dilution_source_vol

########################################################################################################################
# Labware
########################################################################################################################

# Configure the tips
tips300a = load_tiprack('opentrons_96_tiprack_300ul', 1, label='tips300a')
tips10 = load_tiprack('opentrons_96_tiprack_10ul', 4, label='tips10')
tips300b = load_tiprack('opentrons_96_tiprack_300ul', 7, label='tips300b')

# Configure the pipettes.
p10 = EnhancedPipette(instruments.P10_Single(mount='left', tip_racks=[tips10]), config=config)
p50 = EnhancedPipette(instruments.P50_Single(mount='right', tip_racks=[tips300a, tips300b]), config=config)

# Blow out faster than default in an attempt to avoid hanging droplets on the pipettes after blowout
p10.set_flow_rate(blow_out=p10.get_flow_rates()['blow_out'] * config.blow_out_rate_factor)
p50.set_flow_rate(blow_out=p50.get_flow_rates()['blow_out'] * config.blow_out_rate_factor)

# Control tip usage
p10.start_at_tip(tips10[p10_start_tip])
p50.start_at_tip(tips300a[p50_start_tip])

# All the labware containers

temp_slot = 11
temp_module = modules.load('tempdeck', temp_slot)
screwcap_rack = labware.load('opentrons_24_aluminumblock_generic_2ml_screwcap', temp_slot, label='screwcap_rack', share=True)
eppendorf_1_5_rack = labware.load('opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap', 5, label='eppendorf_1_5_rack')
plate = labware.load('biorad_96_wellplate_200ul_pcr', 6, label='plate')
trough = labware.load('usascientific_12_reservoir_22ml', 9, label='trough')

# Name specific places in the labware containers
buffers = list(zip(screwcap_rack.rows(0), buffer_volumes))
evagreens = list(zip(screwcap_rack.rows(1), evagreen_volumes))
strand_a = eppendorf_1_5_rack['A1']
strand_b = eppendorf_1_5_rack['B1']
diluted_strand_a = eppendorf_1_5_rack['A6']
diluted_strand_b = eppendorf_1_5_rack['B6']

# Define geometries
for well, __ in buffers:
    config.set_well_geometry(well, IdtTubeWellGeometry)
for well, __ in evagreens:
    config.set_well_geometry(well, IdtTubeWellGeometry)
config.set_well_geometry(strand_a, Eppendorf1point5mlTubeGeometry)
config.set_well_geometry(strand_b, Eppendorf1point5mlTubeGeometry)
config.set_well_geometry(diluted_strand_a, Eppendorf1point5mlTubeGeometry)
config.set_well_geometry(diluted_strand_b, Eppendorf1point5mlTubeGeometry)
for well in plate.wells():
    config.set_well_geometry(well, Biorad96WellPlateWellGeometry)

if use_eppendorf_for_master_mix:
    master_mix_rack_definition = Opentrons15Rack(config, name='Atkinson 15 Tube Rack 5000 µL', default_well_geometry=Eppendorf5point0mlTubeGeometry)
    master_mix_rack = master_mix_rack_definition.load(slot=8, label='master_mix_rack')
    master_mix = master_mix_rack['A1']
    water = master_mix_rack['C5']
else:
    master_mix_rack = labware.load('opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical', 8, label='master_mix_rack')
    master_mix = master_mix_rack['A1']
    config.set_well_geometry(master_mix, FalconTube15mlGeometry)
    water = trough['A1']


# Remember initial liquid names and volumes
log('Liquid Names')
note_liquid(location=water, name='Water', initial_volume=3000)
assert strand_a_min_vol >= strand_dilution_source_vol + config.well_geometry(strand_a).min_aspiratable_volume
assert strand_b_min_vol >= strand_dilution_source_vol + config.well_geometry(strand_b).min_aspiratable_volume
note_liquid(location=strand_a, name='StrandA', concentration=strand_a_conc, min_volume=strand_a_min_vol)  # i.e.: we have enough, just not specified how much
note_liquid(location=strand_b, name='StrandB', concentration=strand_b_conc, min_volume=strand_b_min_vol)  # ditto
note_liquid(location=diluted_strand_a, name='Diluted StrandA', initial_volume=strand_dilution_vol)
note_liquid(location=diluted_strand_b, name='Diluted StrandB', initial_volume=strand_dilution_vol)
note_liquid(location=master_mix, name='Master Mix', initial_volume=master_mix_vol)
for buffer in buffers:
    note_liquid(location=buffer[0], name='Buffer', initial_volume=buffer[1], concentration='5x')
for evagreen in evagreens:
    note_liquid(location=evagreen[0], name='Evagreen', initial_volume=evagreen[1], concentration='20x')

# Clean up namespace
del well

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

def diluteStrands():
    p50.layered_mix([strand_a])
    p50.layered_mix([strand_b])

    # Create dilutions of strands
    log('Moving water for diluting Strands A and B')
    p50.transfer(strand_dilution_water_vol, water, [diluted_strand_a, diluted_strand_b],
                 new_tip='once',  # can reuse for all diluent dispensing since dest tubes are initially empty
                 trash=config.trash_control
                 )
    log('Diluting Strand A')
    p50.transfer(strand_dilution_source_vol, strand_a, diluted_strand_a, trash=config.trash_control, keep_last_tip=True)
    p50.layered_mix([diluted_strand_a])

    log('Diluting Strand B')
    p50.transfer(strand_dilution_source_vol, strand_b, diluted_strand_b, trash=config.trash_control, keep_last_tip=True)
    p50.layered_mix([diluted_strand_b])


def createMasterMix():
    # Buffer was just unfrozen. Mix to ensure uniformity. EvaGreen doesn't freeze, no need to mix
    p50.layered_mix([buffer for buffer, __ in buffers], incr=2)

    # transfer from multiple source wells, each with a current defined volume
    def transfer_multiple(msg, xfer_vol_remaining, tubes, dest, new_tip, *args, **kwargs):
        tube_index = 0
        cur_well = None
        cur_vol = 0
        min_vol = 0
        while xfer_vol_remaining > 0:
            if xfer_vol_remaining < p50_min_vol:
                warn("remaining transfer volume of %f too small; ignored" % xfer_vol_remaining)
                return
            # advance to next tube if there's not enough in this tube
            while cur_well is None or cur_vol <= min_vol:
                if tube_index >= len(tubes):
                    fatal('%s: more reagent needed' % msg)
                cur_well = tubes[tube_index][0]
                cur_vol = tubes[tube_index][1]
                min_vol = max(p50_min_vol,
                              cur_vol / config.min_aspirate_factor_hack,  # tolerance is proportional to specification of volume. can probably make better guess
                              config.well_geometry(cur_well).min_aspiratable_volume)
                tube_index = tube_index + 1
            this_vol = min(xfer_vol_remaining, cur_vol - min_vol)
            assert this_vol >= p50_min_vol  # TODO: is this always the case?
            log('%s: xfer %f from %s in %s to %s in %s' % (msg, this_vol, cur_well, cur_well.parent, dest, dest.parent))
            p50.transfer(this_vol, cur_well, dest, trash=config.trash_control, new_tip=new_tip, **kwargs)
            xfer_vol_remaining -= this_vol
            cur_vol -= this_vol

    def mix_master_mix():
        log('Mixing Master Mix')
        p50.layered_mix([master_mix], incr=2, initial_turnover=master_mix_evagreen_vol * 1.2, max_tip_cycles=config.layered_mix.max_tip_cycles_large)

    log('Creating Master Mix: Water')
    p50.transfer(master_mix_common_water_vol, water, master_mix, trash=config.trash_control)

    log('Creating Master Mix: Buffer')
    transfer_multiple('Creating Master Mix: Buffer', master_mix_buffer_vol, buffers, master_mix, new_tip='once', keep_last_tip=True)  # 'once' because we've only got water & buffer in context
    p50.done_tip()  # EvaGreen needs a new tip

    log('Creating Master Mix: EvaGreen')
    transfer_multiple('Creating Master Mix: EvaGreen', master_mix_evagreen_vol, evagreens, master_mix, new_tip='always', keep_last_tip=True)  # 'always' to avoid contaminating the Evagreen source w/ buffer

    mix_master_mix()


########################################################################################################################
# Plating
########################################################################################################################

def plateMasterMix():
    log('Plating Master Mix')
    master_mix_per_well = 28
    p50.transfer(master_mix_per_well, master_mix, usedWells(),
                 new_tip='once',
                 trash=config.trash_control,
                 full_dispense=True)

def platePerWellWater():
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

    p50.transfer(water_volumes, water, plate.wells(),
                 new_tip='once',
                 trash=config.trash_control,
                 full_dispense=True)

def plateStrandA():
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
        else:
            p = p50
        log('Plating Strand A: volume %d with %s' % (volume, p.name))
        volumes = [volume] * len(dest_wells)
        p.transfer(volumes, diluted_strand_a, dest_wells,
                   new_tip='never',
                   trash=config.trash_control,
                   full_dispense=True)
    p10.done_tip()
    p50.done_tip()

def mix_plate_well(well, keep_last_tip=False):
    p50.layered_mix([well], incr=0.75, keep_last_tip=keep_last_tip)

def plateStrandBAndMix():
    # Plate strand B and mix
    # Mixing always needs the p50, but plating may need either; optimize tip usage
    log('Plating Strand B')
    mixed_wells = set()
    for iVolume in range(0, len(strand_volumes)):
        dest_wells = calculateStrandBWells(iVolume)
        volume = strand_volumes[iVolume]
        if usesP10(volume, len(dest_wells), allow_zero=True):
            p = p10
        else:
            p = p50

        # We can't use distribute here as we need to avoid cross contamination from plate well to plate well
        for well in dest_wells:
            if volume != 0:
                log("Plating Strand B: well='%s' vol=%d pipette=%s" % (well.get_name(), volume, p.name))
                p.pick_up_tip()
                p.transfer(volume, diluted_strand_b, well, new_tip='never', full_dispense=True)
            if p is p50:
                mix_plate_well(well, keep_last_tip=True)
                mixed_wells.add(well)
            p.done_tip()

    for well in plate.wells():
        if well not in mixed_wells:
            mix_plate_well(well)

def plateEverythingAndMix():
    plateMasterMix()
    platePerWellWater()
    plateStrandA()
    plateStrandBAndMix()


########################################################################################################################
# Off to the races
########################################################################################################################

wells_to_verify = [master_mix, strand_a, strand_b, diluted_strand_a, diluted_strand_b, plate.wells('A1'), plate.wells('A12'), plate.wells('H1'), plate.wells('H12')]
# verify_well_locations(wells_to_verify, p50)
# verify_well_locations(wells_to_verify, p10)

# diluteStrands()
# createMasterMix()
robot.pause('***** Ensure master mix and strand dilutions are present *****')
plateEverythingAndMix()
