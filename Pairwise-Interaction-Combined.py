"""
@author Robert Atkinson
"""

from opentrons import labware, instruments, robot, modules, types
from opentrons.data_storage import database
from typing import List
from opentrons.legacy_api.containers import Container, Well  # Hack: not in public Opentrons API

metadata = {
    'protocolName': 'Pairwise Interaction: Dilute & Master',
    'author': 'Robert Atkinson <bob@theatkinsons.org>',
    'description': 'Study the interaction of two DNA strands'
}

########################################################################################################################
# Configurable parameters
########################################################################################################################

# Volumes of master mix ingredients
buffer_volumes = [2000]        # A1, A2, etc in screwcap rack
evagreen_volumes = [2000]      # B1, B2, etc in screwcap rack

# Control tip usage
p10_start_tip = 'A1'
p50_start_tip = 'A1'
trash_control = True

# Parameters for diluting each strand
strand_dilution_factor = 25.0 / 9.0  # per Excel worksheet
strand_dilution_vol = 1250
strand_dilution_source_vol = strand_dilution_vol / strand_dilution_factor
strand_dilution_water_vol = strand_dilution_vol - strand_dilution_source_vol
simple_mix_vol = 50  # how much to suck and spew for mixing
simple_mix_count = 4

# Parameters for master mix
master_mix_buffer_vol = 1693.44
master_mix_evagreen_vol = 423.36
master_mix_common_water_vol = 705.6

# Define the volumes of diluted strands we will use
strand_volumes = [0, 2, 5, 8, 12, 16, 20, 28]
num_replicates = 3
columns_per_plate = 12
nSamples_per_row = columns_per_plate // num_replicates
per_well_water_volumes = [
    [63, 61, 58, 55],
    [61, 59, 56, 53],
    [58, 56, 53, 50],
    [55, 53, 50, 47],
    [39, 35, 31, 23],
    [35, 31, 27, 19],
    [31, 27, 23, 15],
    [23, 19, 15,  7]]


########################################################################################################################
# Labware
########################################################################################################################

# Configure the tips and the pipettes
tips300a = labware.load('opentrons_96_tiprack_300ul', 1)
tips300b = labware.load('opentrons_96_tiprack_300ul', 4)
tips10 = labware.load('opentrons_96_tiprack_10ul', 7)
p10 = instruments.P10_Single(mount='left', tip_racks=[tips10])
p50 = instruments.P50_Single(mount='right', tip_racks=[tips300a, tips300b])

# Control tip usage
p10.start_at_tip(tips10[p10_start_tip])
p50.start_at_tip(tips300a[p50_start_tip])

# Custom disposal volumes to minimize reagent usage
p50_disposal_vol = 5
p10_disposal_vol = 1


temp_slot = 10
temp_module = modules.load('tempdeck', temp_slot)
screwcap_rack = labware.load('opentrons_24_aluminumblock_generic_2ml_screwcap', temp_slot, label='screwcap_rack', share=True)
eppendorf_1_5_rack = labware.load('opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap', 2, label='eppendorf_1_5_rack')
falcon_rack = labware.load('opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical', 5, label='falcon_rack')
plate = labware.load('biorad_96_wellplate_200ul_pcr', 3, label='plate')
trough = labware.load('usascientific_12_reservoir_22ml', 6, label='trough')

# Name specific places in the labware
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

def log(msg: str):
    robot.comment("*********** %s ***********" % msg)


def done_tip(pp):
    if trash_control:
        pp.drop_tip()
    else:
        pp.return_tip()


########################################################################################################################
# Logic
########################################################################################################################

# Into which wells should we place the n'th sample of strand A
def calculateStrandAWells(iSample: int) -> List[types.Location]:
    row_first = 0 if iSample < nSamples_per_row else nSamples_per_row
    col_first = (num_replicates * iSample) % columns_per_plate
    result = []
    for row in range(row_first, row_first + min(nSamples_per_row, len(strand_volumes))):
        for col in range(col_first, col_first + num_replicates):
            result.append(plate.rows(row).wells(col))
    return result


# Into which wells should we place the n'th sample of strand B
def calculateStrandBWells(iSample: int) -> List[types.Location]:
    if iSample < nSamples_per_row:
        col_max = num_replicates * (len(strand_volumes) if len(strand_volumes) < nSamples_per_row else nSamples_per_row)
    else:
        col_max = num_replicates * (0 if len(strand_volumes) < nSamples_per_row else len(strand_volumes)-nSamples_per_row)
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
def usesP10(queriedVol, count, allowZero):
    return (allowZero or 0 < queriedVol) and (queriedVol < p50_min_vol or queriedVol * count <= p10_max_vol)


########################################################################################################################
# Making master mix and diluting strands
########################################################################################################################

def simple_mix(well, msg, pp=p50):
    log(msg)
    pp.pick_up_tip()
    pp.mix(simple_mix_count, simple_mix_vol, well)
    done_tip(pp)

def diluteStrands():
    simple_mix(strand_a, 'Mixing Strand A')
    simple_mix(strand_b, 'Mixing Strand B')

    # Create dilutions of strands
    log('Moving water for diluting Strands A and B')
    p50.transfer(strand_dilution_water_vol, water, [diluted_strand_a, diluted_strand_b],
                 new_tip='once',  # can reuse for all diluent dispensing since dest tubes are initially empty
                 trash=trash_control
                 )
    log('Diluting Strand A')
    p50.transfer(strand_dilution_source_vol, strand_a, diluted_strand_a, trash=trash_control)
    log('Diluting Strand B')
    p50.transfer(strand_dilution_source_vol, strand_b, diluted_strand_b, trash=trash_control)

    simple_mix(diluted_strand_a, 'Mixing Diluted Strand A')
    simple_mix(diluted_strand_b, 'Mixing Diluted Strand B')

def createMasterMix():
    # Buffer was just unfrozen. Mix to ensure uniformity. EvaGreen doesn't freeze, no need to mix
    for buffer in buffers:
        simple_mix(buffer[0], "Mixing Buffer")

    # transfer from multiple source wells, each with a current defined volume
    def transferMultiple(ctx, volume, tuples, dest, new_tip):
        index = 0
        cur_loc = tuples[index][0]
        cur_volume = tuples[index][1]
        while volume > 0:
            while cur_volume < p50_min_vol * 2:
                index = index + 1
                cur_loc = tuples[index][0]
                cur_volume = tuples[index][1]
            this_volume = min(volume, cur_volume)
            log('%s: xfer %f from %s in %s to %s in %s' % (ctx, this_volume, cur_loc, cur_loc.parent, dest, dest.parent))
            p50.transfer(this_volume, cur_loc, dest, trash=trash_control, new_tip=new_tip)
            volume = volume - this_volume

    log('Creating Master Mix: Water')
    p50.transfer(master_mix_common_water_vol, water, master_mix, trash=trash_control)
    log('Creating Master Mix: Buffer')
    transferMultiple('Creating Master Mix: Buffer', master_mix_buffer_vol, buffers, master_mix, new_tip='once')
    simple_mix(master_mix, 'Mixing Master Mix')     # help eliminate air bubbles: smaller volume right now
    log('Creating Master Mix: EvaGreen')
    transferMultiple('Creating Master Mix: EvaGreen', master_mix_evagreen_vol, evagreens, master_mix, new_tip='always')  # 'always' to avoid contaminating the Evagreen source
    simple_mix(master_mix, 'Mixing Master Mix')


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
            p50.distribute(volume, water, plate.rows(iRow).wells(iCol * num_replicates, length=num_replicates),
                           new_tip='never',
                           disposal_vol=p50_disposal_vol,
                           trash=trash_control)
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
        if usesP10(volume, len(dest_wells), False):
            p = p10
            disposal_vol = p10_disposal_vol
        else:
            p = p50
            disposal_vol = p50_disposal_vol
        log('Plating Strand A: volume %d with %s' % (volume, p.name))
        p.distribute(volume, diluted_strand_a, dest_wells,
                     new_tip='never',
                     disposal_vol=disposal_vol,
                     trash=trash_control)
    done_tip(p10)
    done_tip(p50)

    # Plate strand B and mix
    # We can't use distribute here as we need to avoid cross contamination from plate well to plate well
    log('Plating Strand B')
    mix_vol = 10  # so we can use either pipette
    mix_count = simple_mix_count
    for iVolume in range(0, len(strand_volumes)):
        dest_wells = calculateStrandBWells(iVolume)
        volume = strand_volumes[iVolume]
        # if strand_volumes[index] == 0: continue  # don't skip: we want to mix
        if usesP10(volume, len(dest_wells), True):
            p = p10
            disposal_vol = p10_disposal_vol
        else:
            p = p50
            disposal_vol = p50_disposal_vol
        log('Plating Strand B: volume %d with %s' % (volume, p.name))
        p.transfer(volume, diluted_strand_b, dest_wells,
                   new_tip='always',
                   trash=trash_control,
                   disposal_vol=disposal_vol,
                   mix_after=(mix_count, mix_vol))


########################################################################################################################
# Off to the races
########################################################################################################################

diluteStrands()
createMasterMix()
plateEverything()
