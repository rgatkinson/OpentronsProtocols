"""
@author Robert Atkinson
"""

from opentrons import labware, instruments, robot, modules, types
from typing import List

metadata = {
    'protocolName': 'Pairwise Interaction: Dilute & Master',
    'author': 'Robert Atkinson <bob@theatkinsons.org>',
    'description': 'Study the interaction of two DNA strands'
}

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
p10.start_at_tip(tips10['A1'])
p50.start_at_tip(tips300a['A1'])
trash_control = False  # True trashes tips; False will return trip to rack (use for debugging only)

# Custom disposal volumes to minimize reagent usage
p50_disposal_vol = 2
p10_disposal_vol = 1


# Define labware locations
temp_slot = 10
temp_module = modules.load('tempdeck', temp_slot)
screwcap_rack = labware.load('opentrons_24_aluminumblock_generic_2ml_screwcap', temp_slot, label='screwcap_rack', share=True)  # IDT tubes on temp module
eppendorf_rack = labware.load('opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap', 2, label='eppendorf_rack')  # Eppendorf tubes
falcon_rack = labware.load('opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical', 5, label='falcon_rack')
plate = labware.load('biorad_96_wellplate_200ul_pcr', 3, label='plate')
trough = labware.load('usascientific_12_reservoir_22ml', 6, label='trough')

# Name specific places in the labware
water = trough['A1']
idt_strand_a = screwcap_rack['A1']
idt_strand_b = screwcap_rack['B1']
evagreen = screwcap_rack['A6']
buffer = screwcap_rack['D6']
diluted_strand_a = eppendorf_rack['A1']
diluted_strand_b = eppendorf_rack['B1']
master_mix = falcon_rack['A1']


########################################################################################################################
# Configurable parameters
########################################################################################################################

# Parameters for diluting each strand
dilution_source_vol = 468
dilution_water_vol = 832
mix_vol = 50  # how much to suck and spew for mixing
mix_count = 4

# Parameters for master mix
mm_buffer_vol = 1774.08  # NOTE: this is a LOT. Might have to allow for multiple sources.
mm_evagreen_vol = 443.52  # NOTE: this is a LOT. Might have to allow for multiple sources.
mm_common_water_vol = 739.2

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
# Utilities
########################################################################################################################


def log(msg: str):
    robot.comment("*********** %s ***********" % msg)


def done_tip(pp):
    if trash_control:
        pp.drop_tip()
    else:
        pp.return_tip()


def simple_mix(well, msg, pp=p50):
    log(msg)
    pp.pick_up_tip()
    pp.mix(mix_count, mix_vol, well)
    done_tip(pp)


########################################################################################################################
# Logic
########################################################################################################################


# Into which wells should we place the n'th sample of strand A
def strandAWells(iSample: int) -> List[types.Location]:
    row_first = 0 if iSample < nSamples_per_row else nSamples_per_row
    col_first = (num_replicates * iSample) % columns_per_plate
    result = []
    for row in range(row_first, row_first + min(nSamples_per_row, len(strand_volumes))):
        for col in range(col_first, col_first + num_replicates):
            result.append(plate.rows(row).wells(col))
    return result


# Into which wells should we place the n'th sample of strand B
def strandBWells(iSample: int) -> List[types.Location]:
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
        result.extend(strandAWells(n))
    return result


# Figuring out what pipettes should pipette what volumes
vol_p10_max = 10
vol_p50_min = 5
def usesP10(queriedVol, count, allowZero):
    return (allowZero or 0 < queriedVol) and (queriedVol < vol_p50_min or queriedVol * count <= vol_p10_max)


########################################################################################################################
# Off to the races
########################################################################################################################

simple_mix(idt_strand_a, 'Mixing Strand A')
simple_mix(idt_strand_b, 'Mixing Strand B')


# Create dilutions of strands
log('Moving water for diluting Strands A and B')
p50.transfer(dilution_water_vol, water, [diluted_strand_a, diluted_strand_b],
            new_tip='once',  # can reuse for all diluent dispensing since dest tubes are initially empty
            trash=trash_control
            )
log('Diluting Strand A')
p50.transfer(dilution_source_vol, idt_strand_a, diluted_strand_a, trash=trash_control)
log('Diluting Strand B')
p50.transfer(dilution_source_vol, idt_strand_b, diluted_strand_b, trash=trash_control)


simple_mix(diluted_strand_a, 'Mixing Diluted Strand A')
simple_mix(diluted_strand_b, 'Mixing Diluted Strand B')

# Create master mix
simple_mix(buffer, "Mixing Buffer")
# simple_mix(evagreen, "Mixing EvaGreen")  # mixing not needed, as it doesn't freeze


log('Creating Master Mix: Water')
p50.transfer(mm_common_water_vol, water, master_mix, trash=trash_control)
log('Creating Master Mix: Buffer')
p50.transfer(mm_buffer_vol, buffer, master_mix, trash=trash_control)
log('Creating Master Mix: EvaGreen')
p50.transfer(mm_evagreen_vol, evagreen, master_mix, trash=trash_control, new_tip='always')  # 'always' to avoid contaminating the Evagreen source

simple_mix(master_mix, 'Mixing Master Mix')

########################################################################################################################

# Plate master mix
log('Plating master mix ================================= ')
master_mix_per_well = 28
p50.distribute(master_mix_per_well, master_mix, usedWells(),
               new_tip='once',
               disposal_vol=p50_disposal_vol,
               trash=trash_control)


log('Plating per-well water ================================= ')
# Plate per-well water. We save tips by being happy to pollute our water trough with a bit of master mix.
p50.pick_up_tip()
for iRow in range(len(per_well_water_volumes)):
    for iCol in range(len(per_well_water_volumes[iRow])):
        vol = per_well_water_volumes[iRow][iCol]
        p50.distribute(vol, water, plate.rows(iRow).wells(iCol*num_replicates, length=num_replicates),
                       new_tip='never',
                       disposal_vol=p50_disposal_vol,
                       trash=trash_control)
done_tip(p50)


# Plate strand A
# All plate wells at this point only have water and master mix, so we can't get cross-plate-well
# contamination. We only need to worry about contaminating the Strand A source, which we accomplish
# by using new_tip='always'.
log('Plating Strand A (counts=%s) ================================= ' % (list(map(lambda nSample: len(strandAWells(nSample)), range(len(strand_volumes))))))
for iVolume in range(0, len(strand_volumes)):
    dest_wells = strandAWells(iVolume)
    vol = strand_volumes[iVolume]
    if vol == 0: continue
    if usesP10(vol, len(dest_wells), False):
        p = p10
        disposal_vol = p10_disposal_vol
    else:
        p = p50
        disposal_vol = p50_disposal_vol
    log('Plating Strand A: volume %d with %s' % (vol, p.name))
    p.distribute(vol, diluted_strand_a, dest_wells,
                 new_tip='always',
                 disposal_vol=disposal_vol,
                 trash=trash_control)


# Plate strand B and mix
# We can't use distribute here as we need to avoid cross contamination from plate well to plate well
log('Plating Strand B (counts=%s) ================================= ' % (list(map(lambda nSample: len(strandBWells(nSample)), range(len(strand_volumes))))))
mix_vol = 10
mix_count = 4
for iVolume in range(0, len(strand_volumes)):
    dest_wells = strandBWells(iVolume)
    vol = strand_volumes[iVolume]
    # if strand_volumes[index] == 0: continue  # don't skip: we want to mix
    if usesP10(vol, len(dest_wells), True):
        p = p10
        disposal_vol = p10_disposal_vol
    else:
        p = p50
        disposal_vol = p50_disposal_vol
    log('Plating Strand B: volume %d with %s' % (vol, p.name))
    p.transfer(vol, diluted_strand_b, dest_wells,
               new_tip='always',
               trash=trash_control,
               disposal_vol=disposal_vol,
               mix_after=(mix_count, mix_vol))
