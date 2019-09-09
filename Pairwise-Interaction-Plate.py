"""
@author Robert Atkinson
"""

from opentrons import labware, instruments, robot, modules, types
from typing import List
from itertools import tee

metadata = {
    'protocolName': 'Pairwise Interaction',
    'author': 'Robert Atkinson <bob@theatkinsons.org>',
    'description': 'Plate a 96well plate to study the interaction of two DNA strands'
}

########################################################################################################################
# Labware
########################################################################################################################

# Configure the tips and the pipettes
tips10 = labware.load('opentrons_96_tiprack_10ul', 4)
tips300 = labware.load('opentrons_96_tiprack_300ul', 1)
p10 = instruments.P10_Single(mount='left', tip_racks=[tips10])
p50 = instruments.P50_Single(mount='right', tip_racks=[tips300])

# Control tip usage
p10.start_at_tip(tips10['A1'])
p50.start_at_tip(tips300['A1'])
trash_control = False  # True trashes tips; False will return trip to rack (use for debugging only)


# Define labware locations
temp_module = modules.load('tempdeck', 7)
screwcap_rack = labware.load('opentrons_24_aluminumblock_generic_2ml_screwcap', 7, label='screwcap_rack', share=True)  # IDT tubes on temp module
eppendorf_rack = labware.load('opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap', 2, label='eppendorf_rack')  # Eppendorf tubes
falcon_rack = labware.load('opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical', 8, label='falcon_rack')
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

# Define the volumes of diluted strands we will use
strand_volumes = [0, 2, 5, 8, 12, 16, 20, 28]
num_replicates = 3
columns_per_plate = 12
nSamples_per_row = columns_per_plate // num_replicates
per_well_water_volumes = [
    [63, 61, 48, 55],
    [51, 59, 56, 53],
    [58, 56, 53, 50],
    [55, 53, 50, 47],
    [39, 35, 31, 23],
    [35, 31, 27, 19],
    [31, 27, 23, 15],
    [23, 19, 15, 7]]


########################################################################################################################
# Utilities
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
volume_threshold = 10
def usesP10(vol, allowZero=False):
    return (allowZero or 0 < vol) and vol <= volume_threshold
def usesP50(vol, allowZero=False):
    return (allowZero or 0 < vol) and not usesP10(vol, allowZero)
def p10Volumes(allowZero=False):
    return list(filter(lambda vol: usesP10(vol, allowZero), strand_volumes))
def p50Volumes(allowZero=False):
    return list(filter(lambda vol: usesP50(vol, allowZero), strand_volumes))


# Hack-o-rama: answer whether the indicated pipette has any more tips or not
def hasMoreTips(p, tips_needed: int) -> bool:
    if tips_needed <= 0: return True
    # Remember the current tip and starting tip
    prev_current_tip = p.current_tip()
    prev_starting_tip = p.starting_tip
    try:
        # Try to pull 'tips_needed' tips. An exception will throw if there's not enough
        nxt = next(p.tip_rack_iter)      # pull the first tip and remember same
        for i in range(1, tips_needed):  # pull remaining tips
            next(p.tip_rack_iter)
        # Ok, there were enough! Restore to state before the call
        p.start_at_tip(nxt)  # will call reset_tip_tracking in addition to setting starting tip
        result = True
        log('%s has at least %d more tips' % (p.name, tips_needed))
    except StopIteration:
        # Subsequent calls to next(p.tip_rack_iter) should *also* throw
        log("%s doesn't have %d tips left" % (p.name, tips_needed))
        result = False
    p.starting_tip = prev_starting_tip  # we might have trashed above
    p.current_tip(prev_current_tip)  # was trashed by reset_tip_tracking() if that got called
    return result


# see if pipette tips need to be refilled
def checkForTips(p, tips_needed: int):
    if not hasMoreTips(p, tips_needed):
        robot.pause('Please refill the tips for %s' % p.name)
        p.starting_tip = None
        p.reset_tip_tracking()


def log(msg: str):
    robot.comment(msg)


########################################################################################################################
# Off to the races
########################################################################################################################


# Plate master mix
log('Plating master mix ================================= ')
master_mix_per_well = 28
p50.distribute(master_mix_per_well, master_mix, usedWells(), new_tip='once', trash=trash_control)


log('Plating per-well water ================================= ')
# Plate per-well water. We save tips by being happy to pollute our water trough with a bit of master mix.
p50.pick_up_tip()
for iRow in range(len(per_well_water_volumes)):
    for iCol in range(len(per_well_water_volumes[iRow])):
        vol = per_well_water_volumes[iRow][iCol]
        p50.distribute(vol, water, plate.rows(iRow).wells(iCol*num_replicates, length=num_replicates), new_tip='never')
if trash_control:
    p50.drop_tip()
else:
    p50.return_tip()


# Plate strand A
log('Plating Strand A (counts=%s) ================================= ' % (list(map(lambda nSample: len(strandAWells(nSample)), range(len(strand_volumes))))))
checkForTips(p10, len(p10Volumes()) * len(list(strandAWells(0))))
checkForTips(p50, len(p50Volumes()) * len(list(strandAWells(0))))
for iVolume in range(0, len(strand_volumes)):
    if strand_volumes[iVolume] == 0: continue
    p = p10 if usesP10(strand_volumes[iVolume]) else p50
    p.transfer(strand_volumes[iVolume], diluted_strand_a, strandAWells(iVolume), new_tip='always', trash=trash_control)


# Plate strand B and mix
log('Plating Strand B (counts=%s) ================================= ' % (list(map(lambda nSample: len(strandBWells(nSample)), range(len(strand_volumes))))))
checkForTips(p10, len(p10Volumes(True)) * len(list(strandBWells(0))))
checkForTips(p50, len(p50Volumes(True)) * len(list(strandBWells(0))))
mix_vol = 10
for index in range(0, len(strand_volumes)):
    # if strand_volumes[index] == 0: continue  # don't skip: we want to mix
    p = p10 if usesP10(strand_volumes[index], True) else p50
    p.transfer(strand_volumes[index], diluted_strand_b, strandBWells(index), new_tip='always', trash=trash_control, mix_after=(4, mix_vol))
