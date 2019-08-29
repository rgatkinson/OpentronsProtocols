"""
@author Robert Atkinson
"""

from opentrons import labware, instruments, robot, modules, types
from typing import List

metadata = {
    'protocolName': 'Pairwise Interaction',
    'author': 'Robert Atkinson <bob@theatkinsons.org>',
    'description': 'Plate a 96well plate to study the interaction of two DNA strands'
}

# Configure the tips and the pipettes
tips10 = labware.load('opentrons_96_tiprack_10ul', 4)
tips300 = labware.load('opentrons_96_tiprack_300ul', 1)
p10 = instruments.P10_Single(mount='left', tip_racks=[tips10])
p50 = instruments.P50_Single(mount='right', tip_racks=[tips300])


# Define labware locations
idt_rack = labware.load('opentrons_24_tuberack_generic_2ml_screwcap', 5, label='idt_rack')  # IDT tubes
eppendorf_rack = labware.load('opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap', 2, label='eppendorf_rack')  # Eppendorf tubes
falcon_rack = labware.load('opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical', 8, label='falcon_rack')
plate = labware.load('biorad_96_wellplate_200ul_pcr', 3, label='plate')
trough = labware.load('usascientific_12_reservoir_22ml', 6, label='trough')


# Name specific places in the labware
water = trough['A1']
idt_strand_a = idt_rack['A1']
idt_strand_b = idt_rack['B1']
evagreen = idt_rack['A6']
strand_a = eppendorf_rack['A1']
strand_b = eppendorf_rack['B1']
buffer = eppendorf_rack['A6']
master_mix = falcon_rack['A1']


# Control tip usage
p10.start_at_tip(tips10['A1'])
p50.start_at_tip(tips300['A1'])
trash_control = True  # False will return trip to rack (use for debugging only)

# Define the volumes of diluted strands we will use
strand_volumes = [0, 2, 5, 8, 12, 16, 20, 28]


# Into which wells should we place the n'th volume of strand A
def strandAWells(n: int) -> List[types.Location]:
    row_first = 0 if n < 4 else 4
    col_first = (3 * n) % 12
    result = []
    for row in range(row_first, row_first + 4):
        for col in range(col_first, col_first + 3):
            result.append(plate.rows(row).wells(col))
    return result


# Into which wells should we place the n'th volume of strand B
def strandBWells(n: int) -> List[types.Location]:
    if n < 4:
        col_max = 3 * (len(strand_volumes) if len(strand_volumes) < 4 else 4)
    else:
        col_max = 3 * (0 if len(strand_volumes) < 4 else len(strand_volumes)-4)
    print('len=%d n=%d col_max=%d' % (len(strand_volumes), n, col_max))
    result = []
    for col in range(0, col_max):
        result.append(plate.rows(n).wells(col))
    return result


# Plate strand A
for index in range(0, len(strand_volumes)):
    if strand_volumes[index] == 0: continue
    p = p10 if strand_volumes[index] <= 10 else p50
    p.transfer(strand_volumes[index], strand_a, strandAWells(index), new_tip='once', trash=trash_control)


# Plate strand B
for index in range(0, len(strand_volumes)):
    if strand_volumes[index] == 0: continue
    p = p10 if strand_volumes[index] <= 10 else p50
    p.transfer(strand_volumes[index], strand_b, strandBWells(index), new_tip='always', trash=trash_control)
