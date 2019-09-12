"""
@author Robert Atkinson
"""

from opentrons import labware, instruments, robot, modules, types
from typing import List

metadata = {
    'protocolName': 'Pairwise Interaction: Dilute & Master',
    'author': 'Robert Atkinson <bob@theatkinsons.org>',
    'description': 'Plate a 96well plate to study the interaction of two DNA strands'
}

########################################################################################################################
# Labware
########################################################################################################################

# Configure the tips and the pipettes
tips300a = labware.load('opentrons_96_tiprack_300ul', 1)
tips300b = labware.load('opentrons_96_tiprack_300ul', 4)
tips10 = labware.load('opentrons_96_tiprack_10ul', 7)
p10 = instruments.P10_Single(mount='left', tip_racks=[tips10])
# p50 = instruments.P50_Single(mount='right', tip_racks=[tips300a, tips300b])
p300 = instruments.P300_Single(mount='right', tip_racks=[tips300a])

# Control tip usage
p10.start_at_tip(tips10['A1'])
p300.start_at_tip(tips300a['A1'])
trash_control = False  # True trashes tips; False will return trip to rack (use for debugging only)

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
dilution_source = 432
dilution_water = 768
dilution_mix = 100  # how much to suck and spew for mixing

# Parameters for master mix
mm_buffer = 1774.08
mm_evagreen = 443.52
mm_common_water = 739.2
mm_mix = 100  # how much to suck and spew for mixing


########################################################################################################################
# Utilities
########################################################################################################################


def log(msg: str):
    robot.comment(msg)


########################################################################################################################
# Off to the races
########################################################################################################################

# Create dilutions of strands
p300.transfer(dilution_water, water, [diluted_strand_a, diluted_strand_b],
              new_tip='once',  # can reuse for all diluent dispensing since dest tubes are initially empty
              trash=trash_control
              )
p300.transfer(dilution_source, idt_strand_a, diluted_strand_a, trash=trash_control, mix_before=(3, dilution_mix), mix_after=(3, dilution_mix))
p300.transfer(dilution_source, idt_strand_b, diluted_strand_b, trash=trash_control, mix_before=(3, dilution_mix), mix_after=(3, dilution_mix))


# Create master mix
p300.transfer(mm_buffer, buffer, master_mix, trash=trash_control, mix_before=(3, mm_mix))
p300.transfer(mm_evagreen, evagreen, master_mix, trash=trash_control, mix_before=(3, mm_mix))
p300.transfer(mm_common_water, water, master_mix, trash=trash_control, mix_after=(3, mm_mix))
