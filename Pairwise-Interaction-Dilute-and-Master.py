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

# Configure the tips and the pipettes
tips300 = labware.load('opentrons_96_tiprack_300ul', 1)
p300 = instruments.P300_Single(mount='right', tip_racks=[tips300])


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
buffer = idt_rack['D6']
strand_a = eppendorf_rack['A1']
strand_b = eppendorf_rack['B1']
master_mix = falcon_rack['A1']


# Control tip usage
p300.start_at_tip(tips300['A1'])
trash_control = True  # False will return trip to rack (use for debugging only)


# Create dilutions of strands
dilution_source = 432
dilution_water = 768
dilution_mix = 100
p300.transfer(dilution_water, water, [strand_a, strand_b],
              new_tip='once',  # can reuse for all diluent dispensing since dest tubes are initially empty
              trash=trash_control
              )
p300.transfer(dilution_source, idt_strand_a, strand_a, trash=trash_control, mix_before=(2, dilution_mix), mix_after=(3, dilution_mix))
p300.transfer(dilution_source, idt_strand_b, strand_b, trash=trash_control, mix_before=(2, dilution_mix), mix_after=(3, dilution_mix))


# Create master mix : total volume 2956.8 uL
mm_buffer = 1774.08
mm_evagreen = 443.52
mm_common_water = 739.2
mm_mix = 100
p300.transfer(mm_buffer, buffer, master_mix, trash=trash_control)
p300.transfer(mm_evagreen, evagreen, master_mix, trash=trash_control)
p300.transfer(mm_common_water, water, master_mix, trash=trash_control, mix_after=(3, mm_mix))
