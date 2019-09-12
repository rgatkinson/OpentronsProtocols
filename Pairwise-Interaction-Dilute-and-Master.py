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
# p10 = instruments.P10_Single(mount='left', tip_racks=[tips10])
p50 = instruments.P50_Single(mount='right', tip_racks=[tips300a])
# p300 = instruments.P300_Single(mount='right', tip_racks=[tips300a])
pipette = p50

# Control tip usage
pipette.start_at_tip(tips300a['A1'])
trash_control = True  # True trashes tips; False will return trip to rack (use for debugging only)

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
mm_mix_vol = 50  # how much to suck and spew for mixing
mm_mix_count = 4


########################################################################################################################
# Utilities
########################################################################################################################


def log(msg: str):
    robot.comment(msg)


def done_tip(p):
    if trash_control:
        p.drop_tip()
    else:
        p.return_tip()


########################################################################################################################
# Off to the races
########################################################################################################################

log('Mixing Strand A')
pipette.pick_up_tip()
pipette.mix(mix_count, mix_vol, idt_strand_a)
done_tip(pipette)

log('Mixing Strand B')
pipette.pick_up_tip()
pipette.mix(mix_count, mix_vol, idt_strand_b)
done_tip(pipette)


# Create dilutions of strands
log('Moving water for diluting Strands A and B')
pipette.transfer(dilution_water_vol, water, [diluted_strand_a, diluted_strand_b],
                 new_tip='once',  # can reuse for all diluent dispensing since dest tubes are initially empty
                 trash=trash_control
                 )
log('Diluting Strand A')
pipette.transfer(dilution_source_vol, idt_strand_a, diluted_strand_a, trash=trash_control)
log('Diluting Strand B')
pipette.transfer(dilution_source_vol, idt_strand_b, diluted_strand_b, trash=trash_control)


log('Mixing Diluted Strand A')
pipette.pick_up_tip()
pipette.mix(mix_count, mix_vol, diluted_strand_a)
done_tip(pipette)

log('Mixing Diluted Strand A')
pipette.pick_up_tip()
pipette.mix(mix_count, mix_vol, diluted_strand_b)
done_tip(pipette)


# Create master mix
log('Mixing Buffer')
pipette.pick_up_tip()
pipette.mix(mix_count, mix_vol, buffer)
done_tip(pipette)

# log('Mixing Evagreen')  # Not worth it
# pipette.pick_up_tip()
# pipette.mix(mix_count, evagreen, buffer)
# done_tip(pipette)

log('Creating Master Mix: Water')
pipette.transfer(mm_common_water_vol, water, master_mix, trash=trash_control)
log('Creating Master Mix: Buffer')
pipette.transfer(mm_buffer_vol, buffer, master_mix, trash=trash_control)
log('Creating Master Mix: EvaGreen')
pipette.transfer(mm_evagreen_vol, evagreen, master_mix, trash=trash_control, new_tip='always')  # 'always' to avoid contaminating the Evagreen source

log('Mixing Master Mix')
pipette.pick_up_tip()
pipette.mix(mm_mix_count, mm_mix_vol, master_mix)
done_tip(pipette)
