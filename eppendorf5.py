"""
@author Robert Atkinson
"""

metadata = {
    'protocolName': 'Calibrate Eppendorf 5ml Tubes',
    'author': 'Robert Atkinson <bob@theatkinsons.org>'
}

from typing import List
from opentrons import instruments, labware, modules, robot, types

from rgatkinson import *
from rgatkinson.custom_labware import labware_manager
from rgatkinson.liquid import note_liquid
from rgatkinson.logging import log, fatal, user_prompt
from rgatkinson.pipette import verify_well_locations, instruments_manager

########################################################################################################################
# Parameters we tend to adjust for different runs depending on exact ingredient mix
########################################################################################################################

p50_start_tip = 'A1'

########################################################################################################################
# Labware
########################################################################################################################

tips300a = labware_manager.load('opentrons_96_tiprack_300ul', slot=4, label='tips300a')
rack1 = labware_manager.load('Atkinson_15_tuberack_5ml_eppendorf', slot=8, label='rack1')
rack2 = labware_manager.load('Atkinson_10_tuberack_6x5ml_eppendorf_4x50ml_falcon', slot=2, label='rack2')
trough = labware_manager.load('usascientific_12_reservoir_22ml', slot=6, label='trough')

p50 = instruments_manager.P50_Single(mount='right', tip_racks=[tips300a])
p50.start_at_tip(tips300a[p50_start_tip])

waters = [trough['A1'], trough['A2'], trough['A3']]
water_initial_volume = 7000
for well in waters:
    note_liquid(location=well, name='Water', initially=water_initial_volume)

wells = [
    rack1['A1'], rack1['A2'], rack1['A3'], rack1['A4'], rack1['A5'],
    rack1['C1'], rack1['C2'], rack1['C3'], rack1['C4'], rack1['C5'],
    rack2['A1'], rack2['A2'], rack2['B1'], rack2['B2'], rack2['C1'], rack2['C2']
    ]

########################################################################################################################
# Off to the races
########################################################################################################################

water_well = 0
water_volume = water_initial_volume
for i, vol in enumerate(range(100, (len(wells)+1)*100, 100)):
    if water_volume < vol:
        water_well += 1
        water_volume = water_initial_volume
    p50.transfer(vol, waters[water_well], wells[i], new_tip='once', full_dispense=True, trash=config.trash_control)
    water_volume -= vol
