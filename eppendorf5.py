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

pip_start_tip = 'A1'

########################################################################################################################
# Labware
########################################################################################################################

tips300a = labware_manager.load('opentrons_96_tiprack_300ul', slot=4, label='tips300a')
rack1 = labware_manager.load('Atkinson_15_tuberack_5ml_eppendorf', slot=8, label='rack1')
trough = labware_manager.load('usascientific_12_reservoir_22ml', slot=6, label='trough')

pip = instruments_manager.P300_Single_GEN2(mount='right', tip_racks=[tips300a])
pip.start_at_tip(tips300a[pip_start_tip])

waters = [trough['A1'], trough['A2'], trough['A3']]
vol_use_per_water_well = 15000
water_initially_at_least = 16000
for well in waters:
    note_liquid(location=well, name='Water', initially_at_least=water_initially_at_least)

wells = [
    rack1['A1'], rack1['A2'], rack1['A3'], rack1['A4'], rack1['A5'],
    rack1['C1'], rack1['C2'], rack1['C3'], rack1['C4'], rack1['C5']
    ]

volumes = [
    1700, 1800, 1900, 2000, 2500,
    3000, 3500, 4000, 4500, 5000,
    ]

########################################################################################################################
# Off to the races
########################################################################################################################

water_well = 0
water_volume = vol_use_per_water_well

for well, vol in zip(wells, volumes):
    if water_volume < vol:
        water_well += 1
        water_volume = vol_use_per_water_well
    pip.transfer(vol, waters[water_well], well, new_tip='once', full_dispense=True, trash=config.trash_control)
    water_volume -= vol
