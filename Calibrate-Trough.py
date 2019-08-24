"""
@author Robert Atkinson
@date August 23rd, 2019
"""

from enum import Enum
from opentrons import labware, instruments, robot, modules


metadata = {
    'protocolName': 'Calibrate Trough',
    'author': 'Robert Atkinson <bob@theatkinsons.org>',
    'description': 'Does just enough to understand if our calibration of the trough labware is correct'
    }


# Indicate where to find the diluent
diluent_well = labware.load('usascientific_12_reservoir_22ml', 2).wells('A1')

# Pipette configurations
tipracks = [labware.load('opentrons_96_tiprack_300ul', 1)]
pipette = instruments.P300_Single(mount='right', tip_racks=tipracks)

pipette.pick_up_tip()
pipette.aspirate(50, diluent_well)
pipette.return_tip()
