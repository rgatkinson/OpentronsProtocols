"""
@author Robert Atkinson
@date August 26th, 2019
"""

from enum import Enum
from opentrons import labware, instruments, robot, modules


metadata = {
    'protocolName': 'Calibrate IDT Tubes',
    'author': 'Robert Atkinson <bob@theatkinsons.org>',
    'description': 'Does just enough to understand if our calibration of the IDT tube holder labware is correct'
    }


# Indicate where to find the diluent
diluent_well = labware.load('opentrons_24_tuberack_generic_2ml_screwcap', 2).wells('A1')

# Pipette configurations
tipracks = [labware.load('opentrons_96_tiprack_300ul', 1)]
pipette = instruments.P50_Single(mount='right', tip_racks=tipracks)

robot.pause('This is a message: %f' % 3.1415)

pipette.pick_up_tip()
pipette.aspirate(50, diluent_well)
pipette.return_tip()
