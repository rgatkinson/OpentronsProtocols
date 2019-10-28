"""
@author Robert Atkinson
"""

from opentrons.commands.commands import stringify_location

metadata = {
    'protocolName': 'Calibrate Deck',
    'author': 'Robert Atkinson <bob@theatkinsons.org>',
    'description': 'Loads and retains a p50 tip to facilitate deck calibration'
}

from atkinson.opentrons import *

########################################################################################################################
# Configurable protocol parameters
########################################################################################################################

start_tip = 'A1'

########################################################################################################################
# Labware
########################################################################################################################

# Configure the pipette
tips = labware.load('opentrons_96_tiprack_300ul', 1)
p = instruments.P50_Single(mount='right', tip_racks=[tips])
p.start_at_tip(start_tip)

########################################################################################################################
# Off to the races
########################################################################################################################

p.pick_up_tip()
