"""
@author Robert Atkinson
"""

from opentrons import labware, instruments, robot, modules, types
from typing import List

metadata = {
    'protocolName': 'Pairwise Interaction: Dilute & Master',
    'author': 'Robert Atkinson <bob@theatkinsons.org>',
    'description': 'Preheat the temp module'
}

# Define labware locations
temp_module = modules.load('tempdeck', 7)
temp_module.set_temperature(37)
temp_module.wait_for_temp()
