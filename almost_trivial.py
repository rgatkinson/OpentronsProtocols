"""
@author Robert Atkinson
"""

metadata = {
    'protocolName': 'Almost Trivial',
    'author': 'Robert Atkinson <bob@theatkinsons.org>',
    'description': 'An almost trivial protocol that now fails to load'
}

from opentrons import labware, instruments

tips = labware.load('opentrons_96_tiprack_300ul', 1)
p = instruments.P50_Single(mount='right', tip_racks=[tips])

trough = labware.load('usascientific_12_reservoir_22ml', 6, label='trough')
water = trough['A1']
p.transfer(50, water, water)
