"""
@author Robert Atkinson
@date August 21st, 2019
@version 1.0
"""
from enum import Enum
from opentrons import labware, instruments, robot, modules


metadata = {
    'protocolName': 'Test1',
    'author': 'Robert Atkinson <bob@theatkinsons.org>'
    }


class PipetteType(Enum):
    p10_single = 1
    p50_single = 2
    p300_single = 3


def run_opentrons_logo(
        pipette_type: PipetteType,
        dye_labware_type: 'StringSelection...' = 'trough-12row'):

    if pipette_type == PipetteType.p300_single:
        tiprack = labware.load('opentrons_96_tiprack_300ul', '1')
        pipette = instruments.P300_Single(mount='right', tip_racks=[tiprack])
    elif pipette_type == PipetteType.p50_single:
        tiprack = labware.load('opentrons_96_tiprack_300ul', '1')
        pipette = instruments.P50_Single(mount='right', tip_racks=[tiprack])
    elif pipette_type == PipetteType.p10_single:
        tiprack = labware.load('opentrons_96_tiprack_10ul', '1')
        pipette = instruments.P10_Single(mount='right', tip_racks=[tiprack])

    if dye_labware_type == 'trough-12row':
        dye_container = labware.load('trough-12row', '2')
    else:
        dye_container = labware.load('tube-rack-2ml', '2')

    output = labware.load('biorad_96_wellplate_200ul_pcr', 3)

    # Well Location set-up
    dye1_wells = ['A5', 'A6', 'A8', 'A9', 'B4', 'B10', 'C3', 'C11', 'D3', 'D11', 'E3', 'E11', 'F3', 'F11', 'G4', 'G10', 'H5', 'H6', 'H7', 'H8', 'H9']
    dye2_wells = ['C7', 'D6', 'D7', 'D8', 'E5', 'E6', 'E7', 'E8', 'E9', 'F5', 'F6', 'F7', 'F8', 'F9', 'G6', 'G7', 'G8']

    dye2 = dye_container.wells('A1')
    dye1 = dye_container.wells('A2')

    pipette.distribute(
        50,
        dye1,
        output.wells(dye1_wells),
        new_tip='once')
    pipette.distribute(
        50,
        dye2,
        output.wells(dye2_wells),
        new_tip='once')



run_opentrons_logo(pipette_type=PipetteType.p300_single, dye_labware_type='trough-12row')
