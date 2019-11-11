"""
@author Robert Atkinson
"""

metadata = {
    'protocolName': 'E19110102',
    'author': 'Robert Atkinson <bob@theatkinsons.org>'
    }

import math
from opentrons import instruments, labware, modules, robot, types

from rgatkinson.configuration import config
from rgatkinson.custom_labware import labware_manager
from rgatkinson.liquid import note_liquid
from rgatkinson.logging import log, info
from rgatkinson.pipette import verify_well_locations, instruments_manager

########################################################################################################################
# Configurable protocol parameters
########################################################################################################################

# Tip usage
p10_start_tip = 'D5'
p50_start_tip = 'G3'
config.trash_control = True

stock_volume = 630
water_volume = 5000

########################################################################################################################
# Labware
########################################################################################################################

# Configure the tips
tips10 = labware_manager.load('opentrons_96_tiprack_10ul', 1, label='tips10')
tips300a = labware_manager.load('opentrons_96_tiprack_300ul', 4, label='tips300a')
tips300b = labware_manager.load('opentrons_96_tiprack_300ul', 7, label='tips300b')

# Configure the pipettes.
p10 = instruments_manager.P10_Single(mount='left', tip_racks=[tips10])
p50 = instruments_manager.P50_Single(mount='right', tip_racks=[tips300a, tips300b])

# Control tip usage
p10.start_at_tip(tips10[p10_start_tip])
p50.start_at_tip(tips300a[p50_start_tip])

# All the labware containers
eppendorf_1_5_rack = labware_manager.load('opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap', slot=2, label='eppendorf_1_5_rack')
eppendorf_5_0_rack = labware_manager.load('Atkinson_15_tuberack_5ml_eppendorf', slot=5, label='eppendorf_5_0_rack')
plate = labware_manager.load('biorad_96_wellplate_200ul_pcr', slot=6, label='plateA')

# Name specific places in the labware containers
water = eppendorf_5_0_rack['C5']
initial_stock = eppendorf_1_5_rack['A1']
dilutions = eppendorf_5_0_rack.rows['A']['1':2]

# Remember initial liquid names and volumes
log('Liquid Names')
note_liquid(location=water, name='Water', initial_volume=water_volume)
note_liquid(location=initial_stock, name='AlluraRed', concentration="20.1442 mM", initial_volume=stock_volume)

########################################################################################################################
# Dilutions
########################################################################################################################

def make_dilution(water, source, dilution, dilution_volume, dilution_factor, manual):
    if manual:
        name = f'Dilution of {source.get_name()} by {dilution_factor}x'
        note_liquid(dilution, name, initial_volume=dilution_volume)
    log(f'diluting from {source.get_name()} to {dilution.get_name()}')
    dilution_source_volume = dilution_volume / dilution_factor
    dilution_water_volume = dilution_volume - dilution_source_volume
    if manual:
        info(f'water vol={dilution_water_volume}')
        info(f'source vol={dilution_source_volume}')
    else:
        p50.transfer(dilution_water_volume, water, dilution)
        p50.transfer(dilution_source_volume, source, dilution, new_tip='once', trash=config.trash_control, keep_last_tip=False)

def make_dilutions():
    # TODO: we might be better off mixing these by hand
    make_dilution(water, initial_stock, dilutions[0], 3500, 125, manual=True)
    make_dilution(water, dilutions[0], dilutions[1], 3000, 625/125, manual=False)

def plate_dilution(source, dx):
    # We have two dilutions, three replicates each, so get 16 volumes per
    volumes = [1, 2, 3, 4, 5, 10, 15, 20, 25, 35, 50, 75, 100, 125, 150, 200]
    num_reps = 3
    for i, vol in enumerate(volumes):
        log(f'plating source={source.get_name()} vol={vol}')
        row = i % 8
        col_first = int(i / 8)
        p = p10 if vol <= 10 else p50
        for j in range(num_reps):
            col = col_first * num_reps + j + dx
            dest = plate.rows(row).wells(col)
            # info(f'dest={row},{col}')
            p.transfer(vol, source, dest, trash=config.trash_control)

def plate_dilutions():
    log('Plating')
    plate_dilution(dilutions[0], 0)
    plate_dilution(dilutions[1], 6)

########################################################################################################################
# Off to the races
########################################################################################################################

wells_to_verify = [dilutions[0], dilutions[1], water, plate.wells('A1'), plate.wells('A12'), plate.wells('H1'), plate.wells('H12')]
verify_well_locations(wells_to_verify, p50)

make_dilutions()
log('Pausing to mix dilutions')
robot.pause('Press Return to Continue')
plate_dilutions()
