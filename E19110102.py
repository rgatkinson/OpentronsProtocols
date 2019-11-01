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
from rgatkinson.custom_labware import load_tiprack, Opentrons15Rack
from rgatkinson.liquid import note_liquid
from rgatkinson.logging import log, info
from rgatkinson.pipette import EnhancedPipette
from rgatkinson.well import Eppendorf1point5mlTubeGeometry, Biorad96WellPlateWellGeometry, Eppendorf5point0mlTubeGeometry

########################################################################################################################
# Configurable protocol parameters
########################################################################################################################

# Tip usage
p10_start_tip = 'C5'
p50_start_tip = 'E3'
config.trash_control = True

stock_volume = 630

########################################################################################################################
# Labware
########################################################################################################################

# Configure the tips
tips10 = load_tiprack('opentrons_96_tiprack_10ul', 1, label='tips10')
tips300a = load_tiprack('opentrons_96_tiprack_300ul', 4, label='tips300a')
tips300b = load_tiprack('opentrons_96_tiprack_300ul', 7, label='tips300b')

# Configure the pipettes.
p10 = EnhancedPipette(instruments.P10_Single(mount='left', tip_racks=[tips10]), config)
p50 = EnhancedPipette(instruments.P50_Single(mount='right', tip_racks=[tips300a, tips300b]), config)

# Control tip usage
p10.start_at_tip(tips10[p10_start_tip])
p50.start_at_tip(tips300a[p50_start_tip])

# All the labware containers
eppendorf_1_5_rack = labware.load('opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap', 2, label='eppendorf_1_5_rack')
eppendorf_5_0_rack_definition = Opentrons15Rack(config, name='Atkinson 15 Tube Rack 5000 µL', default_well_geometry=Eppendorf5point0mlTubeGeometry)
eppendorf_5_0_rack = eppendorf_5_0_rack_definition.load(slot=5, label='eppendorf_5_0_rack')
plate = labware.load('biorad_96_wellplate_200ul_pcr', 6, label='plateA')
trough = labware.load('usascientific_12_reservoir_22ml', 8, label='trough')

# Name specific places in the labware containers
water = trough['A1']
initial_stock = eppendorf_1_5_rack['A1']
dilutions = eppendorf_5_0_rack.rows['A']['1':2]

# Define geometries
config.set_well_geometry(initial_stock, Eppendorf1point5mlTubeGeometry)
for tube in dilutions:
    config.set_well_geometry(tube, Eppendorf1point5mlTubeGeometry)
for well in plate.wells():
    config.set_well_geometry(well, Biorad96WellPlateWellGeometry)

# Remember initial liquid names and volumes
log('Liquid Names')
note_liquid(location=water, name='Water', min_volume=7000)  # volume is rough guess
note_liquid(location=initial_stock, name='AlluraRed', concentration="20.1442 mM", initial_volume=stock_volume)

# Clean up namespace
del well

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
        p50.transfer(dilution_source_volume, source, dilution, new_tip='once', trash=config.trash_control, keep_last_tip=True)  # keep tip cause we can use it for mixing

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

make_dilutions()
log('Pausing to mix dilutions')
robot.pause('Press Return to Continue')
plate_dilutions()
