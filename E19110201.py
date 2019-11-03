"""
@author Robert Atkinson
"""

metadata = {
    'protocolName': 'E19110201',
    'author': 'Robert Atkinson <bob@theatkinsons.org>'
    }

from opentrons import modules, robot, types

from rgatkinson.configuration import config
from rgatkinson.custom_labware import labware_manager
from rgatkinson.liquid import note_liquid
from rgatkinson.logging import log, info, user_prompt
from rgatkinson.pipette import verify_well_locations, instruments_manager

########################################################################################################################
# Configurable protocol parameters
########################################################################################################################

# Tip usage
p10_start_tip = 'D5'
p50_start_tip = 'G3'
config.trash_control = True

use_manual_mix = True
stock_volume = 630  # only used if we're not manually mixing
waterA_initial_volume = 5000
waterB_initial_volume = 5000

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
eppendorf_5_0_rack = labware_manager.load('Atkinson 15 Tube Rack 5000 µL', slot=5, label='eppendorf_5_0_rack')
plate = labware_manager.load('biorad_96_wellplate_200ul_pcr', slot=6, label='plateA')
trough = labware_manager.load('usascientific_12_reservoir_22ml', slot=9, label='trough')

# Name specific places in the labware containers
waterA = eppendorf_5_0_rack['C4']
waterB = eppendorf_5_0_rack['C5']
initial_stock = eppendorf_1_5_rack['A1']
dilutions = eppendorf_5_0_rack.rows['A']['1':2]

# Remember initial liquid names and volumes
log('Liquid Names')
note_liquid(location=waterA, name='Water', initial_volume=waterA_initial_volume)
note_liquid(location=waterB, name='Water', initial_volume=waterB_initial_volume)
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
        p50.transfer(dilution_water_volume, water, dilution, new_tip='once', trash=config.trash_control)
        p50.transfer(dilution_source_volume, source, dilution, new_tip='once', trash=config.trash_control)

def make_dilutions(water):
    user_prompt('Make sure first dilution has been manually made')
    make_dilution(water, initial_stock, dilutions[0], 3500, 125, manual=use_manual_mix)
    make_dilution(water, dilutions[0], dilutions[1], 3500, 625 / 125, manual=False)

def plate_dilution(source, stride, parity):
    # We have two dilutions, three replicates each, so get 16 volumes per
    volumes = [5, 10, 15, 20, 25, 30, 35, 50, 60, 70, 80, 90, 100, 125, 150, 175]
    num_reps = 3
    for i, vol in enumerate(reversed(volumes)):
        log(f'Plating: source={source.get_name()} vol={vol}')
        row = i % 8
        col_first = int(i / 8)
        offset = abs(parity - (i % 2))
        p = p10 if vol <= 10 else p50
        for j in range(num_reps):
            col = (col_first * num_reps + j) * stride + offset
            dest = plate.rows(row).wells(col)
            if not p.tip_attached:
                p.pick_up_tip()
            p.transfer(vol, source, dest, trash=config.trash_control, new_tip='never')
        if p10.tip_attached: p10.done_tip()
        if p50.tip_attached: p50.done_tip()

########################################################################################################################
# Off to the races
########################################################################################################################

wells_to_verify = [dilutions[0], dilutions[1], waterA, plate.wells('A1'), plate.wells('A12'), plate.wells('H1'), plate.wells('H12')]
# verify_well_locations(wells_to_verify, p50)

log('Making Dilutions')
make_dilutions(waterA)

user_prompt('Mix dilutions by hand')

log('Plating')
plate_dilution(dilutions[1], stride=2, parity=0)
plate_dilution(waterB, stride=2, parity=1)
