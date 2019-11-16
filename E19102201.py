"""
@author Robert Atkinson
"""

metadata = {
    'protocolName': 'Allura Red Dilution Series',
    'author': 'Robert Atkinson <bob@theatkinsons.org>'
    }

import math
from opentrons import instruments, labware, modules, robot, types
from opentrons.legacy_api.containers import WellSeries

from rgatkinson.configuration import config
from rgatkinson.custom_labware import labware_manager
from rgatkinson.liquid import note_liquid
from rgatkinson.logging import log
from rgatkinson.pipette import instruments_manager

########################################################################################################################
# Configurable protocol parameters
########################################################################################################################

# Tip usage
p10_start_tip = 'A2'
p50_start_tip = 'A1'
config.trash_control = True

stock_volume = 630

########################################################################################################################
# Labware
########################################################################################################################

# Configure the tips
tips10 = labware_manager.load('opentrons_96_tiprack_10ul', 1, label='tips10')
tips300a = labware_manager.load('opentrons_96_tiprack_300ul', 4, label='tips300a')

# Configure the pipettes.
p10 = instruments_manager.P10_Single(mount='left', tip_racks=[tips10])
p50 = instruments_manager.P50_Single(mount='right', tip_racks=[tips300a])

# Control tip usage
p10.start_at_tip(tips10[p10_start_tip])
p50.start_at_tip(tips300a[p50_start_tip])

# All the labware containers
eppendorf_1_5_rack = labware_manager.load('opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap', 5, label='eppendorf_1_5_rack')
plateA = labware_manager.load('biorad_96_wellplate_200ul_pcr', 6, label='plateA')
plateB = labware_manager.load('biorad_96_wellplate_200ul_pcr', 3, label='plateB')
trough = labware_manager.load('usascientific_12_reservoir_22ml', 9, label='trough')

# Name specific places in the labware containers
water = trough['A1']
initial_stock = eppendorf_1_5_rack['A1']
dilutions = eppendorf_1_5_rack.rows(1) + eppendorf_1_5_rack.rows(2)  # 12 in all

# Remember initial liquid names and volumes
log('Liquid Names')
note_liquid(location=water, name='Water', initially_at_least=7000)  # volume is rough guess
note_liquid(location=initial_stock, name='AlluraRed', concentration="20.1442 mM", initially=stock_volume)

########################################################################################################################
# Dilutions
########################################################################################################################

dilution_volume = 600
dilution_factor = math.sqrt(5)  # yes, that's correct

dilution_source_volume = dilution_volume / dilution_factor
dilution_water_volume = dilution_volume - dilution_source_volume

def make_dilutions():
    log('transferring water for dilutions')
    p50.transfer(dilution_water_volume, water, dilutions, new_tip='once', trash=config.trash_control)

    sources = WellSeries([initial_stock]) + dilutions[0:-1]
    destinations = dilutions
    for source, destination in zip(sources, destinations):
        log(f'diluting from {source.get_name()} to {destination.get_name()}')
        p50.transfer(dilution_source_volume, source, destination, new_tip='once', trash=config.trash_control, keep_last_tip=True)  # keep tip cause we can use it for mixing
        p50.layered_mix([destination])

def plate_dilutions():
    log('plating dilutions')
    tubes_to_plate = WellSeries([initial_stock]) + dilutions
    volumes = [50, 25, 10, 5]
    num_replicates = 3
    for row, source_tube in enumerate(tubes_to_plate):
        log(f'plating {source_tube.get_name()}')
        if row < 8:
            plate = plateA
            row_delta = 0
        else:
            plate = plateB
            row_delta = 8
        for iVolume, volume in enumerate(volumes):
            col_first = iVolume * num_replicates
            destination_wells = plate.rows(row-row_delta)[col_first:col_first+num_replicates]
            p = p10 if volume <= 10 else p50
            if not p.has_tip:  # make sure we have a tip. can reuse so long as we keep the same source tube
                p.pick_up_tip()
            p.transfer(volume, source_tube, destination_wells, new_tip='never', trash=config.trash_control)
        # We're going on to another source tube. Any tip we have is now junk
        p10.done_tip()
        p50.done_tip()


########################################################################################################################
# Off to the races
########################################################################################################################

make_dilutions()
plate_dilutions()
