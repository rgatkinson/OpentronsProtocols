"""
@author Robert Atkinson
"""

metadata = {
    'protocolName': 'Test Precision by Mass',
    'author': 'Robert Atkinson <bob@theatkinsons.org>',
    'description': 'Tests precision by pipetting water in various mass sizes'
}

from opentrons import instruments, labware, modules, robot, types

from rgatkinson import *
from rgatkinson.custom_labware import labware_manager
from rgatkinson.liquid import note_liquid
from rgatkinson.logging import log
from rgatkinson.pipette_v1 import instruments_manager

########################################################################################################################
# Configurable protocol parameters
########################################################################################################################

num_replicates = 3   # down rows
num_masses = 6       # across columns
mass_incr_vol = 200

start_tip = 'A1'
tips_vol = '300'
pipette_vol = '50'
pipette_mount = 'right'

########################################################################################################################
# Labware
########################################################################################################################

# Configure the pipette
tips = labware_manager.load('opentrons_96_tiprack_' + tips_vol + 'ul', 1)
p = getattr(instruments_manager, 'P' + pipette_vol + '_Single')(config, mount=pipette_mount, tip_racks=[tips])
p.set_flow_rate(blow_out=p.get_flow_rates()['blow_out'] * config.blow_out_rate_factor)
p.start_at_tip(start_tip)
config.trash_control = True

# All the labware containers
eppendorf_1_5_rack = labware_manager.load('opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap', 5, label='eppendorf_1_5_rack')
trough = labware_manager.load('usascientific_12_reservoir_22ml', 9, label='trough')

# Name specific places in the labware containers
water = trough['A1']
mass_wells = []
for i in range(num_masses):
    vol = (i + 1) * mass_incr_vol
    for j in range(num_replicates):
        well = eppendorf_1_5_rack.cols(i).wells(j)
        well.mass_vol = vol
        mass_wells.append(well)

for well in mass_wells:
    Eppendorf1Point5MlTubeGeometryV1(well)

log('Liquid Names')
note_liquid(location=water, name='Water', initially_at_least=15000)  # volume is rough guess
for well in mass_wells:
    note_liquid(location=well, name=pretty.format('mass_vol={0:n}', well.mass_vol))

# Clean up namespace
del well_v1, i, j

########################################################################################################################
# Off to the races
########################################################################################################################

for well in mass_wells:
    p.transfer(well.mass_vol, water, well,
               new_tip='once',
               trash=config.trash_control,
               allow_blow_elision=True,
               allow_carryover=True,
               pre_wet=True)
