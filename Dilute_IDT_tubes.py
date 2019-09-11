"""
@author Robert Atkinson

Source and destination are tubes laid out in identical pattern.
Using diluent, we dilute from source to destination, ending up with
a designated volume in destination.
"""

from opentrons import labware, instruments, robot, modules

metadata = {
    'protocolName': 'Dilution in Tubes',
    'author': 'Robert Atkinson <bob@theatkinsons.org>',
    'description': 'Dilute from one set of tubes in one rack to the same set of tubes in another.'
}

########################################################################################################################
# Labware
########################################################################################################################

# Configure the tips and the pipettes
# tips10 = labware.load('opentrons_96_tiprack_10ul', 4)
# tips300 = labware.load('opentrons_96_tiprack_300ul', 1)
tips1000 = labware.load('opentrons_96_tiprack_1000uL', 1)
# p10 = instruments.P10_Single(mount='left', tip_racks=[tips10])
# p50 = instruments.P50_Single(mount='right', tip_racks=[tips300])
# p300 = instruments.P300_Single(mount='right', tip_racks=[tips300])
p1000 = instruments.P1000_Single(mount='right', tip_racks=[tips1000])
pipette = p1000

# Control tip usage
pipette.start_at_tip(tips1000['B5'])
trash_control = True  # True trashes tips; False will return trip to rack (use for debugging only)

# Define labware locations
temp_module = modules.load('tempdeck', 7)
screwcap_rack = labware.load('opentrons_24_aluminumblock_generic_2ml_screwcap', 7, label='screwcap_rack', share=True)  # IDT tubes on temp module
eppendorf_rack = labware.load('opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap', 2, label='eppendorf_rack')  # Eppendorf tubes
# falcon_rack = labware.load('opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical', 8, label='falcon_rack')
# plate = labware.load('biorad_96_wellplate_200ul_pcr', 3, label='plate')
trough = labware.load('usascientific_12_reservoir_22ml', 6, label='trough')

# Name specific places in the labware
diluent = trough.wells('A1')


########################################################################################################################
# Configurable parameters
########################################################################################################################

# Define the wells to use. Must be same in source and destination rack
# We use only those in the edge of the rack to allow IDT attached caps
# to dangle off the edge, but be sure to keep out of way of pipette
# movement (ie: keep flat)
all_wells = [
    'A1', 'A2', 'A3', 'A4', 'A5', 'A6',
#   'B1',                         'B6',
#   'C1',                         'C6',
    'D1', 'D2', 'D3', 'D4', 'D5', 'D6'
    ]

# The wells we actually use might be only a subset of these
wells = all_wells[0:10]

# Dilution factor from source to destination (x)
dilution_factor = 10.0

# Final volume in destination wells (uL)
dest_final_volume = 1200.0


########################################################################################################################
# Utilities
########################################################################################################################


def log(msg: str):
    robot.comment(msg)


########################################################################################################################
# Off to the races
########################################################################################################################

temp_module.set_temperature(37)
temp_module.wait_for_temp()
robot.pause("Continue when tubes sufficiently thawed")

# Some math to compute how much we need to move where
source_volume = dest_final_volume / dilution_factor
diluent_volume = dest_final_volume - source_volume

# Confirm math with the user
log("src=%4.1f diluent=%4.1f tot=%4.1f" % (source_volume, diluent_volume, dest_final_volume))


# First fill the destination with diluent
pipette.distribute(diluent_volume, diluent, eppendorf_rack.wells(wells),
                   new_tip='once',  # can reuse for all diluent dispensing since dest tubes are initially empty
                   trash=trash_control,
                   touch_tip=False
                   )

# Next, transfer from source to destination for each well, and mix
pipette.transfer(source_volume, screwcap_rack.wells(wells), eppendorf_rack.wells(wells),
                 new_tip='always',
                 mix_before=(3, source_volume),  # just for good measure, and because material might just have been thawed
                 mix_after=(3, source_volume),  # mix dilution thoroughly
                 trash=trash_control
                 )
