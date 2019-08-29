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

# Indicate where to find the diluent
diluent_well = labware.load('usascientific_12_reservoir_22ml', 2).wells('A1')

# Define the source and destination racks. Note that the tethered screwcaps
# overhang the unit slot area, so we're careful to separate the slots used
source_rack = labware.load('opentrons_24_tuberack_generic_2ml_screwcap', 8)  # IDT tubes
dest_rack = labware.load('opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap', 3)  # Eppendorf tubes

# Define the set of source and destination wells to use, in both
# the source and the destination rack
wells = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6',
         'B1', 'B2', 'B3', 'B4', 'B5', 'B6',
         'C1',                         'C6',
         'D1', 'D2', 'D3', 'D4', 'D5', 'D6'
         ]

# Dilution factor from source to destination (x)
dilution_factor = 10.0

# Final volume in destination wells (uL)
dest_final_volume = 1200.0

# Some math to compute how much we need to move where
source_volume = dest_final_volume / dilution_factor
diluent_volume = dest_final_volume - source_volume

# Confirm math with the user
robot.pause("src=%4.1f diluent=%4.1f tot=%4.1f" % (source_volume, diluent_volume, dest_final_volume))

# Configure the tips and the pipettes
tips10 = labware.load('opentrons_96_tiprack_10ul', 4)
tips300 = labware.load('opentrons_96_tiprack_300ul', 1)
# p10 = instruments.P10_Single(mount='left', tip_racks=[tips10])
# p50 = instruments.P50_Single(mount='right', tip_racks=[tips300])
p300 = instruments.P300_Single(mount='right', tip_racks=[tips300])

# Control tip usage
p300.start_at_tip(tips300['A1'])
trash_control = True  # False will return trip to rack (use for debugging only)

# First fill the destination with diluent
p300.distribute(diluent_volume, diluent_well, dest_rack.wells(wells),
                new_tip='once',  # can reuse for all diluent dispensing since dest tubes are initially empty
                trash=trash_control
                )

# Next, transfer from source to destination for each well, and mix
p300.transfer(source_volume, source_rack.wells(wells), dest_rack.wells(wells),
              mix_before=(2, source_volume),  # just for good measure, and because material might just have been thawed
              mix_after=(3, source_volume),  # mix dilution thoroughly
              trash=trash_control
              )
