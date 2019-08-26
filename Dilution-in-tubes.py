"""
@author Robert Atkinson
@date August 23rd, 2019

Source and destination are tubes laid out in identical pattern.
Using diluent, we dilute from source to destination, ending up with
a designated volume in destination.
"""

from opentrons import labware, instruments, robot, modules

metadata = {
    'protocolName': 'Dilution in Tubes',
    'author': 'Robert Atkinson <bob@theatkinsons.org>',
    'description': 'Dilute from one set of tubes in one rack to the same set of tubs in another.'
}

# Indicate where to find the diluent
diluent_well = labware.load('usascientific_12_reservoir_22ml', 2).wells('A1')

# Define the source and destination racks. Note that the tethered screwcaps
# overhang the unit slot area, so we're careful to separate things well
source_rack = labware.load('opentrons_24_tuberack_generic_2ml_screwcap', 8)  # IDT tubes
dest_rack = labware.load('opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap', 3)  # Eppendorf tubes

# Define the set of source and destination wells to use, in both
# the source and the destination rack
wells = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6',
         'B1', 'B2', 'B3', 'B4', 'B5', 'B6',
         'C1',                         'C6',
         'D1', 'D2', 'D3', 'D4', 'D5', 'D6'
         ]

# Dilution factor from source to destination
dilution_factor = 10.0

# Final volume in destination wells (uL)
dest_final_volume = 100.0

# Some math to compute how much we need to move where
source_volume = dest_final_volume / dilution_factor
diluent_volume = dest_final_volume - source_volume

# Confirm math with the user
robot.pause("src=%f.1f diluent=%4.1f" % (source_volume, diluent_volume))

# Configure the tips and the pipettes
tips10 = [labware.load('opentrons_96_tiprack_10ul', 4)]
tips300 = [labware.load('opentrons_96_tiprack_300ul', 1)]
p10 = instruments.P10_Single(mount='left', tip_racks=tips10)
p50 = instruments.P50_Single(mount='right', tip_racks=tips300)

# Control tip usage while we're debugging WRONG WRONG WRONG WRONG
dirty_tip_control = 'always'
trash_control = False

# Off to the races! First fill the destination with diluent
p50.distribute(diluent_volume, diluent_well, dest_rack.wells(wells),
               new_tip='once',  # can reuse for all diluent dispensing
               trash=trash_control
               )

# Next, transfer from source to destination for each well, and mix
p10.transfer(source_volume, source_rack.wells(wells), dest_rack.wells(wells),
             mix_before=(3, source_volume),  # just for good measure
             mix_after=(3, source_volume),  # mix dilution thoroughly
             new_tip=dirty_tip_control,
             trash=trash_control
             )
