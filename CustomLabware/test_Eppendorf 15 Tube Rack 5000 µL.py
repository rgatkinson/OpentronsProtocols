import json
from opentrons import robot, labware, instruments

CALIBRATION_CROSS_COORDS = [380.87, 9.0, 0.0]
CALIBRATION_CROSS_SLOT = '3'
TEST_LABWARE_SLOT = CALIBRATION_CROSS_SLOT
TIPRACK_SLOT = '5'

RATE = 0.25  # % of default speeds
SLOWER_RATE = 0.1


def uniq(l):
    res = []
    for i in l:
        if i not in res:
            res.append(i)
    return res


def set_speed(rate):
    robot.head_speed(x=(600 * rate), y=(400 * rate),
                      z=(125 * rate), a=(125 * rate))


def run_custom_protocol(pipette_name, mount, tiprack_load_name, labware_def):
    tiprack = labware.load(tiprack_load_name, TIPRACK_SLOT)
    pipette = getattr(instruments, pipette_name)(mount, tip_racks=[tiprack])
    test_labware = robot.add_container_by_definition(
        labware_def,
        TEST_LABWARE_SLOT,
        label=labware_def.get('metadata', {}).get(
            'displayName', 'test labware')
    )

    num_cols = len(labware_def.get('ordering', [[]]))
    num_rows = len(labware_def.get('ordering', [[]])[0])
    well_locs = uniq([
        'A1',
        '{}{}'.format(chr(ord('A') + num_rows - 1), str(num_cols))])

    pipette.pick_up_tip()
    set_speed(RATE)

    pipette.move_to((robot.deck, CALIBRATION_CROSS_COORDS))
    robot.pause(
        f"Confirm {mount} pipette is at slot {CALIBRATION_CROSS_SLOT} calibration cross")

    pipette.retract()
    robot.pause(f"Place your labware in Slot {TEST_LABWARE_SLOT}")

    for well_loc in well_locs:
        well = test_labware.wells(well_loc)
        all_4_edges = [
            [well.from_center(x=-1, y=0, z=1), 'left'],
            [well.from_center(x=1, y=0, z=1), 'right'],
            [well.from_center(x=0, y=-1, z=1), 'front'],
            [well.from_center(x=0, y=1, z=1), 'back']
        ]

        set_speed(RATE)
        pipette.move_to(well.top())
        robot.pause("Moved to the top of the well")

        for edge_pos, edge_name in all_4_edges:
            set_speed(SLOWER_RATE)
            pipette.move_to((well, edge_pos))
            robot.pause(f'Moved to {edge_name} edge')

        set_speed(RATE)
        pipette.move_to(well.bottom())
        robot.pause("Moved to the bottom of the well")

        # need to interact with labware for it to show on deck map
        pipette.blow_out(well)


    set_speed(1.0)
    pipette.return_tip()


LABWARE_DEF = """{"ordering":[["A1","B1","C1"],["A2","B2","C2"],["A3","B3","C3"],["A4","B4","C4"],["A5","B5","C5"]],"brand":{"brand":"Eppendorf","brandId":["0030108310"]},"metadata":{"displayName":"Eppendorf 15 Tube Rack 5000 µL","displayCategory":"tubeRack","displayVolumeUnits":"µL","tags":[]},"dimensions":{"xDimension":127.76,"yDimension":85.48,"zDimension":80.83},"wells":{"A1":{"depth":55.4,"totalLiquidVolume":5000,"shape":"circular","diameter":13.93,"x":13.88,"y":67.74,"z":25.43},"B1":{"depth":55.4,"totalLiquidVolume":5000,"shape":"circular","diameter":13.93,"x":13.88,"y":42.74,"z":25.43},"C1":{"depth":55.4,"totalLiquidVolume":5000,"shape":"circular","diameter":13.93,"x":13.88,"y":17.74,"z":25.43},"A2":{"depth":55.4,"totalLiquidVolume":5000,"shape":"circular","diameter":13.93,"x":38.88,"y":67.74,"z":25.43},"B2":{"depth":55.4,"totalLiquidVolume":5000,"shape":"circular","diameter":13.93,"x":38.88,"y":42.74,"z":25.43},"C2":{"depth":55.4,"totalLiquidVolume":5000,"shape":"circular","diameter":13.93,"x":38.88,"y":17.74,"z":25.43},"A3":{"depth":55.4,"totalLiquidVolume":5000,"shape":"circular","diameter":13.93,"x":63.88,"y":67.74,"z":25.43},"B3":{"depth":55.4,"totalLiquidVolume":5000,"shape":"circular","diameter":13.93,"x":63.88,"y":42.74,"z":25.43},"C3":{"depth":55.4,"totalLiquidVolume":5000,"shape":"circular","diameter":13.93,"x":63.88,"y":17.74,"z":25.43},"A4":{"depth":55.4,"totalLiquidVolume":5000,"shape":"circular","diameter":13.93,"x":88.88,"y":67.74,"z":25.43},"B4":{"depth":55.4,"totalLiquidVolume":5000,"shape":"circular","diameter":13.93,"x":88.88,"y":42.74,"z":25.43},"C4":{"depth":55.4,"totalLiquidVolume":5000,"shape":"circular","diameter":13.93,"x":88.88,"y":17.74,"z":25.43},"A5":{"depth":55.4,"totalLiquidVolume":5000,"shape":"circular","diameter":13.93,"x":113.88,"y":67.74,"z":25.43},"B5":{"depth":55.4,"totalLiquidVolume":5000,"shape":"circular","diameter":13.93,"x":113.88,"y":42.74,"z":25.43},"C5":{"depth":55.4,"totalLiquidVolume":5000,"shape":"circular","diameter":13.93,"x":113.88,"y":17.74,"z":25.43}},"groups":[{"metadata":{"displayName":"Eppendorf 15 Tube Rack 5000 µL","displayCategory":"tubeRack","wellBottomShape":"v"},"brand":{"brand":"Eppendorf","brandId":["0030108310"]},"wells":["A1","B1","C1","A2","B2","C2","A3","B3","C3","A4","B4","C4","A5","B5","C5"]}],"parameters":{"format":"irregular","quirks":[],"isTiprack":false,"isMagneticModuleCompatible":false,"loadName":"eppendorf_15_tuberack_5000ul"},"namespace":"custom_beta","version":1,"schemaVersion":2,"cornerOffsetFromSlot":{"x":0,"y":0,"z":0}}"""

PIPETTE_MOUNT = 'right'
PIPETTE_NAME = 'P300_Single'
TIPRACK_LOADNAME = 'opentrons_96_tiprack_300ul'

run_custom_protocol(PIPETTE_NAME, PIPETTE_MOUNT,
                    TIPRACK_LOADNAME, json.loads(LABWARE_DEF))
