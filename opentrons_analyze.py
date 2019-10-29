#
# OpentronsAnalyze.py
#
# Runs opentrons.simulate, then outputs a summary
#
import argparse
import sys
from typing import Optional

import opentrons.simulate
from opentrons.legacy_api.containers import Slot

from rgatkinson_opentrons_enhancements import *

########################################################################################################################
# Monitors
########################################################################################################################

class PlaceableMonitor(object):

    def __init__(self, config, controller, location_path):
        self.config = config;
        self.controller = controller
        self.location_path = location_path
        self.target = None

    def set_target(self, target):  # idempotent
        assert self.target is None or self.target is target
        self.target = target

    def get_slot(self):
        placeable = self.target
        assert placeable is not None  #
        while not isinstance(placeable, Slot):
            placeable = placeable.parent
        return placeable


class WellMonitor(PlaceableMonitor):
    def __init__(self, config, controller, location_path):
        super(WellMonitor, self).__init__(config, controller, location_path)
        self.volume = WellVolume(None, self.config)
        self.liquid = Liquid(location_path)  # unique to this well unless we're told a better name later
        self.liquid_known = False
        self.mixture = Mixture()

    def aspirate(self, volume, pipette_contents: PipetteContents):
        self.volume.aspirate(volume)
        self.mixture.to_pipette(volume, pipette_contents)

    def dispense(self, volume, pipette_contents: PipetteContents):
        self.volume.dispense(volume)
        self.mixture.from_pipette(volume, pipette_contents)

    def set_liquid(self, liquid):  # idempotent
        assert not self.liquid_known or self.liquid is liquid
        self.liquid = liquid
        self.liquid_known = True

    def set_initial_volume(self, initial_volume):
        self.volume.set_initial_volume(initial_volume)
        self.mixture.set_initial_liquid(self.liquid, initial_volume)

    def formatted(self):
        result = 'well "{0:s}"'.format(self.target.get_name())
        if not getattr(self.target, 'has_labelled_well_name', False):
            if self.liquid is not None:
                result += ' ("{0:s}")'.format(self.liquid)
        result += ':'
        result += Pretty().format(' lo={0:n} hi={1:n} cur={2:n} mix={3:s}\n',
            self.volume.min_volume,
            self.volume.max_volume,
            self.volume.current_volume,
            self.mixture.__str__())
        return result

# region Containers
class AbstractContainerMonitor(PlaceableMonitor):
    def __init__(self, config, controller, location_path):
        super(AbstractContainerMonitor, self).__init__(config, controller, location_path)


class WellContainerMonitor(AbstractContainerMonitor):
    def __init__(self, config, controller, location_path):
        super(WellContainerMonitor, self).__init__(config, controller, location_path)
        self.wells = dict()

    def add_well(self, well_monitor):  # idempotent
        name = well_monitor.target.get_name()
        if name in self.wells:  # avoid change on idempotency (might be iterating over self.wells)
            assert self.wells[name] is well_monitor
        else:
            self.wells[name] = well_monitor

    def formatted(self):
        result = ''
        result += 'container "%s" in "%s"\n' % (self.target.get_name(), self.get_slot().get_name())
        for well in self.target.wells():
            if self.controller.has_well(well):
                result += '   '
                result += self.controller.well_monitor(well).formatted()
        return result


class TipRackMonitor(AbstractContainerMonitor):
    def __init__(self, config, controller, location_path):
        super(TipRackMonitor, self).__init__(config, controller, location_path)
        self.tips_picked = dict()
        self.tips_dropped = dict()

    def pick_up_tip(self, well):
        self.tips_picked[well] = 1 + self.tips_picked.get(well, 0)

    def drop_tip(self, well):
        self.tips_dropped[well] = 1 + self.tips_dropped.get(well, 0)  # trash will have multiple

    def formatted(self):
        result = ''
        result += 'tip rack "%s" in "%s" picked %d tips\n' % (self.target.get_name(), self.get_slot().get_name(), len(self.tips_picked))
        return result
# endregion


class MonitorController(object):
    def __init__(self, config):
        self.config = config
        self._monitors = dict()  # maps location path to monitor
        self._liquids = dict()

    def get_liquid(self, liquid_name):
        try:
            return self._liquids[liquid_name]
        except KeyError:
            self._liquids[liquid_name] = Liquid(liquid_name)
            return self._liquids[liquid_name]

    def note_liquid_name(self, liquid_name, location_path, initial_volume=None, concentration=None):
        well_monitor = self._monitor_from_location_path(WellMonitor, location_path)
        liquid = self.get_liquid(liquid_name)
        if concentration is not None:
            concentration = Concentration(concentration)
            liquid.concentration = concentration
        well_monitor.set_liquid(liquid)
        if initial_volume is not None:
            if isinstance(initial_volume, list):  # work around json parsing deficiency
                initial_volume = Interval(*initial_volume)
            well_monitor.set_initial_volume(initial_volume)

    def well_monitor(self, well):
        well_monitor = self._monitor_from_location_path(WellMonitor, get_location_path(well))
        well_monitor.set_target(well)

        well_container_monitor = self._monitor_from_location_path(WellContainerMonitor, get_location_path(well.parent))
        well_container_monitor.set_target(well.parent)
        well_container_monitor.add_well(well_monitor)

        return well_monitor

    def has_well(self, well):
        well_monitor = self._monitors.get(get_location_path(well), None)
        return well_monitor is not None and well_monitor.target is not None

    def tip_rack_monitor(self, tip_rack):
        tip_rack_monitor = self._monitor_from_location_path(TipRackMonitor, get_location_path(tip_rack))
        tip_rack_monitor.set_target(tip_rack)
        return tip_rack_monitor

    def formatted(self):
        result = ''

        monitors = self._all_monitors(TipRackMonitor)
        slot_numbers = list(monitors.keys())
        slot_numbers.sort()
        for slot_num in slot_numbers:
            if slot_num == 12: continue  # ignore trash
            for monitor in monitors[slot_num]:
                result += monitor.formatted()

        monitors = self._all_monitors(WellContainerMonitor)
        slot_numbers = list(monitors.keys())
        slot_numbers.sort()
        for slot_num in slot_numbers:
            for monitor in monitors[slot_num]:
                result += monitor.formatted()

        return result

    def _all_monitors(self, monitor_type):  # -> map from slot number to list of monitor
        result = dict()
        for monitor in set(monitor for monitor in self._monitors.values() if monitor.target is not None):
            if isinstance(monitor, monitor_type):
                slot_num = int(monitor.get_slot().get_name())
                result[slot_num] = result.get(slot_num, list()) + [monitor]
        return result

    def _monitor_from_location_path(self, monitor_type, location_path):
        if location_path not in self._monitors:
            monitor = monitor_type(self.config, self, location_path)
            self._monitors[location_path] = monitor
        return self._monitors[location_path]


########################################################################################################################
# Utility
########################################################################################################################

def log(msg: str, prefix="***********", suffix=' ***********'):
    print(format_log_msg(msg, prefix=prefix, suffix=suffix))

def info(msg: str):
    log(msg, prefix='info:', suffix='')

def warn(msg: str, prefix="***********", suffix=' ***********'):
    log(msg, prefix=prefix + " WARNING:", suffix=suffix)


########################################################################################################################
# Hardware augmentation
#
# * allow for mounting specific instrument versions on the robot
########################################################################################################################

from opentrons.drivers.smoothie_drivers.driver_3_0 import SmoothieDriver_3_0_0
class EnhancedSimulatingSmoothieDriver(SmoothieDriver_3_0_0):

    #-------------------------------------------------------------------------------------------------------------------
    # Construction
    #-------------------------------------------------------------------------------------------------------------------

    def __new__(cls, parentInst):
        parentInst.__class__ = EnhancedSimulatingSmoothieDriver
        return parentInst

    # noinspection PyMissingConstructor
    def __init__(self, parentInst):
        self.simulated_mountings = dict()

    #-------------------------------------------------------------------------------------------------------------------
    # Accessing
    #-------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def mount(instrument_model, mount):
        self = robot._driver
        self.simulated_mountings[mount] = {'instrument_model': instrument_model}

    def read_pipette_model(self, mount) -> Optional[str]:
        if self.simulating:
            if mount in self.simulated_mountings:
                return self.simulated_mountings[mount]['instrument_model']
        return super().read_pipette_model(mount)

robot._driver = EnhancedSimulatingSmoothieDriver(robot._driver)

########################################################################################################################
# Analyzing
########################################################################################################################

def analyzeRunLog(run_log):

    controller = MonitorController(config)
    pipette_contents = PipetteContents()

    # locations are either placeables or (placable, vector) pairs
    def placeable_from_location(location):
        if isinstance(location, Placeable):
            return location
        else:
            return location[0]

    for log_item in run_log:
        # log_item is a dict with string keys:
        #       level
        #       payload
        #       logs
        payload = log_item['payload']

        # payload is a dict with string keys:
        #       instrument
        #       location
        #       volume
        #       repetitions
        #       text
        #       rate
        text = payload['text']
        lower_words = list(map(lambda word: word.lower(), text.split()))
        if len(lower_words) == 0: continue  # paranoia
        selector = lower_words[0]
        if len(payload) > 1:
            # a non-comment
            if selector == 'aspirating' or selector == 'dispensing':
                well = placeable_from_location(payload['location'])
                volume = payload['volume']
                well_monitor = controller.well_monitor(well)
                if selector == 'aspirating':
                    well_monitor.aspirate(volume, pipette_contents)
                else:
                    well_monitor.dispense(volume, pipette_contents)
            elif selector == 'picking' or selector == 'dropping':
                well = placeable_from_location(payload['location'])
                rack = well.parent
                tip_rack_monitor = controller.tip_rack_monitor(rack)
                if selector == 'picking':
                    tip_rack_monitor.pick_up_tip(well)
                    pipette_contents.pick_up_tip()
                else:
                    tip_rack_monitor.drop_tip(well)
                    pipette_contents.drop_tip()
            elif selector == 'mixing' \
                    or selector == 'transferring' \
                    or selector == 'distributing' \
                    or selector == 'blowing' \
                    or selector == 'touching' \
                    or selector == 'homing'\
                    or selector == 'setting' \
                    or selector == 'thermocycler' \
                    or selector == 'delaying' \
                    or selector == 'consolidating':
                pass
            else:
                warn('unexpected run item: %s' % text)
        else:
            # a comment
            if selector == 'liquid:':
                # Remainder after selector is json dictionary
                serialized = text[len(selector):]  # will include initial white space, but that's ok
                serialized = serialized.replace("}}", "}").replace("{{", "{")
                d = json.loads(serialized)
                controller.note_liquid_name(d['name'], d['location'], initial_volume=d.get('initial_volume', None), concentration=d.get('concentration', None))
            elif selector == 'air' \
                    or selector == 'returning' \
                    or selector == 'engaging' \
                    or selector == 'disengaging' \
                    or selector == 'calibrating' \
                    or selector == 'deactivating' \
                    or selector == 'waiting' \
                    or selector == 'setting' \
                    or selector == 'opening' \
                    or selector == 'closing' \
                    or selector == 'pausing' \
                    or selector == 'resuming':
                pass
            else:
                pass  # nothing to process

    return controller


def main() -> int:
    parser = argparse.ArgumentParser(prog='opentrons-analyze', description='Analyze an OT-2 protocol')
    parser = opentrons.simulate.get_arguments(parser)
    parser.add_argument(
        '-v', '--version', action='version',
        version=f'%(prog)s {opentrons.__version__}',
        help='Print the opentrons package version and exit')
    parser.add_argument(
        '--mount_left',
        type=str
    )
    parser.add_argument(
        '--mount_right',
        type=str
    )
    args = parser.parse_args()
    if args.mount_left or args.mount_right:
        if args.mount_left:
            EnhancedSimulatingSmoothieDriver.mount(args.mount_left, 'left')
        if args.mount_right:
            EnhancedSimulatingSmoothieDriver.mount(args.mount_right, 'right')
        robot.reset()

    run_log = opentrons.simulate.simulate(args.protocol, log_level=args.log_level)
    analysis = analyzeRunLog(run_log)
    print(opentrons.simulate.format_runlog(run_log))
    print("\n")
    print(analysis.formatted())
    return 0


def terminate_simulator_background_threads():
    # hack-o-rama, but necessary or sys.exit() won't actually terminate
    import opentrons.protocol_api.back_compat
    from opentrons.protocol_api.contexts import ProtocolContext
    rbt = opentrons.protocol_api.back_compat.robot
    protocol_context: ProtocolContext = rbt._ctx
    protocol_context._hw_manager._current.join()

if __name__ == '__main__':
    rc = main()
    terminate_simulator_background_threads()
    sys.exit(rc)
