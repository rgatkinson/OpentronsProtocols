#
# OpentronsAnalyze.py
#
# Runs opentrons.simulate, then outputs a summary
#
import argparse
import json
import logging
import queue
import sys
from typing import List, Mapping, Any

import opentrons
import opentrons.simulate
from opentrons.legacy_api.containers import Well, Slot
from opentrons.legacy_api.containers.placeable import Placeable

########################################################################################################################
# Mixtures
########################################################################################################################

class IndeterminateVolume():
    def __init__(self):
        pass  # NYI


class Aliquot(object):
    def __init__(self, well_monitor, volume):
        self.well_monitor = well_monitor
        self.volume = volume


class Mixture(object):
    def __init__(self, initial_aliquot=None):
        self.aliquots = dict()
        if initial_aliquot is not None:
            self.adjust_aliquot(initial_aliquot)

    def get_volume(self):
        result = 0.0
        for volume in self.aliquots.values():
            result += volume
        return result

    def is_empty(self):
        return self.get_volume() == 0

    def adjust_aliquot(self, aliquot):
        assert isinstance(aliquot, Aliquot)
        existing = self.aliquots.get(aliquot.well_monitor, 0)
        existing += aliquot.volume
        assert existing >= 0
        if existing == 0:
            self.aliquots.pop(aliquot.well_monitor, None)
        else:
            self.aliquots[aliquot.well_monitor] = existing

    def adjust_mixture(self, mixture):
        assert isinstance(mixture, Mixture)
        for well_monitor, volume in mixture.aliquots.items():
            self.adjust_aliquot(Aliquot(well_monitor, volume))

    def clear(self):
        self.aliquots = dict()

    def slice(self, volume):
        existing = self.get_volume()
        assert existing >= 0
        result = Mixture()
        ratio = float(volume) / float(existing)
        for well_monitor, volume in self.aliquots.items():
            result.adjust_aliquot(Aliquot(well_monitor, volume * ratio))
        return result

    def negated(self):
        result = Mixture()
        for well_monitor, volume in self.aliquots.items():
            result.adjust_aliquot(Aliquot(well_monitor, -volume))
        return result

########################################################################################################################
# Monitors
########################################################################################################################

class Monitor(object):

    def __init__(self, controller, location_path):
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


class WellMonitor(Monitor):
    def __init__(self, controller, location_path):
        super(WellMonitor, self).__init__(controller, location_path)
        self.end = 0
        self.low_water_mark = 0
        self.high_water_mark = 0
        self.liquid_name = None
        self.mixture = Mixture()

    def _track_volume(self, volume, mixture):
        self.end = self.end + volume
        self.low_water_mark = min(self.low_water_mark, self.end)
        self.high_water_mark = max(self.high_water_mark, self.end)

    def aspirate(self, volume, mixture):
        assert volume >= 0
        self._track_volume(-volume, mixture)
        if self.mixture.is_empty():
            # Assume: aspirating from well of unknown initial volume
            # delta = Mixture(Aliquot(self, volume))
            # mixture.adjust_mixture(delta)
            pass
        else:
            # delta = self.mixture.slice(volume)
            # self.mixture.adjust_mixture(delta.negated())
            # mixture.adjust_mixture(delta)
            pass

    def dispense(self, volume, mixture):
        assert volume >= 0
        self._track_volume(volume, mixture)
        # delta = mixture.slice(volume)
        # self.mixture.adjust_mixture(delta)
        # mixture.adjust_mixture(delta.negated())

    def set_liquid_name(self, name):  # idempotent
        assert self.liquid_name is None or self.liquid_name == name
        self.liquid_name = name

    def formatted(self):
        result = 'well "{0:s}"'.format(self.target.get_name())
        if self.liquid_name is not None:
            result += ' ("{0:s}")'.format(self.liquid_name)
        result += ':'
        result += ' lo=%s hi=%s end=%s\n' % (
            self._format_number(self.low_water_mark),
            self._format_number(self.high_water_mark),
            self._format_number(self.end))
        return result

    @staticmethod
    def _format_number(value, precision=2):
        factor = 1
        for i in range(precision):
            if value * factor == int(value * factor):
                precision = i
                break
            factor *= 10
        return "{:.{}f}".format(value, precision)


class AbstractContainerMonitor(Monitor):
    def __init__(self, controller, location_path):
        super(AbstractContainerMonitor, self).__init__(controller, location_path)


class WellContainerMonitor(AbstractContainerMonitor):
    def __init__(self, controller, location_path):
        super(WellContainerMonitor, self).__init__(controller, location_path)
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
    def __init__(self, controller, location_path):
        super(TipRackMonitor, self).__init__(controller, location_path)
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


# Returns a unique name for the given location. Must track in protocols.
def get_location_path(location):
    return '/'.join(list(reversed([str(item)
                                   for item in location.get_trace(None)
                                   if str(item) is not None])))

class MonitorController(object):
    def __init__(self):
        self._monitors = dict()  # maps location path to monitor

    def note_liquid_name(self, liquid_name, location_path, volume=None):
        well_monitor = self._monitor_from_location_path(WellMonitor, location_path)
        well_monitor.set_liquid_name(liquid_name)

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
            monitor = monitor_type(self, location_path)
            self._monitors[location_path] = monitor
        return self._monitors[location_path]


########################################################################################################################
# Utility
########################################################################################################################

def log(msg: str):
    print("*********** %s ***********" % msg)

def info(msg: str, prefix="***********", suffix=' ***********'):
    print("%s%s%s%s" % (prefix, '' if len(prefix) == 0 else ' ', msg, suffix))

def warn(msg: str, prefix="***********", suffix=' ***********'):
    print("%s%sWARNING: %s%s" % (prefix, '' if len(prefix) == 0 else ' ', msg, suffix))


########################################################################################################################
# Analyzing
########################################################################################################################

def analyzeRunLog(run_log):

    controller = MonitorController()
    pipette_contents = Mixture()

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
                monitor = controller.well_monitor(well)
                if selector == 'aspirating':
                    monitor.aspirate(volume, pipette_contents)
                else:
                    monitor.dispense(volume, pipette_contents)
            elif selector == 'picking' or selector == 'dropping':
                well = placeable_from_location(payload['location'])
                rack = well.parent
                monitor = controller.tip_rack_monitor(rack)
                if selector == 'picking':
                    monitor.pick_up_tip(well)
                else:
                    monitor.drop_tip(well)
                pipette_contents.clear()
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
                controller.note_liquid_name(d['name'], d['location'], volume=d.get('volume', None))
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


def main():
    parser = argparse.ArgumentParser(prog='opentrons-analyze', description=__doc__)
    parser.add_argument(
        'protocol', metavar='PROTOCOL_FILE',
        type=argparse.FileType('r'),
        help='The protocol file to simulate (specify - to read from stdin).')
    parser.add_argument(
        '-v', '--version', action='version',
        version=f'%(prog)s {opentrons.__version__}',
        help='Print the opentrons package version and exit')
    parser.add_argument(
        '-l', '--log-level', action='store',
        help='Log level for the opentrons stack. Anything below warning can be chatty',
        choices=['error', 'warning', 'info', 'debug'],
        default='warning'
    )
    args = parser.parse_args()

    run_log = simulate(args.protocol, log_level=args.log_level)
    analysis = analyzeRunLog(run_log)
    print(opentrons.simulate.format_runlog(run_log))
    print("\n")
    print(analysis.formatted())
    return 0


# Cloned from opentrons.simulate.simulate in order to replace 'exec(proto,{})' with something more debugger-friendly
def simulate(protocol_file,
             propagate_logs=False,
             log_level='warning') -> List[Mapping[str, Any]]:
    stack_logger = logging.getLogger('opentrons')
    stack_logger.propagate = propagate_logs

    contents = protocol_file.read()
    protocol_file_name = None
    if protocol_file is not sys.stdin:
        protocol_file_name = protocol_file.name

    if opentrons.config.feature_flags.use_protocol_api_v2():
        try:
            execute_args = {'protocol_json': json.loads(contents)}
        except json.JSONDecodeError:
            execute_args = {'protocol_code': contents}
        context = opentrons.protocol_api.contexts.ProtocolContext()
        context.home()
        scraper = opentrons.simulate.CommandScraper(stack_logger, log_level, context.broker)
        execute_args.update({'simulate': True, 'context': context})
        opentrons.protocol_api.execute.run_protocol(**execute_args)
    else:
        try:
            proto = json.loads(contents)
        except json.JSONDecodeError:
            proto = contents
        opentrons.robot.disconnect()
        scraper = opentrons.simulate.CommandScraper(stack_logger, log_level, opentrons.robot.broker)
        if isinstance(proto, dict):
            opentrons.protocols.execute_protocol(proto)
        else:
            if protocol_file_name is not None:
                # https://stackoverflow.com/questions/436198/what-is-an-alternative-to-execfile-in-python-3
                code = compile(proto, protocol_file_name, 'exec')
                exec(code, {})
            else:
                exec(proto, {})
    return scraper.commands


if __name__ == '__main__':
    rc = main()
    sys.exit(rc)
