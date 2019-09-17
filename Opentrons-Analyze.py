#
# OpentronsAnalyze.py
#
# Runs opentrons.simulate, then outputs a summary
#
import argparse
import json
import sys
import os
import logging
import queue
from typing import Any, List, Mapping

import opentrons
import opentrons.simulate
from opentrons.legacy_api.containers import Well, Slot

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

    def __init__(self, controller, target):
        self.controller = controller
        self.target = target

    def get_slot(self):
        placeable = self.target
        while not isinstance(placeable, Slot):
            placeable = placeable.parent
        return placeable


class WellMonitor(Monitor):
    def __init__(self, controller, well):
        super(WellMonitor, self).__init__(controller, well)
        self.net = 0
        self.low_water_mark = 0
        self.high_water_mark = 0
        self.liquid_name = None
        self.mixture = Mixture()

    def track_volume(self, volume, mixture):
        self.net = self.net + volume
        self.low_water_mark = min(self.low_water_mark, self.net)
        self.high_water_mark = max(self.high_water_mark, self.net)

    def aspirate(self, volume, mixture):
        assert volume >= 0
        self.track_volume(-volume, mixture)
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
        self.track_volume(volume, mixture)
        # delta = mixture.slice(volume)
        # self.mixture.adjust_mixture(delta)
        # mixture.adjust_mixture(delta.negated())

    def set_liquid_name(self, name):
        self.liquid_name = name

    def formatted(self):
        result = ''
        result += ' well "%s": lo=%.2f hi=%.2f net=%.2f\n' % (self.target.get_name(), self.low_water_mark, self.high_water_mark, self.net)
        return result


class AbstractContainerMonitor(Monitor):
    def __init__(self, controller, rack):
        super(AbstractContainerMonitor, self).__init__(controller, rack)


class WellContainerMonitor(AbstractContainerMonitor):
    def __init__(self, controller, rack):
        super(WellContainerMonitor, self).__init__(controller, rack)
        self.wells = dict()

    def add_well(self, well_monitor):
        self.wells[well_monitor.target.get_name()] = well_monitor

    def formatted(self):
        result = ''
        result += 'container "%s" in "%s"\n' % (self.target.get_name(), self.get_slot().get_name())
        for well in self.target.wells():
            if well in self.controller.monitors:
                result += '  '
                result += self.controller.monitors[well].formatted()
        return result


class TipRackMonitor(AbstractContainerMonitor):
    def __init__(self, controller, rack):
        super(TipRackMonitor, self).__init__(controller, rack)
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


class MonitorController(object):
    def __init__(self):
        self.monitors = dict()                  # maps placeable to monitor
        self.well_container_monitors = dict()   # maps slot name to monitor
        self.tip_rack_monitors = dict()         # maps slot name to monitor

    def well_monitor(self, well):
        try:
            return self.monitors[well]
        except KeyError:
            well_monitor = WellMonitor(self, well)
            self.monitors[well] = well_monitor
            #
            well_container_monitor = WellContainerMonitor(self, well.parent)
            well_container_monitor.add_well(well_monitor)
            self.monitors[well.parent] = well_container_monitor
            self.well_container_monitors[well_container_monitor.get_slot().get_name()] = well_container_monitor
            return well_monitor

    def tip_rack_monitor(self, rack):
        try:
            return self.monitors[rack]
        except KeyError:
            tip_rack_monitor = TipRackMonitor(self, rack)
            self.monitors[rack] = tip_rack_monitor
            self.tip_rack_monitors[tip_rack_monitor.get_slot().get_name()] = tip_rack_monitor
            return tip_rack_monitor

    def formatted(self):
        result = ''
        #
        slot_names = list(self.tip_rack_monitors.keys())
        slot_names.sort()
        for slot_name in slot_names:
            if slot_name == "12": continue  # ignore trash
            result += self.tip_rack_monitors[slot_name].formatted()
        #
        slot_names = list(self.well_container_monitors.keys())
        slot_names.sort()
        for slot_name in slot_names:
            result += self.well_container_monitors[slot_name].formatted()
        #
        return result


########################################################################################################################
# Analyzing
########################################################################################################################

def analyzeRunLog(run_log):

    controller = MonitorController()
    liquid_name = None
    pipette_contents = Mixture()

    def well_from_payload_location(payload):
        if isinstance(payload['location'], Well):
            return payload['location']
        else:
            return payload['location'][0]

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
        words = text.split()
        lower_words = list(map(lambda word: word.lower(), text.split()))
        if len(lower_words) == 0: continue  # paranoia
        selector = lower_words[0]
        if selector == 'aspirating' or selector == 'dispensing':
            well = well_from_payload_location(payload)
            volume = payload['volume']
            monitor = controller.well_monitor(well)
            if selector == 'aspirating':
                if liquid_name is not None:
                    monitor.set_liquid_name(liquid_name)
                    liquid_name = None
                monitor.aspirate(volume, pipette_contents)
            else:
                monitor.dispense(volume, pipette_contents)
        elif selector == 'picking' or selector == 'dropping':
            well = well_from_payload_location(payload)
            rack = well.parent
            monitor = controller.tip_rack_monitor(rack)
            if selector == 'picking':
                monitor.pick_up_tip(well)
            else:
                monitor.drop_tip(well)
            pipette_contents.clear()
        elif selector == 'mixing' or selector == 'transferring' or selector == 'distributing' or selector == 'blowing':
            pass
        elif selector == 'liquid:':
            if len(words) < 2: continue
            liquid_name = " ".join(words[1:])
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

    run_log = opentrons.simulate.simulate(args.protocol, log_level=args.log_level)
    analysis = analyzeRunLog(run_log)
    print(opentrons.simulate.format_runlog(run_log))
    print("\n")
    print(analysis.formatted())
    return 0


if __name__ == '__main__':
    rc = main()
    sys.exit(rc)
