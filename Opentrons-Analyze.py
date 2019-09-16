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
from opentrons.legacy_api.containers import Well

########################################################################################################################
# Monitors
########################################################################################################################

class Monitor(object):

    def __init__(self, target):
        self.target = target


class WellMonitor(Monitor):

    def __init__(self, well):
        super(WellMonitor, self).__init__(well)
        self.net = 0
        self.low_water_mark = 0
        self.high_water_mark = 0

    def adjust_volume(self, volume):
        self.net = self.net + volume
        self.low_water_mark = min(self.low_water_mark, self.net)
        self.high_water_mark = max(self.high_water_mark, self.net)

    def aspirate(self, volume):
        self.adjust_volume(-volume)

    def dispense(self, volume):
        self.adjust_volume(volume)


class RackMonitor(Monitor):

    def __init__(self, rack):
        super(RackMonitor, self).__init__(rack)
        self.tips_picked = dict()
        self.tips_dropped = dict()

    def pick_up_tip(self, well):
        self.tips_picked[well] = 1 + self.tips_picked.get(well, 0)

    def drop_tip(self, well):
        self.tips_dropped[well] = 1 + self.tips_dropped.get(well, 0)  # trash will have multiple


########################################################################################################################
# Analyzing
########################################################################################################################

def analyzeRunLog(run_log):

    monitors = dict()

    def get_well_monitor(well):
        if well in monitors:
            return monitors[well]
        monitor = WellMonitor(well)
        monitors[well] = monitor
        return monitor

    def get_rack_monitor(rack):
        if rack in monitors:
            return monitors[rack]
        monitor = RackMonitor(rack)
        monitors[rack] = monitor
        return monitor

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
        if len(payload) <= 1: continue  # comments have just 'text'

        # payload is a dict with string keys:
        #       instrument
        #       location
        #       volume
        #       repetitions
        #       text
        #       rate
        text = payload['text']
        words = list(map(lambda word: word.lower(), text.split()))
        if len(words) == 0: continue  # paranoia
        selector = words[0]
        if selector == 'aspirating' or selector == 'dispensing':
            well = well_from_payload_location(payload)
            volume = payload['volume']
            monitor = get_well_monitor(well)
            if selector == 'aspirating':
                monitor.aspirate(volume)
            else:
                monitor.dispense(volume)
        elif selector == 'picking' or selector == 'dropping':
            well = well_from_payload_location(payload)
            rack = well.parent
            monitor = get_rack_monitor(rack)
            if selector == 'picking':
                monitor.pick_up_tip(well)
            else:
                monitor.drop_tip(well)
        else:
            pass  # nothing to process

    return monitors


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
    return 0


if __name__ == '__main__':
    rc = main()
    sys.exit(rc)
