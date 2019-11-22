#
# OpentronsAnalyze.py
#
# Runs opentrons.simulate, then outputs a summary
#
import argparse
import json
import sys
from typing import Optional

import opentrons.simulate
from opentrons import robot
from opentrons.legacy_api.containers import Slot
from opentrons.legacy_api.containers.placeable import Placeable

from rgatkinson.configuration import config, AbstractConfigurationContext
from rgatkinson.interval import Interval, infimum
from rgatkinson.liquid import Liquid, Concentration, Mixture, PipetteContents, LiquidVolume
from rgatkinson.logging import Pretty, get_location_path, format_log_msg, pretty


########################################################################################################################
# Analyzers
########################################################################################################################


class PlaceableAnalyzer(object):

    def __init__(self, config, manager, location_path):
        self.config = config
        self.manager = manager
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


class WellAnalyzer(PlaceableAnalyzer):
    def __init__(self, config, manager, location_path):
        super(WellAnalyzer, self).__init__(config, manager, location_path)
        self.liquid_volume = LiquidVolume(None)
        self.liquid = Liquid(location_path)  # unique to this well unless we're told a better name later
        self.liquid_known = False
        self.mixture = Mixture()

    def set_target(self, target):
        super().set_target(target)
        # Note: we *don't* want to set target.liquid_volume, as *our* liquid_volume is a post-run analyzer, not a real-time tracker

    def aspirate(self, volume, pipette_contents: PipetteContents):
        self.liquid_volume.aspirate(volume)
        self.mixture.to_pipette(volume, pipette_contents)

    def dispense(self, volume, pipette_contents: PipetteContents):
        self.liquid_volume.dispense(volume)
        self.mixture.from_pipette(volume, pipette_contents)

    def set_liquid(self, liquid):  # idempotent
        assert not self.liquid_known or self.liquid is liquid
        self.liquid = liquid
        self.liquid_known = True

    def set_initially(self, initially):
        self.liquid_volume.set_initially(initially)
        self.mixture.set_initial_liquid(self.liquid, initially)

    def formatted(self):
        result = 'well "{0:s}"'.format(self.target.get_name())
        if not getattr(self.target, 'has_labelled_well_name', False):
            if self.liquid is not None:
                result += ' ("{0:s}")'.format(self.liquid)
        result += ':'
        result += pretty.format(' lo={0:n} hi={1:n} cur={2:n} taken={3:n} mix={4:s}\n',
                                  self.liquid_volume.lo_volume,
                                  self.liquid_volume.hi_volume,
                                  self.liquid_volume.current_volume,
                                  infimum(self.liquid_volume.hi_volume) - infimum(self.liquid_volume.current_volume),
                                  self.mixture.__str__())
        return result

# region Containers
class AbstractContainerAnalyzer(PlaceableAnalyzer):
    def __init__(self, config, manager, location_path):
        super(AbstractContainerAnalyzer, self).__init__(config, manager, location_path)


class WellContainerAnalyzer(AbstractContainerAnalyzer):
    def __init__(self, config, manager, location_path):
        super(WellContainerAnalyzer, self).__init__(config, manager, location_path)
        self.wells = dict()

    def add_well(self, well_analyzer):  # idempotent
        name = well_analyzer.target.get_name()
        if name in self.wells:  # avoid change on idempotency (might be iterating over self.wells)
            assert self.wells[name] is well_analyzer
        else:
            self.wells[name] = well_analyzer

    def formatted(self):
        result = ''
        result += 'container "%s" in "%s"\n' % (self.target.get_name(), self.get_slot().get_name())
        for well in self.target.wells():
            if self.manager.has_well(well):
                result += '   '
                result += self.manager.well_analyzer(well).formatted()
        return result


class TipRackAnalyzer(AbstractContainerAnalyzer):
    def __init__(self, config, manager, location_path):
        super(TipRackAnalyzer, self).__init__(config, manager, location_path)
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


class AnalyzerManager(object):
    def __init__(self, config: AbstractConfigurationContext):
        self.config = config
        self._analyzers = dict()  # maps location path to analyzer

    def get_liquid(self, liquid_name):
        return self.config.execution_context.liquids.get_liquid(liquid_name)

    def analyze_liquid_name(self, liquid_name, location_path, initially=None, concentration=None):
        # Keep in sync with (global) note_liquid_name
        well_analyzer: WellAnalyzer = self._analyzer_from_location_path(WellAnalyzer, location_path)
        liquid = self.get_liquid(liquid_name)
        if concentration is not None:
            concentration = Concentration(concentration)
            liquid.concentration = concentration
        well_analyzer.set_liquid(liquid)
        if initially is not None:
            well_analyzer.set_initially(initially)

    def well_analyzer(self, well):
        well_analzyer = self._analyzer_from_location_path(WellAnalyzer, get_location_path(well))
        well_analzyer.set_target(well)

        well_container_analyzer = self._analyzer_from_location_path(WellContainerAnalyzer, get_location_path(well.parent))
        well_container_analyzer.set_target(well.parent)
        well_container_analyzer.add_well(well_analzyer)

        return well_analzyer

    def has_well(self, well):
        well_anzlyzer = self._analyzers.get(get_location_path(well), None)
        return well_anzlyzer is not None and well_anzlyzer.target is not None

    def tip_rack_analyzer(self, tip_rack):
        tip_rack_analyzer = self._analyzer_from_location_path(TipRackAnalyzer, get_location_path(tip_rack))
        tip_rack_analyzer.set_target(tip_rack)
        return tip_rack_analyzer

    def formatted(self):
        result = ''

        analyzers = self._all_analyzers(TipRackAnalyzer)
        slot_numbers = list(analyzers.keys())
        slot_numbers.sort()
        for slot_num in slot_numbers:
            if slot_num == 12: continue  # ignore trash
            for analyzer in analyzers[slot_num]:
                result += analyzer.formatted()

        analyzers = self._all_analyzers(WellContainerAnalyzer)
        slot_numbers = list(analyzers.keys())
        slot_numbers.sort()
        for slot_num in slot_numbers:
            for analyzer in analyzers[slot_num]:
                result += analyzer.formatted()

        return result

    def _all_analyzers(self, analyzer_type):  # -> map from slot number to list of analyzer
        result = dict()
        for analyzer in set(analyzer for analyzer in self._analyzers.values() if analyzer.target is not None):
            if isinstance(analyzer, analyzer_type):
                slot_num = int(analyzer.get_slot().get_name())
                result[slot_num] = result.get(slot_num, list()) + [analyzer]
        return result

    def _analyzer_from_location_path(self, analyzer_type, location_path):
        if location_path not in self._analyzers:
            analyzer = analyzer_type(self.config, self, location_path)
            self._analyzers[location_path] = analyzer
        return self._analyzers[location_path]


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

    manager = AnalyzerManager(config)
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
                well_analyzer = manager.well_analyzer(well)
                if selector == 'aspirating':
                    well_analyzer.aspirate(volume, pipette_contents)
                else:
                    well_analyzer.dispense(volume, pipette_contents)
            elif selector == 'picking' or selector == 'dropping':
                well = placeable_from_location(payload['location'])
                rack = well.parent
                tip_rack_analyzer = manager.tip_rack_analyzer(rack)
                if selector == 'picking':
                    tip_rack_analyzer.pick_up_tip(well)
                    pipette_contents.pick_up_tip()
                else:
                    tip_rack_analyzer.drop_tip(well)
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
                manager.analyze_liquid_name(d['name'], d['location'], initially=d.get('initially', None), concentration=d.get('concentration', None))
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

    return manager


def main() -> int:
    parser = argparse.ArgumentParser(prog='opentrons-analyze', description='Analyze an OT-2 protocol')
    parser = opentrons.simulate.get_arguments(parser)
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
        robot.reset()  # pick up the new definitions

    run_log_and_bundle = opentrons.simulate.simulate(args.protocol, args.protocol.name, log_level=args.log_level)
    run_log = run_log_and_bundle[0]
    analysis = analyzeRunLog(run_log)
    print(opentrons.simulate.format_runlog(run_log))
    print("\n")
    print(analysis.formatted())
    return 0


def terminate_simulator_background_threads():
    return  # seems to be no longer needed ????
    # hack-o-rama, but necessary or sys.exit() won't actually terminate
    import opentrons.protocol_api.back_compat  # module has been removed
    from opentrons.protocol_api.contexts import ProtocolContext
    rbt = opentrons.protocol_api.back_compat.robot
    protocol_context: ProtocolContext = rbt._ctx
    protocol_context._hw_manager._current.join()

if __name__ == '__main__':
    rc = main()
    terminate_simulator_background_threads()
    sys.exit(rc)
