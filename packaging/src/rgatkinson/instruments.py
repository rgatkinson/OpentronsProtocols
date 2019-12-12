#
# instruments.py
#
from opentrons.legacy_api.instruments import Pipette
from opentrons.protocol_api import InstrumentContext

from rgatkinson.types import PipetteName
from rgatkinson.tls import tls


class InstrumentsManager(object):

    def __init__(self):
        self.config = tls.config
        self._instruments = set()
        if self.config.isApiV1:
            from rgatkinson.perf_hacks import perf_hack_manager
            perf_hack_manager.install()

    def _add_instrument(self, instrument):
        self._instruments.add(instrument)
        return instrument

    @property
    def instruments(self):
        return self._instruments

    def P10_Single(self, mount, trash_container='', tip_racks=[], aspirate_flow_rate=None, dispense_flow_rate=None, blow_out_flow_rate=None, config=None):
        if config is None:
            config = tls.config
        result = self.create(config, PipetteName.P10_Single, mount, trash_container, tip_racks, aspirate_flow_rate, dispense_flow_rate, blow_out_flow_rate)
        return self._add_instrument(result)

    def P20_Single_GEN2(self, mount, trash_container='', tip_racks=[], aspirate_flow_rate=None, dispense_flow_rate=None, blow_out_flow_rate=None, config=None):
        if config is None:
            config = tls.config
        result = self.create(config, PipetteName.P20_Single_Gen2, mount, trash_container, tip_racks, aspirate_flow_rate, dispense_flow_rate, blow_out_flow_rate)
        return self._add_instrument(result)

    def P50_Single(self, mount, trash_container='', tip_racks=[], aspirate_flow_rate=None, dispense_flow_rate=None, blow_out_flow_rate=None, config=None):
        if config is None:
            config = tls.config
        result = self.create(config, PipetteName.P50_Single, mount, trash_container, tip_racks, aspirate_flow_rate, dispense_flow_rate, blow_out_flow_rate)
        return self._add_instrument(result)

    def P300_Single(self, mount, trash_container='', tip_racks=[], aspirate_flow_rate=None, dispense_flow_rate=None, blow_out_flow_rate=None, config=None):
        if config is None:
            config = tls.config
        result = self.create(config, PipetteName.P300_Single, mount, trash_container, tip_racks, aspirate_flow_rate, dispense_flow_rate, blow_out_flow_rate)
        return self._add_instrument(result)

    def P300_Single_GEN2(self, mount, trash_container='', tip_racks=[], aspirate_flow_rate=None, dispense_flow_rate=None, blow_out_flow_rate=None, config=None):
        if config is None:
            config = tls.config
        result = self.create(config, PipetteName.P300_Single_Gen2, mount, trash_container, tip_racks, aspirate_flow_rate, dispense_flow_rate, blow_out_flow_rate)
        return self._add_instrument(result)

    def P1000_Single(self, mount, trash_container='', tip_racks=[], aspirate_flow_rate=None, dispense_flow_rate=None, blow_out_flow_rate=None, config=None):
        if config is None:
            config = tls.config
        result = self.create(config, PipetteName.P1000_Single, mount, trash_container, tip_racks, aspirate_flow_rate, dispense_flow_rate, blow_out_flow_rate)
        return self._add_instrument(result)

    def P1000_Single_GEN2(self, mount, trash_container='', tip_racks=[], aspirate_flow_rate=None, dispense_flow_rate=None, blow_out_flow_rate=None, config=None):
        if config is None:
            config = tls.config
        result = self.create(config, PipetteName.P1000_Single_Gen2, mount, trash_container, tip_racks, aspirate_flow_rate, dispense_flow_rate, blow_out_flow_rate)
        return self._add_instrument(result)

    def create(self, config, name: PipetteName, mount=None, trash_container=None, tip_racks=None, aspirate_flow_rate=None, dispense_flow_rate=None, blow_out_flow_rate=None):
        from rgatkinson.pipettev1 import EnhancedPipetteV1
        from rgatkinson.pipettev2 import EnhancedPipetteV2
        if config.isApiV1:
            result: Pipette = name.value.v1_ctor(mount=mount,
                            trash_container=trash_container,
                            tip_racks=tip_racks,
                            aspirate_flow_rate=aspirate_flow_rate,
                            dispense_flow_rate=dispense_flow_rate,
                            min_volume=None,
                            max_volume=None,
                            blow_out_flow_rate=blow_out_flow_rate)
            return EnhancedPipetteV1.hook(config, result)
        else:
            result: InstrumentContext = config.execution_context.protocol_context.load_instrument(name.value.v2_name, mount, tip_racks)
            if aspirate_flow_rate:
                result.flow_rate.aspirate = aspirate_flow_rate
            if dispense_flow_rate:
                result.flow_rate.dispense = dispense_flow_rate
            if blow_out_flow_rate:
                result.flow_rate.blow_out = blow_out_flow_rate
            return EnhancedPipetteV2.hook(config, result)


instruments_manager = InstrumentsManager()