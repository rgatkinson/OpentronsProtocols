#
# hardware.py
#
from opentrons.hardware_control import adapters, API, Pipette, Controller
from opentrons.protocol_api.util import HardwareManager

from rgatkinson.tls import tls

########################################################################################################################

class EnhancedAPI(API):

    #-------------------------------------------------------------------------------------------------------------------
    # Construction
    #-------------------------------------------------------------------------------------------------------------------

    @classmethod
    def hook(cls, existingInstance: API):
        return cls(existingInstance)

    def __new__(cls, existingInstance: API):
        assert isinstance(existingInstance, API)
        existingInstance.__class__ = EnhancedAPI
        return existingInstance

    # noinspection PyMissingConstructor
    def __init__(self, existingInstance: API):
        pass

    #-------------------------------------------------------------------------------------------------------------------
    # Accessing
    #-------------------------------------------------------------------------------------------------------------------

    def smoothie_driver(self) -> 'SmoothieDriver_3_0_0':
        if self._backend is Controller:
            return self._backend._smoothie_driver
        else:
            return None

    #-------------------------------------------------------------------------------------------------------------------
    # API
    #-------------------------------------------------------------------------------------------------------------------

    def _plunger_position(self, instr: Pipette, ul: float, action: str) -> float:
        def call_super():
            return super()._plunger_position(instr, ul, action)
        if tls.enhanced_pipette:
            result = tls.enhanced_pipette.plunger_position(instr, ul, action, call_super)
        else:
            result = call_super()
        return result

    async def shake_off_tips_drop(self, mount, tiprack_diameter):
        return await super()._shake_off_tips_drop(mount, tiprack_diameter)

    def dwell(self, seconds):
        driver = self.smoothie_driver()
        if driver:
            driver.delay(seconds)


########################################################################################################################

class EnhancedHardwareManager(HardwareManager):

    #-------------------------------------------------------------------------------------------------------------------
    # Construction
    #-------------------------------------------------------------------------------------------------------------------

    @classmethod
    def hook(cls, existingInstance: HardwareManager):
        return cls(existingInstance)

    def __new__(cls, existingInstance: HardwareManager):
        assert isinstance(existingInstance, HardwareManager)
        existingInstance.__class__ = EnhancedHardwareManager
        return existingInstance

    # noinspection PyMissingConstructor
    def __init__(self, existingInstance: HardwareManager):
        self.config = tls.config
        #
        # more to come
        #

    #-------------------------------------------------------------------------------------------------------------------
    # Accessing
    #-------------------------------------------------------------------------------------------------------------------

    def enhanced_hardware(self):
        target = super().hardware
        if isinstance(target, adapters.SynchronousAdapter):
            target = target._api
        assert isinstance(target, API)
        if not isinstance(target, EnhancedAPI):
            EnhancedAPI.hook(target)
        return target

    @property
    def hardware(self):
        self.enhanced_hardware()  # make sure hook is in place
        return super().hardware




