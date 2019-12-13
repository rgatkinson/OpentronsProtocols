#
# modules.py
#
import asyncio
import functools
import typing
from opentrons import modules, types as otypes
from opentrons.hardware_control.adapters import SynchronousAdapter
from opentrons.legacy_api.containers.placeable import Module, Slot
from opentrons.legacy_api.modules import MagDeck, TempDeck
from opentrons.protocol_api.contexts import ModuleContext
from opentrons.protocol_api.labware import load as oload

from rgatkinson.custom_labware import labware_manager
from rgatkinson.tls import tls

ModulesTypeV1 = typing.Union[MagDeck, TempDeck]

class SynchronousAdapterFixed(SynchronousAdapter):

    #-------------------------------------------------------------------------------------------------------------------
    # Class methods
    #-------------------------------------------------------------------------------------------------------------------

    @classmethod
    def fix_bugs(cls):
        # SynchronousAdapter.__getattribute__, when called with 'name', incorrectly returns
        # a bound method (e.g. TempDeck.name) rather than a string, which is what the PyCharm
        # debugger requires; this crashes the debugger
        cls.hook()

    _is_hooked = False

    @classmethod
    def hook(cls):
        if not cls._is_hooked:
            cls.hook_class()
            cls._is_hooked = True

    @classmethod
    def hook_class(cls):
        SynchronousAdapter.__getattribute__ = cls.get_attribute_fixed

    #-------------------------------------------------------------------------------------------------------------------
    # Instance methods
    #-------------------------------------------------------------------------------------------------------------------

    def get_attribute_fixed(self, attr_name):
        """ Retrieve attributes from our API and wrap coroutines """
        # Almost every attribute retrieved from us will be fore people actually
        # looking for an attribute of the hardware API, so check there first.

        if attr_name == 'name':
            return super(SynchronousAdapter, self).name

        if attr_name == 'discover_modules':
            return object.__getattribute__(self, attr_name)

        api = object.__getattribute__(self, '_api')
        try:
            attr = getattr(api, attr_name)
        except AttributeError:
            # Maybe this actually was for us? Let’s find it
            return object.__getattribute__(self, attr_name)

        try:
            check = attr.__wrapped__
        except AttributeError:
            check = attr
        loop = object.__getattribute__(self, '_loop')
        if asyncio.iscoroutinefunction(check):
            # Return a synchronized version of the coroutine
            return functools.partial(self.call_coroutine_sync, loop, attr)
        elif asyncio.iscoroutine(check):
            # Catch awaitable properties and reify the future before returning
            fut = asyncio.run_coroutine_threadsafe(check, loop)
            return fut.result()

        return attr

class ModulesManager(object):

    def __init__(self):
        self.config = tls.config
        self.checked_fix = False

    def check_fix(self):
        if not self.checked_fix:
            if not self.config.isApiV1:
                SynchronousAdapterFixed.fix_bugs()
            self.checked_fix = True

    def load(self, name, slot):
        self.check_fix()

        if self.config.isApiV1:
            result = modules.load(name, slot)
        else:
            result = self.config.protocol_context.load_module(name, slot)

        def load_labware_func(
                theModule,
                load_name: str,
                label: str = None,
                namespace: str = None,
                version: int = None
                ):
            if self.config.isApiV1:
                theModule1 = typing.cast(ModulesTypeV1, theModule)
                lw_module: Module = theModule1.labware
                slot: Slot = lw_module.parent
                location = slot.get_name()
                lw = labware_manager.load(name=load_name, slot=location, label=label, share=True, namespace=namespace, version=version, config=self.config)
                return lw
            else:
                theModule2 = typing.cast(ModuleContext, theModule)
                location = theModule2.geometry.location
                lw = labware_manager.load(name=load_name, slot=location, label=label, share=True, namespace=namespace, version=version, config=self.config)
                return theModule2.load_labware_object(lw)

        # add for v1, improve what's there for v2
        result.__class__.load_labware = load_labware_func

        return result


modules_manager = ModulesManager()
