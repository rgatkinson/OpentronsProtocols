#
# modules.py
#

from opentrons import modules

from rgatkinson.util import tls

class ModulesManager(object):
    def __init__(self):
        self.config = tls.config

    def load(self, name, slot):
        if self.config.execution_context.isApiV1:
            result = modules.load(name, slot)
        else:
            result = self.config.protocol_context.load_module(name, slot)
        return result


modules_manager = ModulesManager()
