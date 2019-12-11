#
# modules.py
#

from opentrons import modules

from rgatkinson.configuration import TopConfigurationContext
from rgatkinson.util import tls

class ModulesManager(object):
    def __init__(self):
        self.config = tls.config

    def load(self, name, slot):
        if self.config.execution_context.isApiV1:
            return modules.load(name, slot)
        return None  # WRONG

modules_manager = ModulesManager()
