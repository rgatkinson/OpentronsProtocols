#
# tls.py
#
import threading
import typing

import rgatkinson
from rgatkinson.configuration import config
from rgatkinson.configuration import TopConfigurationContext
# from rgatkinson.pipette import AspirateParamsTransfer, DispenseParamsTransfer, DispenseParams
# from rgatkinson.pipettev2 import EnhancedPipetteV2

# make thread local storage
class TLS(threading.local):
    def __init__(self):
        # This gets called on every thread we're used on. Define default values
        self.update_pose_tree_in_place = False
        self.config: TopConfigurationContext = config
        self.enhanced_pipette: typing.TypeVar
        self.enhanced_pipette: 'EnhancedPipetteV2' = None
        self.aspirate_params_transfer: 'AspirateParamsTransfer' = None
        self.dispense_params_transfer: 'DispenseParamsTransfer' = None
        self.dispense_params: 'DispenseParams' = None

tls: TLS = TLS()


