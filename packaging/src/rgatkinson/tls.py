#
# tls.py
#
import threading

import rgatkinson
from rgatkinson.configuration import config
from rgatkinson.configuration import TopConfigurationContext
# from rgatkinson.pipette import AspirateParamsTransfer, DispenseParamsTransfer, DispenseParams
# from rgatkinson.pipettev2 import EnhancedPipetteV2

class TLS(threading.local):
    def __init__(self):
        # This gets called on every thread we're used on. Define default values
        self.update_pose_tree_in_place = False
        self.config: TopConfigurationContext = config

        # The typing on these still isn't right
        self.enhanced_pipette: 'rgatkinson.pipettev2.EnhancedPipetteV2' = None
        self.aspirate_params_transfer: 'rgatkinson.pipette.AspirateParamsTransfer' = None
        self.dispense_params_transfer: 'rgatkinson.pipette.DispenseParamsTransfer' = None
        self.dispense_params: 'rgatkinson.pipette.DispenseParams' = None

tls: TLS = TLS()


