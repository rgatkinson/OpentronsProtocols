#
# testv2.py
#
from opentrons.protocol_api import ProtocolContext

from rgatkinson.configuration import config
from rgatkinson.custom_labware import labware_manager
from rgatkinson.modules import modules_manager

metadata = {
    'protocolName': 'Test Test Test',
    'author': 'Robert Atkinson <bob@theatkinsons.org>',
    'apiLevel': '2.0'
}

def run(protocol_context: ProtocolContext):
    config.protocol_context = protocol_context
    temp_module = modules_manager.load('tempdeck', slot=11)
    screwcap_rack = temp_module.load_labware('opentrons_24_aluminumblock_generic_2ml_screwcap', label='screwcap_rack')
    plate = labware_manager.load('biorad_96_wellplate_200ul_pcr', slot=6, label='plate')

if metadata.get('apiLevel', '1.0') == '1.0':
    run(None)

