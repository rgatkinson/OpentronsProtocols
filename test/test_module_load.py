
from opentrons.protocol_api import ProtocolContext

metadata = {
    'protocolName': 'Test Test Test',
    'author': 'Robert Atkinson <bob@theatkinsons.org>',
    'apiLevel': '2.0'
}

def run(protocol_context: ProtocolContext):
    temp_module = protocol_context.load_module('tempdeck', location=11)

