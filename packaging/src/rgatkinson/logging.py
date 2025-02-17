#
# logging.py
#

import string
import warnings

import opentrons
from opentrons import robot
from opentrons.commands import stringify_location, make_command, command_types
from opentrons.legacy_api.containers.placeable import Placeable

import rgatkinson
from rgatkinson.math_util import is_finite_scalar, is_close


#-----------------------------------------------------------------------------------------------------------------------

class Pretty(string.Formatter):
    def format_field(self, value, spec):
        if spec.endswith('n'):  # 'n' for number
            precision = 2
            if spec.startswith('.', 0, -1):
                precision = int(spec[1:-1])
            if is_finite_scalar(value):
                factor = 1
                abs_val = abs(value)
                for i in range(precision):
                    no_decimals = int(abs_val * factor + 0.5)
                    scaled = abs_val * factor
                    if is_close(scaled, no_decimals):
                        precision = i
                        break
                    factor *= 10
                fmt = f'{{:.{ precision }f}}'
                result = fmt.format(value)
                return result
            elif hasattr(value, 'format'):
                return value.format(format_spec="{0:%s}" % spec, formatter=self)
            else:
                return str(value)
        return super().format_field(value, spec)

pretty = Pretty()

#-----------------------------------------------------------------------------------------------------------------------

def format_log_msg(msg: str, prefix="***********", suffix=' ***********'):
    return "%s%s%s%s" % (prefix, '' if len(prefix) == 0 else ' ', msg, suffix)

def log(msg: str, prefix="***********", suffix=' ***********'):
    robot.comment(format_log_msg(msg, prefix=prefix, suffix=suffix))

def log_while_core(msg: str, func, prefix='', suffix=''):
    msg = format_log_msg(msg, prefix, suffix)
    def comment_func(msg):
        pass
    opentrons.commands.do_publish(robot.broker, opentrons.commands.comment, f=comment_func, when='before', res=None, meta=None, msg=msg)
    if func is not None:
        func()
    opentrons.commands.do_publish(robot.broker, opentrons.commands.comment, f=comment_func, when='after', res=None, meta=None, msg=msg)

def log_while(msg: str, func, prefix="***********", suffix=' ***********'):
    log_while_core(msg, func, prefix, suffix)

def info(msg):
    log(msg, prefix='info:', suffix='')

def info_while(msg, func):
    log_while_core(msg, func, prefix='info:', suffix='')

def warn(msg: str, prefix="***********", suffix=' ***********'):
    log(msg, prefix=prefix + " WARNING:", suffix=suffix)

def fatal(msg: str, prefix="***********", suffix=' ***********'):
    formatted = format_log_msg(msg, prefix=prefix + " FATAL ERROR:", suffix=suffix)
    warnings.warn(formatted, stacklevel=2)
    log(formatted, prefix='', suffix='')
    raise RuntimeError  # could do better

def silent_log(msg):
    pass

def user_prompt(msg: str, prefix="***********", suffix=' ***********', pause=True):
    if msg:
        log(msg=msg, prefix=prefix, suffix=suffix)
    if pause:
        robot.pause('Press Return to Continue :-)')

#-----------------------------------------------------------------------------------------------------------------------

def get_location_path(location):
    result = getattr(location, 'location_path', None)
    if result is None:
        result = '/'.join(list(reversed([str(item)
                                       for item in location.get_trace(None)
                                       if str(item) is not None])))
        location.location_path = result
    return result

#-----------------------------------------------------------------------------------------------------------------------

def _z_from_bottom(location, clearance):
    if isinstance(location, Placeable):
        return min(location.z_size(), clearance)
    elif isinstance(location, tuple):
        well, vector = location
        _, vector_bottom = well.bottom(0)
        return vector.coordinates.z - vector_bottom.coordinates.z
    else:
        raise ValueError('Location should be (Placeable, (x, y, z)) or Placeable')

def command_aspirate(instrument, volume, location, rate):
    local_config = instrument.config if hasattr(instrument, 'config') else rgatkinson.configuration.config
    z = _z_from_bottom(location, local_config.aspirate.bottom_clearance)
    location_text = stringify_location(location)
    text = pretty.format('Aspirating {volume:n} uL z={z:n} rate={rate:n} at {location}', volume=volume, location=location_text, rate=rate, z=z)
    return make_command(
        name=command_types.ASPIRATE,
        payload={
            'instrument': instrument,
            'volume': volume,
            'location': location,
            'rate': rate,
            'text': text
        }
    )

def command_dispense(instrument, volume, location, rate):
    local_config = instrument.config if hasattr(instrument, 'config') else rgatkinson.configuration.config
    z = _z_from_bottom(location, local_config.dispense.bottom_clearance)
    location_text = stringify_location(location)
    text = pretty.format('Dispensing {volume:n} uL z={z:n} rate={rate:n} at {location}', volume=volume, location=location_text, rate=rate, z=z)
    return make_command(
        name=command_types.DISPENSE,
        payload={
            'instrument': instrument,
            'volume': volume,
            'location': location,
            'rate': rate,
            'text': text
        }
    )

# Hook commands to provide more informative text
opentrons.commands.aspirate = command_aspirate
opentrons.commands.dispense = command_dispense
