#
# performance hacks
#
from opentrons.drivers.smoothie_drivers.driver_3_0 import DISABLE_AXES, GCODE_ROUNDING_PRECISION, PLUNGER_BACKLASH_MM, GCODES, DEFAULT_MOVEMENT_TIMEOUT, AXES, SmoothieDriver_3_0_0

from rgatkinson.util import is_close


def SmoothieDriver_3_0_0_move(self, target, home_flagged_axes=False):
    """
    Move to the `target` Smoothieware coordinate, along any of the size
    axes, XYZABC.

    target: dict
        dict setting the coordinate that Smoothieware will be at when
        `move()` returns. `target` keys are the axis in upper-case, and the
        values are the coordinate in millimeters (float)

    home_flagged_axes: boolean (default=False)
        If set to `True`, each axis included within the target coordinate
        may be homed before moving, determined by Smoothieware's internal
        homing-status flags (`True` means it has already homed). All axes'
        flags are set to `False` by Smoothieware under three conditions:
        1) Smoothieware boots or resets, 2) if a HALT gcode or signal
        is sent, or 3) a homing/limitswitch error occured.
    """
    from opentrons.drivers.smoothie_drivers.driver_3_0 import log

    self.run_flag.wait()

    def valid_movement(coords, axis):
        return not (
            (axis in DISABLE_AXES) or
            (coords is None) or
            is_close(coords, self.position[axis])
        )

    def create_coords_list(coords_dict):
        return [
            axis + str(round(coords, GCODE_ROUNDING_PRECISION))
            for axis, coords in sorted(coords_dict.items())
            if valid_movement(coords, axis)
        ]

    backlash_target = target.copy()
    backlash_target.update({
        axis: value + PLUNGER_BACKLASH_MM
        for axis, value in sorted(target.items())
        if axis in 'BC' and self.position[axis] < value
    })

    target_coords = create_coords_list(target)
    backlash_coords = create_coords_list(backlash_target)

    if target_coords:
        non_moving_axes = ''.join([
            ax
            for ax in AXES
            if ax not in target.keys()
        ])
        self.dwell_axes(non_moving_axes)
        self.activate_axes(target.keys())

        # include the current-setting gcodes within the moving gcode string
        # to reduce latency, since we're setting current so much
        command = self._generate_current_command()

        if backlash_coords != target_coords:
            command += ' ' + GCODES['MOVE'] + ''.join(backlash_coords)
        command += ' ' + GCODES['MOVE'] + ''.join(target_coords)

        try:
            for axis in target.keys():
                self.engaged_axes[axis] = True
            if home_flagged_axes:
                self.home_flagged_axes(''.join(list(target.keys())))
            log.debug("move: {}".format(command))
            # TODO (andy) a movement's timeout should be calculated by
            # how long the movement is expected to take. A default timeout
            # of 30 seconds prevents any movements that take longer
            self._send_command(command, timeout=DEFAULT_MOVEMENT_TIMEOUT)
        finally:
            # dwell pipette motors because they get hot
            plunger_axis_moved = ''.join(set('BC') & set(target.keys()))
            if plunger_axis_moved:
                self.dwell_axes(plunger_axis_moved)
                self._set_saved_current()

        self._update_position(target)


class PerfHackManager(object):
    def __init__(self):
        self.installed = False

    def install(self):
        if not self.installed:
            SmoothieDriver_3_0_0.move = SmoothieDriver_3_0_0_move
            self.installed = True

perf_hack_manager = PerfHackManager()