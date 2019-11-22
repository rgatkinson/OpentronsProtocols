#
# performance hacks
#
import numpy as np
from numpy.linalg import inv
from opentrons.drivers.smoothie_drivers.driver_3_0 import DISABLE_AXES, GCODE_ROUNDING_PRECISION, PLUNGER_BACKLASH_MM, GCODES, DEFAULT_MOVEMENT_TIMEOUT, AXES, SmoothieDriver_3_0_0
from opentrons.trackers.pose_tracker import Point, translate, change_base

from rgatkinson.util import is_close

def transform_dot_inv_translate(point: Point, transform=None):
    if transform is None:
        x, y, z, = point
        result = np.array([
            [1.0, 0.0, 0.0,  -x],
            [0.0, 1.0, 0.0,  -y],
            [0.0, 0.0, 1.0,  -z],
            [0.0, 0.0, 0.0, 1.0]
        ])
    else:
        # numpyupdated = transform.dot(inv(translate(point)))
        x, y, z, = point
        result = np.array([
            [transform[0][0], transform[0][1], transform[0][2], transform[0][3] - transform[0][0] * x - transform[0][1] * y - transform[0][2] * z],
            [transform[1][0], transform[1][1], transform[1][2], transform[1][3] - transform[1][0] * x - transform[1][1] * y - transform[1][2] * z],
            [transform[2][0], transform[2][1], transform[2][2], transform[2][3] - transform[2][0] * x - transform[2][1] * y - transform[2][2] * z],
            [transform[3][0], transform[3][1], transform[3][2], transform[3][3] - transform[3][0] * x - transform[3][1] * y - transform[3][2] * z],
        ])
        # assert result == numpyupdated
    return result

def pose_tracker_update(state, obj, point: Point, transform=None):
    state = state.copy()
    state[obj] = state[obj].update(transform_dot_inv_translate(point, transform))
    return state

def mover_move(self, pose_tree, x=None, y=None, z=None, home_flagged_axes=True):
    """
    Dispatch move command to the driver changing base of
    x, y and z from source coordinate system to destination.

    Value must be set for each axis that is mapped.

    home_flagged_axes: (default=True)
        This kwarg is passed to the driver. This ensures that any axes
        within this Mover's axis_mapping is homed before moving, if it has
        not yet done so. See driver docstring for details
    """
    def defaults(_x, _y, _z):
        _x = _x if x is not None else 0
        _y = _y if y is not None else 0
        _z = _z if z is not None else 0
        return _x, _y, _z

    dst_x, dst_y, dst_z = change_base(
        pose_tree,
        src=self._src,
        dst=self._dst,
        point=Point(*defaults(x, y, z)))
    driver_target = {}

    if 'x' in self._axis_mapping:
        assert x is not None, "Value must be set for each axis mapped"
        driver_target[self._axis_mapping['x']] = dst_x

    if 'y' in self._axis_mapping:
        assert y is not None, "Value must be set for each axis mapped"
        driver_target[self._axis_mapping['y']] = dst_y

    if 'z' in self._axis_mapping:
        assert z is not None, "Value must be set for each axis mapped"
        driver_target[self._axis_mapping['z']] = dst_z
    self._driver.move(driver_target, home_flagged_axes=home_flagged_axes)

    # Update pose with the new value. Since stepper motors are open loop
    # there is no need to to query diver for position
    return pose_tracker_update(pose_tree, self, Point(*defaults(dst_x, dst_y, dst_z)))


def mover_update_pose_from_driver(self, pose_tree):
    from opentrons.legacy_api.robot.mover import log
    # map from driver axis names to xyz and expand position
    # into point object
    point = Point(
        x=self._driver.position.get(self._axis_mapping.get('x', ''), 0.0),
        y=self._driver.position.get(self._axis_mapping.get('y', ''), 0.0),
        z=self._driver.position.get(self._axis_mapping.get('z', ''), 0.0)
    )
    log.debug(f'Point in update pose from driver {point}')
    return pose_tracker_update(pose_tree, self, point)

#-----------------------------------------------------------------------------------------------------------------------

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
            #
            SmoothieDriver_3_0_0.move = SmoothieDriver_3_0_0_move
            #
            from opentrons.trackers import pose_tracker
            pose_tracker.update = pose_tracker_update
            #
            from opentrons.legacy_api.robot import mover
            mover.Mover.move = mover_move
            mover.Mover.update_pose_from_driver = mover_update_pose_from_driver
            #
            self.installed = True

perf_hack_manager = PerfHackManager()