#
# performance hacks
#
import functools
import itertools
import numpy as np
from numpy.linalg import inv
from opentrons.trackers.pose_tracker import Point, translate, change_base, ascend, ROOT

from rgatkinson.util import is_close, thread_local_storage


#-----------------------------------------------------------------------------------------------------------------------
# Pose tracking
#-----------------------------------------------------------------------------------------------------------------------

def transform_dot_inv_translate(point: Point, transform=None):
    if transform is None:  # so treat as identity matrix
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
            [transform[0, 0], transform[0, 1], transform[0, 2], transform[0, 3] - transform[0, 0] * x - transform[0, 1] * y - transform[0, 2] * z],
            [transform[1, 0], transform[1, 1], transform[1, 2], transform[1, 3] - transform[1, 0] * x - transform[1, 1] * y - transform[1, 2] * z],
            [transform[2, 0], transform[2, 1], transform[2, 2], transform[2, 3] - transform[2, 0] * x - transform[2, 1] * y - transform[2, 2] * z],
            [transform[3, 0], transform[3, 1], transform[3, 2], transform[3, 3] - transform[3, 0] * x - transform[3, 1] * y - transform[3, 2] * z],
        ])
        # assert result == numpyupdated
    return result

def pose_tracker_update(state, obj, point: Point, transform=None):
    if not thread_local_storage.update_pose_tree_in_place:
        state = state.copy()
    state[obj] = state[obj].update(transform_dot_inv_translate(point, transform))
    return state

def inv4x4(m):
    m00 = m[0, 0]; m01 = m[0, 1]; m02 = m[0, 2]; m03 = m[0, 3]
    m10 = m[1, 0]; m11 = m[1, 1]; m12 = m[1, 2]; m13 = m[1, 3]
    m20 = m[2, 0]; m21 = m[2, 1]; m22 = m[2, 2]; m23 = m[2, 3]
    m30 = m[3, 0]; m31 = m[3, 1]; m32 = m[3, 2]; m33 = m[3, 3]

    d = m03*m12*m21*m30 - m02*m13*m21*m30 - m03*m11*m22*m30 + m01*m13*m22*m30 + m02*m11*m23*m30 - m01*m12*m23*m30 - m03*m12*m20*m31 + \
        m02*m13*m20*m31 + m03*m10*m22*m31 - m00*m13*m22*m31 - m02*m10*m23*m31 + m00*m12*m23*m31 + m03*m11*m20*m32 - m01*m13*m20*m32 - \
        m03*m10*m21*m32 + m00*m13*m21*m32 + m01*m10*m23*m32 - m00*m11*m23*m32 - m02*m11*m20*m33 + m01*m12*m20*m33 + m02*m10*m21*m33 - \
        m00*m12*m21*m33 - m01*m10*m22*m33 + m00*m11*m22*m33

    result = [
        [(-(m13*m22*m31) + m12*m23*m31 + m13*m21*m32 - m11*m23*m32 - m12*m21*m33 + m11*m22*m33)/d,
            (m03*m22*m31 - m02*m23*m31 - m03*m21*m32 + m01*m23*m32 + m02*m21*m33 - m01*m22*m33)/d,
            (-(m03*m12*m31) + m02*m13*m31 + m03*m11*m32 - m01*m13*m32 - m02*m11*m33 + m01*m12*m33)/d,
            (m03*m12*m21 - m02*m13*m21 - m03*m11*m22 + m01*m13*m22 + m02*m11*m23 - m01*m12*m23)/d],
        [(m13*m22*m30 - m12*m23*m30 - m13*m20*m32 + m10*m23*m32 + m12*m20*m33 - m10*m22*m33)/d,
            (-(m03*m22*m30) + m02*m23*m30 + m03*m20*m32 - m00*m23*m32 - m02*m20*m33 + m00*m22*m33)/d,
            (m03*m12*m30 - m02*m13*m30 - m03*m10*m32 + m00*m13*m32 + m02*m10*m33 - m00*m12*m33)/d,
            (-(m03*m12*m20) + m02*m13*m20 + m03*m10*m22 - m00*m13*m22 - m02*m10*m23 + m00*m12*m23)/d],
        [(-(m13*m21*m30) + m11*m23*m30 + m13*m20*m31 - m10*m23*m31 - m11*m20*m33 + m10*m21*m33)/d,
            (m03*m21*m30 - m01*m23*m30 - m03*m20*m31 + m00*m23*m31 + m01*m20*m33 - m00*m21*m33)/d,
            (-(m03*m11*m30) + m01*m13*m30 + m03*m10*m31 - m00*m13*m31 - m01*m10*m33 + m00*m11*m33)/d,
            (m03*m11*m20 - m01*m13*m20 - m03*m10*m21 + m00*m13*m21 + m01*m10*m23 - m00*m11*m23)/d],
        [(m12*m21*m30 - m11*m22*m30 - m12*m20*m31 + m10*m22*m31 + m11*m20*m32 - m10*m21*m32)/d,
            (-(m02*m21*m30) + m01*m22*m30 + m02*m20*m31 - m00*m22*m31 - m01*m20*m32 + m00*m21*m32)/d,
            (m02*m11*m30 - m01*m12*m30 - m02*m10*m31 + m00*m12*m31 + m01*m10*m32 - m00*m11*m32)/d,
         (-(m02*m11*m20) + m01*m12*m20 + m02*m10*m21 - m00*m12*m21 - m01*m10*m22 + m00*m11*m22)/d]
    ]
    return np.array(result, copy=False)

def pose_tracker_change_base(state, point=Point(0, 0, 0), src=ROOT, dst=ROOT):
    """
    Transforms point from source coordinate system to destination.
    Point(0, 0, 0) means the origin of the source.
    """
    def fold(objects):
        return functools.reduce(
            lambda a, b: a.dot(b),
            [state[key].transform for key in objects],
            np.identity(4)
        )

    up, down = ascend(state, src), list(reversed(ascend(state, dst)))

    # Find common prefix. Last item is common root
    root = [n1 for n1, n2 in zip(reversed(up), down) if n1 is n2].pop()

    # Nodes up to root, EXCLUDING root
    up = list(itertools.takewhile(lambda node: node is not root, up))

    # Nodes down from root, EXCLUDING root
    down = list(itertools.dropwhile(lambda node: node is not root, down))[1:]

    # Point in root's coordinate system
    if src is ROOT:
        # folded = fold(up)  # will be identity matrix; thus so will inverse
        inv_folded = np.identity(4)
    else:
        inv_folded = inv4x4(fold(up))
    point_in_root = inv_folded.dot((*point, 1))

    # Return point in destination's coordinate system
    return fold(down).dot(point_in_root)[:-1]

#-----------------------------------------------------------------------------------------------------------------------
# Movement
#-----------------------------------------------------------------------------------------------------------------------

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

    dst_x, dst_y, dst_z = pose_tracker_change_base(
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
# Installation
#-----------------------------------------------------------------------------------------------------------------------

class PerfHackManager(object):
    def __init__(self):
        self.installed = False

    def install(self):
        if not self.installed:
            #
            from opentrons.trackers import pose_tracker
            pose_tracker.update = pose_tracker_update
            pose_tracker.change_base = pose_tracker_change_base
            #
            from opentrons.legacy_api.robot import mover
            mover.Mover.move = mover_move
            mover.Mover.update_pose_from_driver = mover_update_pose_from_driver
            #
            from rgatkinson.pipette_v1 import EnhancedPipetteV1
            thread_local_storage.update_pose_tree_in_place = False
            EnhancedPipetteV1.install_perf_hacks()
            #
            self.installed = True

perf_hack_manager = PerfHackManager()