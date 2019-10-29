#
# well.py
#

from abc import abstractmethod

from opentrons import robot
from opentrons.legacy_api.containers import Well
from opentrons.trackers import pose_tracker
from opentrons.util.vector import Vector

from rgatkinson.interval import fpu, is_interval, Interval
from rgatkinson.util import sqrt, square, cube, cubeRoot


def is_well(location):
    return isinstance(location, Well)


class WellGeometry(object):

    #-------------------------------------------------------------------------------------------------------------------
    # Construction
    #-------------------------------------------------------------------------------------------------------------------

    def __init__(self, well, config):
        self.__well = None
        self.well = well
        self.config = config

    @property
    def well(self):
        return self.__well

    @well.setter
    def well(self, value):
        if self.__well is not None:
            self.__well.geometry = None
        self.__well = value
        if self.__well is not None:
            self.__well.geometry = self

    #-------------------------------------------------------------------------------------------------------------------
    # Accessing
    #-------------------------------------------------------------------------------------------------------------------

    @property
    def well_capacity(self):  # default to what Opentrons gave us
        result = self.well.max_volume() if self.well is not None else None
        if result is None:
            result = fpu.infinity
        return result

    @property
    def well_depth(self):  # default to what Opentrons gave us
        return self.well.z_size() if self.well is not None else fpu.infinity

    @property
    def outside_height(self):  # outside height of the tube
        return 0

    @property
    def rim_lip_height(self):  # when hanging in a rack, this is how much the tube sits about the reference plane of the rack
        return 0

    def height_above_reference_plane(self, hangable_tube_height, rack):
        return max(0, self.outside_height - hangable_tube_height, self.rim_lip_height);

    @property
    def well_diameter_at_top(self):
        return self.radius_from_depth(self.well_depth) * 2  # a generic impl; subclasses can optimize

    @property
    def min_aspiratable_volume(self):  # minimum volume we can aspirate from (i.e.: we leave at least this much behind when aspirating)
        return 0

    @property
    def radial_clearance_tolerance(self):
        return self.config.wells.radial_clearance_tolerance

    @abstractmethod
    def depth_from_volume(self, volume):  # best calc'n of depth from the given volume. may be an interval
        pass

    @abstractmethod
    def volume_from_depth(self, depth):
        pass

    @abstractmethod
    def radius_from_depth(self, depth):
        pass

    #-------------------------------------------------------------------------------------------------------------------
    # Calculations
    #-------------------------------------------------------------------------------------------------------------------

    def depth_from_volume_min(self, volume):  # lowest possible depth for the given volume
        vol = self.depth_from_volume(volume)
        if is_interval(vol):
            return vol.infimum
        else:
            return vol


class UnknownWellGeometry(WellGeometry):
    def __init__(self, well, config):
        super().__init__(well, config)

    def depth_from_volume(self, vol):
        return Interval([0, self.well_depth])

    def volume_from_depth(self, depth):
        return Interval([0, self.well_capacity])

    def radius_from_depth(self, depth):
        return self.well.properties['diameter'] / 2 if self.well is not None else fpu.infinity


class IdtTubeWellGeometry(WellGeometry):
    def __init__(self, well, config):
        super().__init__(well, config)

    @property
    def radial_clearance_tolerance(self):
        return 1.5  # extra because these tubes have some slop in their labware, don't want to rattle tube todo: make retval dependent on labware

    def depth_from_volume(self, vol):
        if vol <= 0.0:
            return 0.0
        if vol <= 67.1109:
            return 0.9095678851543723*cubeRoot(vol)
        return 2.464193794602757 + 0.018359120058446303*vol

    def volume_from_depth(self, depth):
        if depth <= 0:
            return 0
        if depth <= 3.69629:
            return 1.3289071745212766*cube(depth)
        return -134.221781150621 + 54.46884147042437*depth

    def radius_from_depth(self, depth):
        if depth <= 0:
            return 0
        if depth <= 3.69629:
            return 1.126504715663486*depth
        return 4.163888894893057

    @property
    def well_capacity(self):
        return 2153.47

    @property
    def well_depth(self):
        return 42

    @property
    def well_diameter_at_top(self):
        return 8.32778

    @property
    def min_aspiratable_volume(self):
        return 75  # a rough estimate, but seems functionally useful

    @property
    def rim_lip_height(self):
        raise 6.38

    @property
    def outside_height(self):
        return 45.01


class Biorad96WellPlateWellGeometry(WellGeometry):
    def __init__(self, well, config):
        super().__init__(well, config)

    def depth_from_volume(self, vol):
        if vol <= 0.0:
            return 0.0
        if vol <= 122.784:
            return -8.576177710037857 + 3.105085210707993*cubeRoot(21.069816179177707 + 1.3464508185574342*vol)
        return 3.9168885170626426 + 0.04270953403155694*vol

    def volume_from_depth(self, depth):
        if depth <= 0.0:
            return 0.0
        if depth <= 9.16092:
            return depth*(5.473911039614858 + (0.6382693111883633 + 0.024807839139547*depth)*depth)
        return -91.7099332942513 + 23.41397588793937*depth

    def radius_from_depth(self, depth):
        if depth <= 0:
            return 0
        if depth <= 9.16092:
            return 1.32 + 0.15391472105982892*depth
        return 2.73

    @property
    def well_capacity(self):
        return 255.051

    @property
    def well_depth(self):
        return 14.81

    @property
    def well_diameter_at_top(self):
        return 5.46


class Eppendorf1point5mlTubeGeometry(WellGeometry):
    def __init__(self, well, config):
        super().__init__(well, config)

    def depth_from_volume(self, vol):
        if vol <= 0:
            return 0
        if vol <= 0.67718:
            return 3.2346271740790455/cubeRoot(-3*vol + sqrt(106.32185983362221 + 9*square(vol))) - 0.6827840632552957*cubeRoot(-3*vol + sqrt(106.32185983362221 + 9*square(vol)))
        if vol <= 463.316:
            return -8.597168410942386 + 2.324577069455727*cubeRoot(52.28910925575565 + 2.660323800283652*vol)
        return -214.3418544824842 + 19.561686679619903*cubeRoot(1474.2109760964095 + 0.37305557584692783*vol)

    def volume_from_depth(self, depth):
        if depth <= 0:
            return 0
        if depth <= 0.194095:
            return 3.4691851885468576*depth + 0.5235987755982988*cube(depth)
        if depth <= 16.6742:
            return -0.6400085458081618 + depth*(6.6353796285770015 + (0.7718098926771707 + 0.029924965049919594*depth)*depth)
        return -425.3442361254281 + depth*(49.356322291514765 + (0.23026917636167102 + 0.00035810267810680666*depth)*depth)

    def radius_from_depth(self, depth):
        if depth <= 0:
            return 0
        if depth <= 0.194095:
            return 2.2698281410529737*sqrt((2.2462247020231834 - 0.19409486595347666*depth)*depth)
        if depth <= 16.6742:
            return 1.4533089603930036 + 0.16904507285715673*depth
        return 3.963660597359791 + 0.018492238050892146*depth

    @property
    def well_capacity(self):
        return 1788.68

    @property
    def well_depth(self):
        return 37.8

    @property
    def outside_height(self):
        return 38.9

    @property
    def well_diameter_at_top(self):
        return 9.32533

    @property
    def rim_lip_height(self):
        return 2


class Eppendorf5point0mlTubeGeometry(WellGeometry):
    def __init__(self, well, config):
        super().__init__(well, config)

    def depth_from_volume(self, vol):
        if vol <= 0:
            return 0
        if vol <= 4.97137:
            return 3.33494414484718/cubeRoot(-3*vol + sqrt(116.52404878202921 + 9*square(vol))) - 0.6827840632552957*cubeRoot(-3*vol + sqrt(116.52404878202921 + 9*square(vol)))
        if vol <= 1014.06:
            return -4.527482527717488 + 1.4293857409730928*cubeRoot(39.92570761834668 + 4.646502581189681*vol)
        return -302.1252531106694 + 15.294554814291097*cubeRoot(8610.391237925141 + 0.6794188941067961*vol)

    def volume_from_depth(self, depth):
        if depth <= 0:
            return 0
        if depth <= 1.16088:
            return 3.576776614210246*depth + 0.5235987755982988*cube(depth)
        if depth <= 19.5033:
            return -1.7535856833793924 + depth*(4.531691130053943 + (1.000929567880315 + 0.07369287175618172*depth)*depth)
        return -1327.949641310943 + depth*(112.65414304343028 + (0.3728723166420145 + 0.0004113882270161532*depth)*depth)

    def radius_from_depth(self, depth):
        if depth <= 0:
            return 0
        if depth <= 1.16088:
            return 0.9281232870726978*sqrt((3.624697354653752 - 1.1608835603563077*depth)*depth)
        if depth <= 19.5033:
            return 1.2010337579883252 + 0.26527628779029744*depth
        return 5.9882324145182295 + 0.01982036374935098*depth

    @property
    def well_capacity(self):
        return 6127.44

    @property
    def well_depth(self):
        return 55.4

    @property
    def outside_height(self):
        return 56.7

    @property
    def well_diameter_at_top(self):
        return 14.1726

    @property
    def rim_lip_height(self):
        return 2.2


class FalconTube15mlGeometry(WellGeometry):
    def __init__(self, well, config):
        super().__init__(well, config)

    def depth_from_volume(self, vol):
        if vol <= 0.0:
            return 0.0
        if vol <= 1232.34:
            return -4.605312927271903 + 1.4295474713971166*cubeRoot(33.43348831212188 + 5.259708112808352*vol)
        return -803.7743858256094 + 27.100445027181177*cubeRoot(27390.881699748476 + 0.7386443071956942*vol)

    def volume_from_depth(self, depth):
        if depth <= 0:
            return 0.0
        if depth <= 22.0945:
            return depth*(4.1407799998941535 + (0.8991310830091779 + 0.06507926078773585*depth)*depth)
        return -1761.2447144832822 + depth*(131.83324928521762 + (0.1640177288677879 + 0.00006801980413086297*depth)*depth)

    def radius_from_depth(self, depth):
        if depth <= 0:
            return 0.0
        if depth <= 22.0945:
            return 1.1480641142716852 + 0.2492912278496944*depth
        return 6.477949256918969 + 0.008059412406212692*depth

    @property
    def well_capacity(self):
        return 16202.8  # compare to 15000 in opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical

    @property
    def outside_height(self):
        return 119.40

    @property
    def well_depth(self):
        return 118.07  # compare to 117.5 in opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical

    @property
    def well_diameter_at_top(self):
        return 14.859

    @property
    def rim_lip_height(self):
        return 7.28


class FalconTube50mlGeometry(WellGeometry):  # not yet finished
    def __init__(self, well, config):
        super().__init__(well, config)

    def depth_from_volume(self, volume):
        pass

    def volume_from_depth(self, depth):
        pass

    def radius_from_depth(self, depth):
        pass

    @property
    def well_capacity(self):
        return 50000  # nominal

    @property
    def outside_height(self):
        return 114.11

    @property
    def well_depth(self):
        return 113.5

    @property
    def well_diameter_at_top(self):
        return 27.74

    @property
    def rim_lip_height(self):
        return 10.26


def Well_get_name(self):
    result = super(Well, self).get_name()
    label = getattr(self, 'label', None)
    if label is not None:
        result += ' (' + label + ')'
    return result


def Well_top_coords_absolute(self):
    xyz = pose_tracker.absolute(robot.poses, self)
    return Vector(xyz)

# Enhance well name to include any label that might be present
Well.has_labelled_well_name = True
Well.get_name = Well_get_name
Well.top_coords_absolute = Well_top_coords_absolute
