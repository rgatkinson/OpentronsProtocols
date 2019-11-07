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
    # parts[tube] -> <|cylindrical->cylinder[38.3037,4.16389],conical->invertedCone[3.69629,4.16389],cap->cylinder[0,0]|>
    def __init__(self, well, config):
        super().__init__(well, config)

    @property
    def radial_clearance_tolerance(self):
        return 1.5  # extra because these tubes have some slop as they sit in their rack, don't want to rattle tube todo: make retval dependent on labware

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
    # parts[tube]-><|cylindrical->cylinder[6.69498,2.61859],conical->invertedFrustum[8.11502,2.61859,1.16608],cap->cylinder[0,0]|>
    def __init__(self, well, config):
        super().__init__(well, config)

    def depth_from_volume(self, vol):
        if vol <= 0.0:
            return 0.0
        if vol <= 95.7748:
            return -6.514739207958923 + 2.7801804906856553*cubeRoot(12.86684682940816 + 1.3870479041474308*vol)
        return 3.669055001226564 + 0.04642103427328387*vol

    def volume_from_depth(self, depth):
        if depth <= 0.0:
            return 0.0
        if depth <= 8.11502:
            return depth*(4.271740774393597 + (0.6557040332750236 + 0.033549771389874604*depth)*depth)
        return -79.03863105734744 + 21.541958632651962*depth

    def radius_from_depth(self, depth):
        if depth <= 0:
            return 0
        if depth <= 8.11502:
            return 1.166077750282495 + 0.1789907029367993*depth
        return 2.6185909188980574

    @property
    def well_capacity(self):
        return 239.998

    @property
    def well_depth(self):
        return 14.81

    @property
    def well_diameter_at_top(self):
        return 5.46


class Eppendorf1point5mlTubeGeometry(WellGeometry):
    # parts[tube] -> <|cylindrical->invertedFrustum[21.1258,4.66267,4.272],conical->invertedFrustum[16.4801,4.272,1.48612],cap->invertedSphericalCap[0.194089,1.48612,rCap]|>
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
    # parts[tube] -> <|cylindrical->invertedFrustum[35.8967,7.08628,6.37479],conical->invertedFrustum[18.3424,6.37479,1.50899],cap->invertedSphericalCap[1.16088,1.50899,rCap]|>
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
    # parts[tube] -> <|cylindrical->invertedFrustum[95.7737,7.47822,6.70634],conical->invertedFrustum[22.2963,6.70634,1.14806],cap->cylinder[0,0]|>
    def __init__(self, well, config):
        super().__init__(well, config)

    def depth_from_volume(self, vol):
        if vol <= 0.0:
            return 0.0
        if vol <= 1260.65:
            return -4.605312927271903 + 1.425220154402649*cubeRoot(33.73895064080807 + 5.3077630053562075*vol)
        return -809.8165210055173 + 27.119471721476614*cubeRoot(27957.824136197134 + 0.7370907258662586*vol)

    def volume_from_depth(self, depth):
        if depth <= 0:
            return 0.0
        if depth <= 22.2963:
            return depth*(4.1407799998941535 + (0.8991310830091779 + 0.06507926078773585*depth)*depth)
        return -1806.0097363707396 + depth*(133.82273354274736 + (0.1652506834221966 + 0.00006801980413086301*depth)*depth)

    def radius_from_depth(self, depth):
        if depth <= 0:
            return 0.0
        if depth <= 22.2963:
            return 1.1480641142716852 + 0.2492912278496944*depth
        return 6.526645316147934 + 0.008059412406212692*depth

    @property
    def well_capacity(self):
        return 16410.1  # compare to 15000 in opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical

    @property
    def outside_height(self):
        return 119.46

    @property
    def well_depth(self):
        return 118.07  # compare to 117.5 in opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical

    @property
    def well_diameter_at_top(self):
        return 14.859

    @property
    def rim_lip_height(self):
        return 7.28


class FalconTube50mlGeometry(WellGeometry):
    # parts[tube] -> <|cylindrical->invertedFrustum[99.4458,13.6982,13.1264],conical->invertedFrustum[13.2242,13.1264,3.86673],cap->cylinder[0,0]|>
    def __init__(self, well, config):
        super().__init__(well, config)

    def depth_from_volume(self, vol):
        if vol <= 0.0:
            return 0.0
        if vol <= 3296.08:
            return -5.522264395071952 + 0.6039249881108911*cubeRoot(764.5441851977812 + 8.842372775534407*vol)
        return -2269.6881765411304 + 37.538777353484434*cubeRoot(223119.88753911393 + 0.5460286683567588*vol)

    def volume_from_depth(self, depth):
        if depth <= 0:
            return 0.0
        if depth <= 13.2242:
            return depth*(46.97186764441949 + (8.505907048988277 + 0.5134311120983222*depth)*depth)
        return -3820.9148917040493 + depth*(535.0542643832791 + (0.23573910721016753 + 0.00003462136482692524*depth)*depth)

    def radius_from_depth(self, depth):
        if depth <= 0:
            return 0.0
        if depth <= 13.2242:
            return 3.8667311574164636 + 0.7002075382097096*depth
        return 13.050404667978436 + 0.0057498667891316*depth

    @property
    def well_capacity(self):
        return 59505.8

    @property
    def outside_height(self):
        return 114.55

    @property
    def well_depth(self):
        return 112.67

    @property
    def well_diameter_at_top(self):
        return 27.86

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
