#
# well.py
#

from abc import abstractmethod
from typing import Union

from opentrons import robot
from opentrons.legacy_api.containers import Well as WellV1, Container, Slot, location_to_list
from opentrons.trackers import pose_tracker
from opentrons.types import Location
from opentrons.util.vector import Vector
from opentrons.protocol_api.labware import Well as WellV2
from opentrons.protocols.types import APIVersion

from rgatkinson.configuration import WellGeometryConfigurationContext
from rgatkinson.interval import Interval, infimum
from rgatkinson.liquid import LiquidVolume
from rgatkinson.util import sqrt, square, cube, cubeRoot, instance_count, thread_local_storage, infinity

#-----------------------------------------------------------------------------------------------------------------------
# Utility
#-----------------------------------------------------------------------------------------------------------------------

EnhancedWellType = Union['EnhancedWell']

def is_well_v1(location):
    return isinstance(location, WellV1)

#-----------------------------------------------------------------------------------------------------------------------
# Geometries
#-----------------------------------------------------------------------------------------------------------------------

# region Well Geometries

class WellGeometry(object):

    #-------------------------------------------------------------------------------------------------------------------
    # Construction
    #-------------------------------------------------------------------------------------------------------------------

    def __init__(self, well: EnhancedWellType = None):
        self.__well: EnhancedWellType = None
        self.well = well

    @property
    def well(self) -> EnhancedWellType:
        return self.__well

    @well.setter
    def well(self, value: EnhancedWellType):
        if self.__well is not value:
            old_well = self.__well
            self.__well = None
            if old_well is not None:
                old_well.geometry = None

            self.__well = value
            if self.__well is not None:
                self.__well.geometry = self

    @property
    def config(self) -> WellGeometryConfigurationContext:
        if self.well is not None:
            return self.well.config
        return thread_local_storage.config.wells

    #-------------------------------------------------------------------------------------------------------------------
    # Accessing
    #-------------------------------------------------------------------------------------------------------------------

    @property
    def shape(self):
        return 'circular'

    @property
    def well_capacity(self):
        """
        How much can the well hold, in microliters? Default here to what Opentrons provides.
        """
        result = self.well.max_volume() if self.well is not None else None
        if result is None:
            result = infinity
        return result

    @property
    def well_depth(self):
        """
        How deep is the interior of the well, in mm? Default here to what Opentrons provides.
        """
        return self.well.well_depth if self.well is not None else infinity

    @property
    def well_diameter_at_top(self):
        """
        what is the diameter of this well at it's top
        todo: what's the reasonable behavior for non-circular wells?
        """
        return self.radius_from_liquid_depth(self.well_depth) * 2  # a generic impl; subclasses can optimize

    @property
    def min_aspiratable_volume(self):
        """
        minimum volume we can aspirate from (i.e.: we leave at least this much behind when aspirating)
        """
        return 0

    @property
    def radial_clearance_tolerance(self):
        return self.config.radial_clearance_tolerance

    @abstractmethod
    def liquid_depth_from_volume(self, volume):
        """
        best calc'n of depth from the given volume. may be an interval
        """
        pass

    def liquid_depth_from_volume_min(self, volume):  # lowest possible depth for the given volume
        vol = self.liquid_depth_from_volume(volume)
        return infimum(vol)

    @abstractmethod
    def volume_from_liquid_depth(self, depth):
        pass

    @abstractmethod
    def radius_from_liquid_depth(self, depth):
        pass

    #-------------------------------------------------------------------------------------------------------------------
    # Stuff about the *external* view of the well
    #-------------------------------------------------------------------------------------------------------------------

    @property
    def is_tube(self):
        return True

    @property
    def external_tube_height(self):
        """
        External height of the tube, in mm
        """
        assert not self.is_tube  # tube subclasses should override
        return 0

    @property
    def rim_lip_height(self):
        """
        when hanging in a rack, this is how much the tube sits about the reference plane of the rack
        """
        assert not self.is_tube  # tube subclasses should override
        return 0

    def height_above_reference_plane(self, space_below_reference_plane, rack):
        """
        how high does this well extend about the reference plane of this rack
        :param space_below_reference_plane: how much space (at least) is available below the reference plane of this rack
        :param rack: the rack in question
        """
        if self.is_tube:
            return max(0, self.external_tube_height - space_below_reference_plane, self.rim_lip_height)
        else:
            return 0

    def _is_rack(self, rack, custom_tube_rack_classes, load_names):
        load_names = [load_name.lower() for load_name in load_names]
        if rack is None:
            if self.well is not None:
                rack = self.well.parent
        if rack is not None:
            if rack.__class__ in custom_tube_rack_classes:
                # custom tube rack, before loading
                return True
            if isinstance(rack, Container):
                if rack.properties['type'] in load_names:
                    return True
                custom = rack.properties.get('custom_tube_rack', None)
                if custom is not None and custom.__class__ in custom_tube_rack_classes:
                    # custom tube rack, after loading
                    return True
        return False

#-----------------------------------------------------------------------------------------------------------------------

class UnknownWellGeometry(WellGeometry):
    def __init__(self, well=None):
        super().__init__(well)

    @property
    def is_tube(self):
        return False

    def liquid_depth_from_volume(self, vol):
        return Interval([0, self.well_depth])

    def volume_from_liquid_depth(self, depth):
        return Interval([0, self.well_capacity])

    def radius_from_liquid_depth(self, depth):
        return self.well.properties['diameter'] / 2 if self.well is not None else infinity


class IdtTubeWellGeometry(WellGeometry):
    # parts[tube] -> <|cylindrical->cylinder[38.3037,4.16389],conical->invertedCone[3.69629,4.16389],cap->cylinder[0,0]|>
    def __init__(self, well=None):
        super().__init__(well)

    @property
    def radial_clearance_tolerance(self):
        return 1.5  # extra because these tubes have some slop as they sit in their rack, don't want to rattle tube todo: make retval dependent on labware

    def liquid_depth_from_volume(self, vol):
        if vol <= 0.0:
            return 0.0
        if vol <= 67.1109:
            return 0.9095678851543723*cubeRoot(vol)
        return 2.464193794602757 + 0.018359120058446303*vol

    def volume_from_liquid_depth(self, depth):
        if depth <= 0:
            return 0
        if depth <= 3.69629:
            return 1.3289071745212766*cube(depth)
        return -134.221781150621 + 54.46884147042437*depth

    def radius_from_liquid_depth(self, depth):
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
        return 6.38

    @property
    def external_tube_height(self):
        return 45.01


class Biorad96WellPlateWellGeometry(WellGeometry):
    # parts[tube]-><|cylindrical->cylinder[6.69498,2.61859],conical->invertedFrustum[8.11502,2.61859,1.16608],cap->cylinder[0,0]|>
    def __init__(self, well=None):
        super().__init__(well)

    def liquid_depth_from_volume(self, vol):
        if vol <= 0.0:
            return 0.0
        if vol <= 95.7748:
            return -6.514739207958923 + 2.7801804906856553*cubeRoot(12.86684682940816 + 1.3870479041474308*vol)
        return 3.669055001226564 + 0.04642103427328387*vol

    def volume_from_liquid_depth(self, depth):
        if depth <= 0.0:
            return 0.0
        if depth <= 8.11502:
            return depth*(4.271740774393597 + (0.6557040332750236 + 0.033549771389874604*depth)*depth)
        return -79.03863105734744 + 21.541958632651962*depth

    def radius_from_liquid_depth(self, depth):
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

    @property
    def is_tube(self):
        return False


class Eppendorf1Point5MlTubeGeometry(WellGeometry):
    # parts[tube] -> <|cylindrical->invertedFrustum[21.1258,4.66267,4.272],conical->invertedFrustum[16.4801,4.272,1.48612],cap->invertedSphericalCap[0.194089,1.48612,rCap]|>
    def __init__(self, well=None):
        super().__init__(well)

    def liquid_depth_from_volume(self, vol):
        if vol <= 0:
            return 0
        if vol <= 0.67718:
            return 3.2346271740790455/cubeRoot(-3*vol + sqrt(106.32185983362221 + 9*square(vol))) - 0.6827840632552957*cubeRoot(-3*vol + sqrt(106.32185983362221 + 9*square(vol)))
        if vol <= 463.316:
            return -8.597168410942386 + 2.324577069455727*cubeRoot(52.28910925575565 + 2.660323800283652*vol)
        return -214.3418544824842 + 19.561686679619903*cubeRoot(1474.2109760964095 + 0.37305557584692783*vol)

    def volume_from_liquid_depth(self, depth):
        if depth <= 0:
            return 0
        if depth <= 0.194095:
            return 3.4691851885468576*depth + 0.5235987755982988*cube(depth)
        if depth <= 16.6742:
            return -0.6400085458081618 + depth*(6.6353796285770015 + (0.7718098926771707 + 0.029924965049919594*depth)*depth)
        return -425.3442361254281 + depth*(49.356322291514765 + (0.23026917636167102 + 0.00035810267810680666*depth)*depth)

    def radius_from_liquid_depth(self, depth):
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
    def external_tube_height(self):
        return 38.9

    @property
    def well_diameter_at_top(self):
        return 9.32533

    @property
    def rim_lip_height(self):
        return 2


class Eppendorf5Point0MlTubeGeometry(WellGeometry):
    # parts[tube] -> <|cylindrical->invertedFrustum[35.8967,7.08628,6.37479],conical->invertedFrustum[18.3424,6.37479,1.50899],cap->invertedSphericalCap[1.16088,1.50899,rCap]|>
    def __init__(self, well=None):
        super().__init__(well)

    def liquid_depth_from_volume(self, vol):
        if vol <= 0:
            return 0
        if vol <= 4.97137:
            return 3.33494414484718/cubeRoot(-3*vol + sqrt(116.52404878202921 + 9*square(vol))) - 0.6827840632552957*cubeRoot(-3*vol + sqrt(116.52404878202921 + 9*square(vol)))
        if vol <= 1014.06:
            return -4.527482527717488 + 1.4293857409730928*cubeRoot(39.92570761834668 + 4.646502581189681*vol)
        return -302.1252531106694 + 15.294554814291097*cubeRoot(8610.391237925141 + 0.6794188941067961*vol)

    def volume_from_liquid_depth(self, depth):
        if depth <= 0:
            return 0
        if depth <= 1.16088:
            return 3.576776614210246*depth + 0.5235987755982988*cube(depth)
        if depth <= 19.5033:
            return -1.7535856833793924 + depth*(4.531691130053943 + (1.000929567880315 + 0.07369287175618172*depth)*depth)
        return -1327.949641310943 + depth*(112.65414304343028 + (0.3728723166420145 + 0.0004113882270161532*depth)*depth)

    def radius_from_liquid_depth(self, depth):
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
    def external_tube_height(self):
        return 56.7

    @property
    def well_diameter_at_top(self):
        return 14.1726

    @property
    def rim_lip_height(self):
        return 2.2


class FalconTube15MlGeometry(WellGeometry):
    # parts[tube] -> <|cylindrical->invertedFrustum[95.7737,7.47822,6.70634],conical->invertedFrustum[22.2963,6.70634,1.14806],cap->cylinder[0,0]|>
    def __init__(self, well=None):
        super().__init__(well)

    def liquid_depth_from_volume(self, vol):
        if vol <= 0.0:
            return 0.0
        if vol <= 1260.65:
            return -4.605312927271903 + 1.425220154402649*cubeRoot(33.73895064080807 + 5.3077630053562075*vol)
        return -809.8165210055173 + 27.119471721476614*cubeRoot(27957.824136197134 + 0.7370907258662586*vol)

    def volume_from_liquid_depth(self, depth):
        if depth <= 0:
            return 0.0
        if depth <= 22.2963:
            return depth*(4.1407799998941535 + (0.8991310830091779 + 0.06507926078773585*depth)*depth)
        return -1806.0097363707396 + depth*(133.82273354274736 + (0.1652506834221966 + 0.00006801980413086301*depth)*depth)

    def radius_from_liquid_depth(self, depth):
        if depth <= 0:
            return 0.0
        if depth <= 22.2963:
            return 1.1480641142716852 + 0.2492912278496944*depth
        return 6.526645316147934 + 0.008059412406212692*depth

    @property
    def well_capacity(self):
        return 16410.1  # compare to 15000 in opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical

    @property
    def external_tube_height(self):
        return 119.46

    def height_above_reference_plane(self, space_below_reference_plane, rack):
        from rgatkinson.custom_labware import Opentrons10RackV1, Opentrons15RackV1
        if self._is_rack(rack, [Opentrons15RackV1, Opentrons10RackV1], ['opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical']):
            return 45.65  # a measured value that accounts for the portion of the tube lying in the dimple at the bottom
        return super().height_above_reference_plane(space_below_reference_plane, rack)

    @property
    def well_depth(self):
        return 118.07  # compare to 117.5 in opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical

    @property
    def well_diameter_at_top(self):
        return 14.859

    @property
    def rim_lip_height(self):
        return 7.28


class FalconTube50MlGeometry(WellGeometry):
    # parts[tube] -> <|cylindrical->invertedFrustum[99.4458,13.6982,13.1264],conical->invertedFrustum[13.2242,13.1264,3.86673],cap->cylinder[0,0]|>
    def __init__(self, well=None):
        super().__init__(well)

    def liquid_depth_from_volume(self, vol):
        if vol <= 0.0:
            return 0.0
        if vol <= 3296.08:
            return -5.522264395071952 + 0.6039249881108911*cubeRoot(764.5441851977812 + 8.842372775534407*vol)
        return -2269.6881765411304 + 37.538777353484434*cubeRoot(223119.88753911393 + 0.5460286683567588*vol)

    def volume_from_liquid_depth(self, depth):
        if depth <= 0:
            return 0.0
        if depth <= 13.2242:
            return depth*(46.97186764441949 + (8.505907048988277 + 0.5134311120983222*depth)*depth)
        return -3820.9148917040493 + depth*(535.0542643832791 + (0.23573910721016753 + 0.00003462136482692524*depth)*depth)

    def radius_from_liquid_depth(self, depth):
        if depth <= 0:
            return 0.0
        if depth <= 13.2242:
            return 3.8667311574164636 + 0.7002075382097096*depth
        return 13.050404667978436 + 0.0057498667891316*depth

    @property
    def well_capacity(self):
        return 59505.8

    @property
    def external_tube_height(self):
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

    def height_above_reference_plane(self, space_below_reference_plane, rack):
        from rgatkinson.custom_labware import Opentrons6RackV1, Opentrons10RackV1
        if self._is_rack(rack, [Opentrons6RackV1, Opentrons10RackV1], ['opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical', 'opentrons_6_tuberack_falcon_50ml_conical']):
            return 42.04  # a measured value that accounts for the portion of the tube lying in the dimple at the bottom
        return super().height_above_reference_plane(space_below_reference_plane, rack)

# endregion

#-----------------------------------------------------------------------------------------------------------------------
# Enhanced Well
#-----------------------------------------------------------------------------------------------------------------------

class EnhancedWell(object):

    #-------------------------------------------------------------------------------------------------------------------
    # Construction
    #-------------------------------------------------------------------------------------------------------------------

    # region Construction

    def _initialize(self):
        self.label = None
        from rgatkinson.util import thread_local_storage
        self.config = thread_local_storage.config.wells
        self.__geometry = None
        self.__liquid_volume = None
        self.liquid_volume = LiquidVolume(well=self)

    # endregion

    #-------------------------------------------------------------------------------------------------------------------
    # Accessing
    #-------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def wellSuper(self):
        pass

    @abstractmethod
    def get_parent(self):
        pass

    @property
    @abstractmethod
    def well_depth(self):
        """
        How deep is the interior of the well, in mm?
        """
        pass

    @abstractmethod
    def top_coords_absolute(self):
        """
        Returns the coordinates of the top-center of the well in absolute coordinate space
        """
        pass

    @property
    def geometry(self):
        return self.__geometry

    @geometry.setter
    def geometry(self, value):
        if self.__geometry is not value:
            #
            old_geometry = self.__geometry
            self.__geometry = None
            if old_geometry is not None:
                old_geometry.well = None
            #
            self.__geometry = value
            if self.__geometry is not None:
                self.__geometry.well = self

    @property
    def liquid_volume(self):
        return self.__liquid_volume

    @liquid_volume.setter
    def liquid_volume(self, value):
        if self.__liquid_volume is not value:
            #
            old_liquid_volume = self.__liquid_volume
            self.__liquid_volume = None
            if old_liquid_volume is not None:
                old_liquid_volume.well = None
            #
            self.__liquid_volume = value
            if self.__liquid_volume is not None:
                self.__liquid_volume.well = self

    #-------------------------------------------------------------------------------------------------------------------
    # Pretty printing
    #-------------------------------------------------------------------------------------------------------------------

    # region Pretty Printing

    @abstractmethod
    def _display_class_name(self):
        pass

    def __str__(self):
        if not self.get_parent():  # todo: is the parent *ever* empty?
            return '<{}>'.format(self._display_class_name)
        return '<{} {}>'.format(self._display_class_name, self.get_name())

    def get_name(self):
        result = self.wellSuper().get_name()
        if self.label is not None:
            result += ' (' + self.label + ')'
        return result

    @property
    def has_labelled_well_name(self):
        return True

    # endregion

    #-------------------------------------------------------------------------------------------------------------------
    # Hooking
    #-------------------------------------------------------------------------------------------------------------------

    # region Hooking
    _is_hooked = False

    @classmethod
    def hook_well(cls):
        if not cls._is_hooked:  # make idempotent
            cls._hook_well_instances()
            cls._hook_well_instance_creation()
            cls._hook_other()
            cls._is_hooked = True

    @classmethod
    def _hook_well_instances(cls):
        # Upgrade any existing well instances: usually (always?) only the two possible trash instances
        import gc
        for obj in gc.get_objects():
            if obj.__class__ is WellV1:
                obj.__class__ = EnhancedWellV1
                well: EnhancedWellV1 = obj
                well._initialize()
            if obj.__class__ is WellV2:
                obj.__class__ = EnhancedWellV2
                well: EnhancedWellV1 = obj
                well._initialize()
        assert instance_count(lambda obj: obj.__class__ is WellV1) == 0
        assert instance_count(lambda obj: obj.__class__ is WellV2) == 0

    @classmethod
    def _hook_well_instance_creation(cls):
        # Make sure that any new attempts at instantiating a Well in fact create an EnhancedWell instead
        WellV1.__new__ = cls._well_new_v1
        WellV2.__new__ = cls._well_new_v2

    @staticmethod
    def _well_new_v1(cls, parent=None, properties=None):
        super_class = super(WellV1, EnhancedWellV1)  # Placeable
        result = super_class.__new__(EnhancedWellV1)
        assert result.__class__ is EnhancedWellV1
        return result

    @staticmethod
    def _well_new_v2(cls, well_props: dict,
                 parent: Location,
                 display_name: str,
                 has_tip: bool,
                 api_level: APIVersion):
        super_class = super(WellV2, EnhancedWellV2)  # object
        result = super_class.__new__(EnhancedWellV2)
        assert result.__class__ is EnhancedWellV2
        return result

    @classmethod
    def _hook_other(cls):
        import opentrons.commands.commands as commands
        commands._stringify_legacy_loc = cls._stringify_legacy_loc

    @staticmethod
    def _stringify_legacy_loc(loc: Union[WellV1, Container, Slot, None]) -> str:
        # reworking of that found in commands.py in order to allow for subclasses
        def get_slot(location):
            trace = location.get_trace()
            for item in trace:
                if isinstance(item, Slot):
                    return item

        type_to_text = {Slot: 'slot', Container: 'container', WellV1: 'well'}

        # Coordinates only
        if loc is None:
            return '?'

        location = location_to_list(loc)
        multiple = len(location) > 1

        for cls, name in type_to_text.items():
            if isinstance(location[0], cls):
                text = name
                break
        else:
            text = 'unknown'

        return '{object_text}{suffix} {first}{last} in "{slot_text}"'.format(
                object_text=text,
                suffix='s' if multiple else '',
                first=location[0].get_name(),
                last='...'+location[-1].get_name() if multiple else '',
                slot_text=get_slot(location[0]).get_name())

    # endregion


#-----------------------------------------------------------------------------------------------------------------------

class EnhancedWellV1(EnhancedWell, WellV1):

    #-------------------------------------------------------------------------------------------------------------------
    # Construction
    #-------------------------------------------------------------------------------------------------------------------

    # region Construction

    def __init__(self, parent=None, properties=None):
        super().__init__(parent=parent, properties=properties)
        self._initialize()

    def _initialize(self):
        super()._initialize()
        self.geometry = UnknownWellGeometry(well=self)

    # endregion

    #-------------------------------------------------------------------------------------------------------------------
    # Accessing
    #-------------------------------------------------------------------------------------------------------------------

    def wellSuper(self):
        return super(WellV1, self)

    def get_parent(self):
        return self.parent

    @property
    def well_depth(self):
        return self.z_size()

    def top_coords_absolute(self):
        xyz = pose_tracker.absolute(robot.poses, self)
        return Vector(xyz)

    #-------------------------------------------------------------------------------------------------------------------
    # Pretty printing
    #-------------------------------------------------------------------------------------------------------------------

    # region Pretty Printing

    @property
    def _display_class_name(self):
        return WellV1.__name__

    def get_type(self):
        return self.properties.get('type', self._display_class_name)

    # endregion

#-----------------------------------------------------------------------------------------------------------------------

class EnhancedWellV2(EnhancedWell, WellV2):

    #-------------------------------------------------------------------------------------------------------------------
    # Construction
    #-------------------------------------------------------------------------------------------------------------------

    # region Construction

    def __init__(self, well_props: dict,
                 parent: Location,
                 display_name: str,
                 has_tip: bool,
                 api_level: APIVersion) -> None:
        super().__init__(well_props, parent, display_name, has_tip, api_level)
        self._initialize()

    def _initialize(self):
        super()._initialize()
        self.geometry = UnknownWellGeometry(well=self)

    #-------------------------------------------------------------------------------------------------------------------
    # Accessing
    #-------------------------------------------------------------------------------------------------------------------

    def wellSuper(self):
        return super(WellV2, self)

    def get_parent(self):
        return self._parent

    @property
    def well_depth(self):
        return self._depth

    def top_coords_absolute(self):
        pass  # WRONG

    #-------------------------------------------------------------------------------------------------------------------
    # Pretty printing
    #-------------------------------------------------------------------------------------------------------------------

    # region Pretty Printing

    @property
    def _display_class_name(self):
        return WellV2.__name__

    # endregion

