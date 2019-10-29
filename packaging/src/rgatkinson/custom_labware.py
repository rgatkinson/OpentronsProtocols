#
# custom_labware.py
#
# Support for custom labware

import collections

from opentrons import robot
from opentrons.util.vector import Vector

from rgatkinson.interval import fpu
from rgatkinson.util import is_indexable
from rgatkinson.well import WellGeometry, Eppendorf5point0mlTubeGeometry

class Point(object):
    def __init__(self, x=0, y=0):
        if is_indexable(x):
            y = x[1]
            x = x[0]
        self.x = x
        self.y = y

    def __add__(self, p):
        return Point(self.x + p.x, self.y + p.y)

    def __radd__(self, other):
        return self + other

    def __sub__(self, p):
        return Point(self.x - p.x, self.y - p.y)

    def __mul__(self, scalar):
        return Point(self.x * scalar, self.y * scalar)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, scalar):
        return Point(self.x / scalar, self.y / scalar)

    def __getitem__(self, index):
        if index == 0:
            return self.x
        if index == 1:
            return self.y
        raise IndexError


class PointF(Point):
    def __init__(self, x: float = 0.0, y: float = 0.0):
        super().__init__(x,y)


class WellGrid(object):
    def __init__(self, config, grid_size: Point, incr: PointF, offset=PointF(), origin_name='A1', origin=None, well_geometry=None):
        self.config = config
        self.grid_size = grid_size
        self.origin = self.well_name_to_indices(origin_name) if origin is None else origin
        self.max = self.origin + self.grid_size
        self.incr = incr
        self.offset = offset
        self.wells_matrix = self._create_wells_matrix(well_geometry)
        self.wells_by_name = dict()
        for row in self.wells_matrix:
            for well_dict in row:
                self.wells_by_name[well_dict['name']] = well_dict

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.wells_by_name[item]
        if is_indexable(item):
            return self.wells_matrix[item[1]][item[0]]  # item is (x,y), which is (col, row)
        raise IndexError

    @staticmethod
    def well_name_to_indices(name):  # zero-based
        return Point(ord(name[1]) - ord('1'), ord(name[0]) - ord('A'))

    def contains_indices(self, indices):
        return self.origin.x <= indices[0] < self.max.x and self.origin.y <= indices[1] < self.max.y

    def definition_map(self, rack, z_reference, hangable_tube_height):
        result = collections.OrderedDict()
        for col in range(self.grid_size.x):
            for row in range(self.grid_size.y):
                rc = self.wells_matrix[row][col]
                if rc['geometry'] is not None:
                    d = dict()
                    geometry: WellGeometry = rc['geometry']
                    d['depth'] = geometry.well_depth
                    d['totalLiquidVolume'] = geometry.well_capacity
                    d['shape'] = 'circular'
                    d['diameter'] = geometry.well_diameter_at_top
                    d['x'] = rc['x']
                    d['y'] = rc['y']
                    d['z'] = geometry.height_above_reference_plane(hangable_tube_height, rack) + z_reference - d['depth']
                    result[rc['name']] = d
        return result

    @property
    def well_ordering_names(self):  # values are well name, but in a 2D matrix
        result = []
        for col in range(self.grid_size.x):
            col_result = []
            for row in range(self.grid_size.y):
                col_result.append(self.wells_matrix[row][col]['name'])
            result.append(col_result)
        return result

    @property
    def well_names(self):
        return [well_name for column in self.well_ordering_names for well_name in column]

    @property
    def well_geometries(self):
        return [well_dict['geometry'] for row in self.wells_matrix for well_dict in row if well_dict['geometry'] is not None]

    def _create_wells_matrix(self, well_geometry=None):
        result = [None] * self.grid_size.y
        for row in range(self.grid_size.y):
            result[row] = [None] * self.grid_size.x
            for col in range(self.grid_size.x):
                d = {  # coordinate system origin is in lower left
                    'x': self.offset.x + self.incr.x * col,
                    'y': self.offset.y + self.incr.y * (self.grid_size.y - 1 - row),
                    'name': chr(ord('A')+row+self.origin.x) + chr(ord('1')+col+self.origin.y),
                    'geometry': None
                }
                if well_geometry is not None:
                    d['geometry'] = well_geometry(config=self.config, well=None)
                result[row][col] = d
        return result


class CustomTubeRack(object):
    def __init__(self, config, name,
                 dimensions=None,  # is either to reference plane (dimensions_measurement_geometry is None) or to the top of rack measure with some tube in place (otherwise)
                 dimensions_measurement_geometry=None,  # geometry used, if any, when 'dimensions' were measured
                 hangable_tube_height=None,  # tubes taller than this don't hang. this value is conservative, in that tubes slightly larger than this might still hang, depending on geometries of the tube and rack indentations
                 brand=None,
                 brandIds=None,
                 well_grids=None
                 ):
        assert name is not None
        self.config = config
        self.name = name
        self.reference_dimensions = dimensions if dimensions is not None else Vector(x=0, y=0, z=0)
        if dimensions_measurement_geometry is not None:
            # find the z height of the reference plane of the labware
            self.reference_dimensions = self.reference_dimensions - (0, 0, dimensions_measurement_geometry(well=None, config=self.config).rim_lip_height)
        self.hangable_tube_height = hangable_tube_height if hangable_tube_height is not None else fpu.infinity
        self.brand = {
            'brand': brand if brand is not None else 'Atkinson Labs'
        }
        if brandIds is not None:
            self.brand['brandId'] = brandIds
        self.metadata = {
            'displayName': name,
            'displayCategory': 'tubeRack',
            'displayVolumeUnits': 'µL',
            'tags': []
        }
        self.well_grids = [] if well_grids is None else well_grids
        self.load_result = None

    def __getitem__(self, item_name):
        if isinstance(item_name, str):
            return self.__getitem__(WellGrid.well_name_to_indices(item_name))
        if is_indexable(item_name):
            for well_grid in self.well_grids:
                if well_grid.contains_indices(item_name):
                    return well_grid.__getitem__(Point(item_name) - well_grid.origin)
        raise IndexError

    @property
    def max_rim_lip_height(self):
        result = 0
        for geometry in self.well_geometries:
            result = max(result, geometry.rim_lip_height)
        return result

    @property
    def max_tube_height_above_reference_plane(self):
        result = 0
        for geometry in self.well_geometries:
            result = max(result, geometry.height_above_reference_plane(self.hangable_tube_height, self))
        return result

    @property
    def dimensions(self):
        return self.reference_dimensions + (0, 0, self.max_tube_height_above_reference_plane)

    @property
    def well_names(self):
        result = []
        for well_grid in self.well_grids:
            result.extend(well_grid.well_names)
        return result

    @property
    def well_geometries(self):
        result = []
        for well_grid in self.well_grids:
            result.extend(well_grid.well_geometries)
        return result

    @property
    def _definition_map(self):
        dimensions = self.dimensions
        result = collections.OrderedDict()
        result['ordering'] = []
        for well_grid in self.well_grids:
            result['ordering'].extend(well_grid.well_ordering_names)
        result['brand'] = self.brand
        result['metadata'] = self.metadata
        result['dimensions'] = {
            'xDimension': dimensions.coordinates.x,
            'yDimension': dimensions.coordinates.y,
            'zDimension': dimensions.coordinates.z
        }
        result['wells'] = collections.OrderedDict()
        for well_grid in self.well_grids:
            for name, definition in well_grid.definition_map(self, self.reference_dimensions.coordinates.z, self.hangable_tube_height).items():
                result['wells'][name] = definition
        # todo: add 'groups', if that's still significant / worthwhile
        result['parameters'] = {
            'format': 'irregular',  # not 'regular': per correspondence from Opentrons, this field is obsolete, and 'irregular' is best for back-compat
            'quirks': [],
            'isTiprack': False,
            'isMagneticModuleCompatible': False,
            'loadName': self.name
        }
        result['namespace'] = 'custom_beta'
        result['version'] = 1
        result['schemaVersion'] = 2
        result['cornerOffsetFromSlot'] = {'x': 0, 'y': 0, 'z': 0}
        return result

    def load(self, slot=None, label=None):
        slot = str(slot)
        if self.load_result is None:
            def_map = self._definition_map
            if label is None:
                label = self.name
            self.load_result = robot.add_container_by_definition(def_map, slot, label=label)
            for well_name in self.well_names:
                well = self.load_result.wells(well_name)
                geometry = self[well_name].get('geometry', None)
                if geometry is not None:
                    assert geometry.well is None or geometry.well is well
                    assert getattr(well, 'geometry', None) is None or well.geometry is geometry
                    geometry.well = well
        return self.load_result


class Opentrons15Rack(CustomTubeRack):
    def __init__(self, config, name, brand=None, default_well_geometry=None):
        super().__init__(
            config,
            dimensions=Vector(127.76, 85.48, 80.83),
            dimensions_measurement_geometry=Eppendorf5point0mlTubeGeometry,
            hangable_tube_height=71.40,
            name=name,
            brand=brand,
            well_grids=[WellGrid(config,
                grid_size=Point(5, 3),
                incr=PointF(25.0, 25.0),
                offset=PointF(13.88, 17.74),
                well_geometry=default_well_geometry)
            ])


def load_tiprack(tiprack_type, slot, label=None):
    # an enhanced version of labware.load(tiprack_type, slot) that grabs more metadata
    from opentrons.protocol_api import labware as new_labware
    from opentrons.legacy_api.robot.robot import _setup_container
    from opentrons.legacy_api.containers import load_new_labware_def
    slot = str(slot)
    share = False
    definition = new_labware.get_labware_definition(load_name=tiprack_type)
    container = load_new_labware_def(definition)
    container = _setup_container(container)
    #
    container.uri = new_labware.uri_from_definition(definition)
    container.tip_length = definition['parameters']['tipLength']
    container.tip_overlap = definition['parameters']['tipOverlap']
    #
    robot._add_container_obj(container, tiprack_type, slot, label, share)
    return container
