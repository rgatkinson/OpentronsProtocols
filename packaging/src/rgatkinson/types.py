#
# types.py
#
import enum
from enum import Enum
from typing import NamedTuple, Callable, Union

from opentrons import instruments


@enum.unique
class TipWetness(Enum):
    NONE = enum.auto()
    DRY = enum.auto()
    WETTING = enum.auto()
    WET = enum.auto()


class PipNameV1V2(NamedTuple):
    v1_ctor: Callable
    v2_name: str


@enum.unique
class PipetteName(Enum):
    P10_Single = PipNameV1V2(instruments.P10_Single, 'p10_single')
    P10_Multi = PipNameV1V2(instruments.P10_Multi, 'p10_multi')
    P50_Single = PipNameV1V2(instruments.P50_Single, 'p50_single')
    P50_Multi = PipNameV1V2(instruments.P50_Multi, 'p50_multi')
    P300_Single = PipNameV1V2(instruments.P300_Single, 'p300_single')
    P300_Multi = PipNameV1V2(instruments.P300_Multi, 'p300_multi')
    P1000_Single = PipNameV1V2(instruments.P1000_Single, 'p1000_single')
    P20_Single_Gen2 = PipNameV1V2(instruments.P20_Single_GEN2, 'p20_single_gen2')
    P300_Single_Gen2 = PipNameV1V2(instruments.P300_Single_GEN2, 'p300_single_gen2')
    P1000_Single_Gen2 = PipNameV1V2(instruments.P1000_Single_GEN2, 'p1000_single_gen2')


EnhancedWellType = Union['EnhancedWell']