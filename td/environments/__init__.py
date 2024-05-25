from .blockworld import Blockworld
from .csg2d import CSG2D
from .csg2dh import CSG2DH
from .csg2da import CSG2DA
from .environment import Environment
from .nanosvg import NanoSVG
from .rainbow import Rainbow
from .tinysvg import TinySVG
from .tinysvgoffset import TinySVGOffset

environments = {
    Blockworld.name(): Blockworld,
    Rainbow.name(): Rainbow,
    TinySVG.name(): TinySVG,
    NanoSVG.name(): NanoSVG,
    CSG2D.name(): CSG2D,
    CSG2DH.name(): CSG2DH,
    TinySVGOffset.name(): TinySVGOffset,
    CSG2DA.name(): CSG2DA,
}

__all__ = [
    "Environment",
    "environments",
    "Blockworld",
    "Rainbow",
    "TinySVG",
    "NanoSVG",
    "CSG2D",
    "CSG2DH",
    "TinySVGOffset",
    "CSG2DA",
]
