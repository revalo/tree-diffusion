from typing import Tuple

from lark import Tree, Transformer
from td.grammar import Grammar, Compiler
from td.environments.environment import Environment
from td.environments.goal_checker import GaussianImageGoalChecker
import iceberg as ice

_grammar_spec = r"""
s: "(" arrange ")" | box
direction: "v" -> v | "h" -> h                  
box: "Box"
arrange: "Arrange" " " direction " " s " " s

%ignore /[\t\n\f\r]+/ 
"""

_CANVAS_WIDTH = 64
_CANVAS_HEIGHT = 64

_ice_renderer = ice.Renderer(gpu=False)
_ice_canvas = ice.Blank(
    ice.Bounds(size=(_CANVAS_WIDTH, _CANVAS_HEIGHT)), ice.Colors.WHITE
)


class BlockworldToIceberg(Transformer):
    def __init__(
        self, box_w: float = 10, box_h: float = 10, visit_tokens: bool = True
    ) -> None:
        super().__init__(visit_tokens)

        self._box_w = box_w
        self._box_h = box_h

    def box(self, _):
        return ice.Rectangle(
            ice.Bounds(size=(self._box_w, self._box_h)),
            fill_color=ice.Color(0, 0, 0, 0.5),
            anti_alias=False,
        )

    def arrange(self, children):
        direction, left, right = children

        return ice.Arrange(
            [left, right],
            arrange_direction=ice.Arrange.Direction.HORIZONTAL
            if direction == "h"
            else ice.Arrange.Direction.VERTICAL,
        )

    def s(self, children):
        return children[0]

    def v(self, _):
        return "v"

    def h(self, _):
        return "h"


class BlockworldCompiler(Compiler):
    def __init__(self) -> None:
        super().__init__()
        self._expression_to_iceberg = BlockworldToIceberg()

    def compile(self, expression: Tree):
        drawable = self._expression_to_iceberg.transform(expression)

        scene = ice.Anchor((_ice_canvas, _ice_canvas.add_centered(drawable)))
        _ice_renderer.render(scene)
        rv = _ice_renderer.get_rendered_image()[:, :, :1] / 255.0

        return rv


class Blockworld(Environment):
    def __init__(self) -> None:
        super().__init__()

        self._grammar = Grammar(
            _grammar_spec,
            start="s",
            sampling_weights={
                "s": [0.5, 0.5],
            },
            primitives=["box"],
        )

        self._compiler = BlockworldCompiler()
        self._goal_checker = GaussianImageGoalChecker(self.compiled_shape)

    @property
    def grammar(self) -> Grammar:
        return self._grammar

    @property
    def compiler(self) -> Compiler:
        return self._compiler

    @property
    def compiled_shape(self) -> Tuple[int, ...]:
        return _CANVAS_WIDTH, _CANVAS_HEIGHT, 1

    @classmethod
    def name(self) -> str:
        return "blockworld"

    def goal_reached(self, compiledA, compiledB) -> bool:
        return self._goal_checker.goal_reached(compiledA, compiledB)
