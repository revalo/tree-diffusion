from typing import Tuple

import iceberg as ice
from lark import Transformer, Tree
from lark.visitors import v_args

from td.environments.environment import Environment
from td.environments.goal_checker import GaussianImageGoalChecker
from td.grammar import Compiler, Grammar

_grammar_spec = r"""
s: arrange | box | ball
direction: "v" -> v | "h" -> h
color: "red" -> red | "green" -> green | "blue" -> blue | "yellow" -> yellow | "purple" -> purple | "orange" -> orange | "black" -> black | "white" -> white
box: "(" "Box" " " color ")"
ball: "(" "Ball" " " color ")"
arrange: "(" "Arrange" " " direction " " s " " s ")"

%ignore /[\t\n\f\r]+/ 
"""

_CANVAS_WIDTH = 64
_CANVAS_HEIGHT = 64

_ice_renderer = ice.Renderer(gpu=False)
_ice_canvas = ice.Blank(
    ice.Bounds(size=(_CANVAS_WIDTH, _CANVAS_HEIGHT)), ice.Colors.WHITE
)


class RainbowToIceberg(Transformer):
    def __init__(
        self, box_w: float = 10, box_h: float = 10, visit_tokens: bool = True
    ) -> None:
        super().__init__(visit_tokens)

        self._box_w = box_w
        self._box_h = box_h

    @v_args(meta=True)
    def box(self, meta, children):
        ice_obj = ice.Rectangle(
            ice.Bounds(size=(self._box_w, self._box_h)),
            fill_color=children[0],
            anti_alias=False,
        )
        ice_obj._lark_meta = meta

        return ice_obj

    @v_args(meta=True)
    def ball(self, meta, children):
        ice_obj = ice.Ellipse(
            rectangle=ice.Bounds(size=(self._box_w, self._box_h)),
            fill_color=children[0],
            anti_alias=True,
        )
        ice_obj._lark_meta = meta

        return ice_obj

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

    def red(self, _):
        return ice.Colors.RED

    def green(self, _):
        return ice.Colors.GREEN

    def blue(self, _):
        return ice.Colors.BLUE

    def yellow(self, _):
        return ice.Colors.YELLOW

    def purple(self, _):
        return ice.Color.from_hex("#800080")

    def orange(self, _):
        return ice.Color.from_hex("#FFA500")

    def black(self, _):
        return ice.Colors.BLACK

    def white(self, _):
        return ice.Colors.WHITE


class RainbowCompiler(Compiler):
    def __init__(self) -> None:
        super().__init__()
        self._expression_to_iceberg = RainbowToIceberg()

    def compile(self, expression: Tree):
        drawable = self._expression_to_iceberg.transform(expression)

        scene = ice.Anchor((_ice_canvas, _ice_canvas.add_centered(drawable)))
        _ice_renderer.render(scene)
        rv = _ice_renderer.get_rendered_image()[:, :, :3] / 255.0

        return rv


class Rainbow(Environment):
    def __init__(self) -> None:
        super().__init__()

        self._grammar = Grammar(
            _grammar_spec,
            start="s",
            primitives=["box", "ball"],
        )

        self._compiler = RainbowCompiler()
        self._goal_checker = GaussianImageGoalChecker(self.compiled_shape)

    @property
    def grammar(self) -> Grammar:
        return self._grammar

    @property
    def compiler(self) -> Compiler:
        return self._compiler

    @property
    def compiled_shape(self) -> Tuple[int, ...]:
        return _CANVAS_WIDTH, _CANVAS_HEIGHT, 3

    @classmethod
    def name(self) -> str:
        return "rainbow"

    def goal_reached(self, compiledA, compiledB) -> bool:
        return self._goal_checker.goal_reached(compiledA, compiledB)
