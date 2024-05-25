from typing import Tuple

import iceberg as ice
from lark import Transformer, Tree
from lark.visitors import v_args

from td.environments.environment import Environment
from td.environments.goal_checker import GaussianImageGoalChecker
from td.grammar import Compiler, Grammar

_grammar_spec = r"""
// s: arrange | move | pad | compose | rect | ellipse
s: arrange | rect | ellipse | move
direction: "v" -> v | "h" -> h
color: "red" -> red | "green" -> green | "blue" -> blue | "yellow" -> yellow | "purple" -> purple | "orange" -> orange | "black" -> black | "white" -> white | "none" -> none
number: "0" -> zero | "1" -> one | "2" -> two | "3" -> three | "4" -> four | "5" -> five | "6" -> six | "7" -> seven | "8" -> eight | "9" -> nine
boolean: "true" -> true | "false" -> false

// Rectangle w h fillcolor strokecolor strokewidth
rect: "(" "Rectangle" " " number " " number " " color " " color " " number ")"

// Ellipse w h fillcolor strokecolor strokewidth
ellipse: "(" "Ellipse" " " number " " number " " color " " color " " number ")"

// Arrange direction left right gap
arrange: "(" "Arrange" " " direction " " s " " s " " number ")"

// Move x y negx negy
move: "(" "Move" " " s " " number " " number " " boolean " " boolean ")"

// Pad t r b l
// pad: "(" "Pad" " " s " " number " " number " " number " " number ")"

// Compose without arranging
// compose: "(" "Compose" " " s " " s ")"

%ignore /[\t\n\f\r]+/ 
"""

_CANVAS_WIDTH = 224
_CANVAS_HEIGHT = 224

_ice_renderer = ice.Renderer(gpu=False)
_ice_canvas = ice.Blank(
    ice.Bounds(size=(_CANVAS_WIDTH, _CANVAS_HEIGHT)), ice.Colors.WHITE
)


class _Move(ice.Drawable):
    child: ice.Drawable
    x: float
    y: float

    def setup(self):
        self._child_bounds = self.child.bounds
        self._moved = self.child.move(self.x, self.y)

    @property
    def bounds(self):
        return self._child_bounds

    def draw(self, canvas):
        self._moved.draw(canvas)

    @property
    def children(self):
        return [self._moved]


class TSVGToIceberg(Transformer):
    def __init__(
        self,
        visit_tokens: bool = True,
        stroke_width_divisor: float = 2.0,
        size_multiplier: float = 6.0,
    ) -> None:
        super().__init__(visit_tokens)
        self._stroke_width_divisor = stroke_width_divisor
        self._size_multiplier = size_multiplier

    @v_args(meta=True)
    def rect(self, meta, children):
        w, h, fill_color, stroke_color, stroke_width = children
        stroke_width = stroke_width / self._stroke_width_divisor
        w = w * self._size_multiplier
        h = h * self._size_multiplier

        rv = ice.Rectangle(
            ice.Bounds(size=(w, h)),
            fill_color=fill_color,
            border_color=stroke_color,
            border_thickness=stroke_width,
            anti_alias=False,
            dont_modify_bounds=True,
        )
        rv._lark_meta = meta

        return rv

    @v_args(meta=True)
    def ellipse(self, meta, children):
        w, h, fill_color, stroke_color, stroke_width = children
        stroke_width = stroke_width / self._stroke_width_divisor
        w = w * self._size_multiplier
        h = h * self._size_multiplier

        rv = ice.Ellipse(
            rectangle=ice.Bounds(size=(w, h)),
            fill_color=fill_color,
            border_color=stroke_color,
            border_thickness=stroke_width,
            anti_alias=True,
            dont_modify_bounds=True,
        )
        rv._lark_meta = meta

        return rv

    def arrange(self, children):
        direction, left, right, gap = children

        return ice.Arrange(
            [left, right],
            arrange_direction=ice.Arrange.Direction.HORIZONTAL
            if direction == "h"
            else ice.Arrange.Direction.VERTICAL,
            gap=gap,
        )

    def move(self, children):
        drawable, x, y, negx, negy = children

        x = x * self._size_multiplier
        y = y * self._size_multiplier

        x = x if not negx else -x
        y = y if not negy else -y
        return _Move(child=drawable, x=x, y=y)

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

    def none(self, _):
        return None

    def zero(self, _):
        return 0

    def one(self, _):
        return 1

    def two(self, _):
        return 2

    def three(self, _):
        return 3

    def four(self, _):
        return 4

    def five(self, _):
        return 5

    def six(self, _):
        return 6

    def seven(self, _):
        return 7

    def eight(self, _):
        return 8

    def nine(self, _):
        return 9

    def true(self, _):
        return True

    def false(self, _):
        return False


class TSVGCompiler(Compiler):
    def __init__(self) -> None:
        super().__init__()
        self._expression_to_iceberg = TSVGToIceberg()

    def compile(self, expression: Tree):
        drawable = self._expression_to_iceberg.transform(expression)
        scene = ice.Anchor((_ice_canvas, _ice_canvas.add_centered(drawable)))
        # return scene
        _ice_renderer.render(scene)
        rv = _ice_renderer.get_rendered_image()[:, :, :3] / 255.0

        return rv


class TinySVG(Environment):
    def __init__(self) -> None:
        super().__init__()

        self._grammar = Grammar(
            _grammar_spec,
            start="s",
            primitives=["rect", "ellipse", "move"],
        )

        self._compiler = TSVGCompiler()
        self._goal_checker = GaussianImageGoalChecker(self.compiled_shape, sigma=0.1)

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
        return "tinysvg"

    def goal_reached(self, compiledA, compiledB) -> bool:
        return self._goal_checker.goal_reached(compiledA, compiledB)
