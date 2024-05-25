from typing import Tuple

from lark import Transformer, Tree

from td.environments.environment import Environment
from td.environments.goal_checker import BinaryIOUGoalChecker
from td.grammar import Compiler, Grammar

import skia
import math
import random

_grammar_spec = r"""
s: add | subtract | circle | quad

// Number quantized 0 to 16.
number: "0" -> zero | "1" -> one | "2" -> two | "3" -> three | "4" -> four | "5" -> five | "6" -> six | "7" -> seven | "8" -> eight | "9" -> nine | "A" -> ten | "B" -> eleven | "C" -> twelve | "D" -> thirteen | "E" -> fourteen | "F" -> fifteen

// angles [0, 45, 90, 135, 180, 225, 270, 315]
angle: "G" -> zerodeg | "H" -> onedeg | "I" -> twodeg | "J" -> threedeg | "K" -> fourdeg | "L" -> fivedeg | "M" -> sixdeg | "N" -> sevendeg

// (Circle radius x y)
circle: "(" "Circle" " " number " " number " " number ")"

// (Quad x0 y0 x1 y1 x2 y2 x3 y3)
// quad: "(" "Quad" " " number " " number " " number " " number " " number " " number " " number " " number ")"

// (Quad x y w h angle)
quad: "(" "Quad" " " number " " number " " number " " number " " angle ")"

// (+ a b)
add: "(" "+" " " s " " s ")"

// (- a b)
subtract: "(" "-" " " s " " s ")"

%ignore /[\t\n\f\r]+/ 
"""

_CANVAS_WIDTH = 224
_CANVAS_HEIGHT = 224

_SCALE_X = 224 / 32
_SCALE_Y = 224 / 32


class CSG2DtoPath(Transformer):
    def __init__(
        self,
        visit_tokens: bool = True,
    ) -> None:
        super().__init__(visit_tokens)

    def quad(self, children):
        x, y, w, h, angle_degrees = children

        x *= 2
        y *= 2
        w *= 2
        h *= 2

        # Coordinates of the four corners of the quad.
        # (x, y) is the center of the quad.
        x0 = x - w / 2
        y0 = y - h / 2
        x1 = x + w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        x3 = x - w / 2
        y3 = y + h / 2

        # Rotate the quad.
        angle = math.radians(angle_degrees)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        x0, y0 = (
            x + (x0 - x) * cos_angle - (y0 - y) * sin_angle,
            y + (x0 - x) * sin_angle + (y0 - y) * cos_angle,
        )
        x1, y1 = (
            x + (x1 - x) * cos_angle - (y1 - y) * sin_angle,
            y + (x1 - x) * sin_angle + (y1 - y) * cos_angle,
        )
        x2, y2 = (
            x + (x2 - x) * cos_angle - (y2 - y) * sin_angle,
            y + (x2 - x) * sin_angle + (y2 - y) * cos_angle,
        )
        x3, y3 = (
            x + (x3 - x) * cos_angle - (y3 - y) * sin_angle,
            y + (x3 - x) * sin_angle + (y3 - y) * cos_angle,
        )

        path = skia.Path()
        path.moveTo(x0, y0)
        path.lineTo(x1, y1)
        path.lineTo(x2, y2)
        path.lineTo(x3, y3)
        path.close()
        return path

    def circle(self, children):
        r, x, y = children
        path = skia.Path()
        path.addCircle(x * 2, y * 2, r * 2)
        return path

    def add(self, children):
        left, right = children
        rv = []
        if isinstance(left, skia.Path):
            rv.append(left)
        else:
            rv.extend(left)

        rv.append("+")

        if isinstance(right, skia.Path):
            rv.append(right)
        else:
            rv.extend(right)

        return rv

    def subtract(self, children):
        left, right = children
        rv = []
        if isinstance(left, skia.Path):
            rv.append(left)
        else:
            rv.extend(left)

        rv.append("-")

        if isinstance(right, skia.Path):
            rv.append(right)
        else:
            rv.extend(right)

        return rv

    def s(self, children):
        return children[0]

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

    def ten(self, _):
        return 10

    def eleven(self, _):
        return 11

    def twelve(self, _):
        return 12

    def thirteen(self, _):
        return 13

    def fourteen(self, _):
        return 14

    def fifteen(self, _):
        return 15

    def zerodeg(self, _):
        return 0

    def onedeg(self, _):
        return 45

    def twodeg(self, _):
        return 90

    def threedeg(self, _):
        return 135

    def fourdeg(self, _):
        return 180

    def fivedeg(self, _):
        return 225

    def sixdeg(self, _):
        return 270

    def sevendeg(self, _):
        return 315


class CSG2DCompiler(Compiler):
    def __init__(self) -> None:
        super().__init__()
        self._expression_to_path = CSG2DtoPath()

    def compile(self, expression: Tree):
        surface = skia.Surface(_CANVAS_WIDTH, _CANVAS_HEIGHT)

        with surface as canvas:
            paint = skia.Paint()
            paint.setAntiAlias(True)
            paths_and_ops = self._expression_to_path.transform(expression)
            builder = skia.OpBuilder()
            current_op = skia.kUnion_PathOp

            if not isinstance(paths_and_ops, list):
                paths_and_ops = [paths_and_ops]

            for item in paths_and_ops:
                if item == "+":
                    current_op = skia.kUnion_PathOp
                elif item == "-":
                    current_op = skia.kDifference_PathOp
                else:
                    builder.add(item, current_op)

            path = builder.resolve()

            canvas.scale(_SCALE_X, _SCALE_Y)
            canvas.drawPath(path, paint)

        image = surface.makeImageSnapshot()
        array = image.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        return array[:, :, -1:] / 255.0


def _clip(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


class CSG2DObservsationCompiler(Compiler):
    def __init__(self) -> None:
        super().__init__()
        self._expression_to_path = CSG2DtoPath()

    def compile(self, expression: Tree):
        surface = skia.Surface(_CANVAS_WIDTH, _CANVAS_HEIGHT)

        stroke_width = _clip(random.gauss(0.2, 0.01), 0.03, 0.5)
        seg_length = _clip(random.gauss(0.2, 0.1), 0.05, 0.5)
        seg_deviation = _clip(random.gauss(0.2, 0.1), 0.05, 0.5)

        with surface as canvas:
            canvas.clear(skia.ColorWHITE)

            paint = skia.Paint(
                PathEffect=skia.DiscretePathEffect.Make(seg_length, seg_deviation),
                Style=skia.Paint.kStroke_Style,
                StrokeWidth=stroke_width,
                AntiAlias=True,
                Color=skia.ColorBLACK,
            )

            paths_and_ops = self._expression_to_path.transform(expression)
            builder = skia.OpBuilder()
            current_op = skia.kUnion_PathOp

            if not isinstance(paths_and_ops, list):
                paths_and_ops = [paths_and_ops]

            for item in paths_and_ops:
                if item == "+":
                    current_op = skia.kUnion_PathOp
                elif item == "-":
                    current_op = skia.kDifference_PathOp
                else:
                    builder.add(item, current_op)

            path = builder.resolve()

            canvas.scale(_SCALE_X, _SCALE_Y)
            canvas.drawPath(path, paint)

        image = surface.makeImageSnapshot()
        array = image.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        return array[:, :, :1] / 255.0


class CSG2D(Environment):
    def __init__(self) -> None:
        super().__init__()

        self._grammar = Grammar(
            _grammar_spec,
            start="s",
            primitives=["circle", "quad"],
        )

        self._compiler = CSG2DCompiler()
        self._observation_compiler = CSG2DObservsationCompiler()
        self._goal_checker = BinaryIOUGoalChecker()

    @property
    def grammar(self) -> Grammar:
        return self._grammar

    @property
    def compiler(self) -> Compiler:
        return self._compiler

    @property
    def observation_compiler(self) -> Compiler:
        return self._observation_compiler

    @property
    def compiled_shape(self) -> Tuple[int, ...]:
        return _CANVAS_WIDTH, _CANVAS_HEIGHT, 1

    @classmethod
    def name(self) -> str:
        return "csg2d"

    def goal_reached(self, compiledA, compiledB) -> bool:
        return self._goal_checker.goal_reached(compiledA, compiledB)
