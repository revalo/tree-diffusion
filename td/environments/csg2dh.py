from typing import Tuple

from lark import Transformer, Tree

from td.environments.environment import Environment
from td.environments.goal_checker import BinaryIOUGoalChecker
from td.grammar import Compiler, Grammar

import skia
import math
import random
import numpy as np

_grammar_spec = r"""
s: binop | circle | quad

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

binop: "(" op " " s " " s ")"

op: "+" -> add | "-" -> subtract | "^" -> intersect

%ignore /[\t\n\f\r]+/ 
"""

_CANVAS_WIDTH = 128
_CANVAS_HEIGHT = 128

_SCALE_X = _CANVAS_WIDTH / 32
_SCALE_Y = _CANVAS_HEIGHT / 32


class CSG2DHtoPath(Transformer):
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

    def binop(self, children):
        op, left, right = children

        _op_table = {
            "+": skia.kUnion_PathOp,
            "-": skia.kDifference_PathOp,
            "^": skia.kIntersect_PathOp,
        }

        builder = skia.OpBuilder()
        builder.add(left, skia.kUnion_PathOp)
        builder.add(right, _op_table[op])
        if op == "+":
            builder.add(left, skia.kUnion_PathOp)

        return builder.resolve()

    def add(self, children):
        return "+"

    def subtract(self, children):
        return "-"

    def intersect(self, children):
        return "^"

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


class CSG2DHCompiler(Compiler):
    def __init__(self) -> None:
        super().__init__()
        self._expression_to_path = CSG2DHtoPath()

    def compile(self, expression: Tree):
        surface = skia.Surface(_CANVAS_WIDTH, _CANVAS_HEIGHT)

        with surface as canvas:
            paint = skia.Paint()
            paint.setAntiAlias(True)
            path = self._expression_to_path.transform(expression)

            canvas.scale(_SCALE_X, _SCALE_Y)
            canvas.drawPath(path, paint)

        image = surface.makeImageSnapshot()
        array = image.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        return array[:, :, -1:] / 255.0


def _clip(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


def _jitter_point(point, jitter_std=0.2):
    return point + skia.Point(random.gauss(0, jitter_std), random.gauss(0, jitter_std))


class CSG2DHtoSketchPath(Transformer):
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

        # Add some noise to these parameters.
        x += random.gauss(0, 0.5)
        y += random.gauss(0, 0.5)
        w += random.gauss(0, 0.5)
        h += random.gauss(0, 0.5)
        angle_degrees += random.gauss(0, 1)

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

        rot_noise = 0.1
        # Add some noise to the rotated coordinates.
        x0 += random.gauss(0, rot_noise)
        y0 += random.gauss(0, rot_noise)
        x1 += random.gauss(0, rot_noise)
        y1 += random.gauss(0, rot_noise)
        x2 += random.gauss(0, rot_noise)
        y2 += random.gauss(0, rot_noise)
        x3 += random.gauss(0, rot_noise)
        y3 += random.gauss(0, rot_noise)

        path = skia.Path()
        path.moveTo(x0, y0)
        path.lineTo(x1, y1)
        path.lineTo(x2, y2)
        path.lineTo(x3, y3)
        path.close()
        return path

    def circle(self, children):
        r, x, y = children

        r *= 2
        x *= 2
        y *= 2
        r2 = r

        # Add some noise to these parameters.
        r += random.gauss(0, 0.3)
        r2 += random.gauss(0, 0.3)
        x += random.gauss(0, 0.5)
        y += random.gauss(0, 0.5)

        x = _clip(x, 0, 32)
        y = _clip(y, 0, 32)
        r = _clip(r, 0, 32)
        r2 = _clip(r2, 0, 32)

        path = skia.Path()
        path.addOval(skia.Rect.MakeXYWH(x - r, y - r2, 2 * r, 2 * r2))

        jitter_angle = random.gauss(0, 2)
        transform_matrix = skia.Matrix()
        transform_matrix.setRotate(jitter_angle, x, y)
        path.transform(transform_matrix)

        try:
            segment_length = 3
            path_measure = skia.PathMeasure(path, False)
            total_length = path_measure.getLength()

            points = []
            tangents = []

            for i in np.arange(0, total_length, segment_length):
                point = path_measure.getPosTan(i)
                points.append(_jitter_point(point[0]))
                tangents.append(point[1])

            last_point, last_tangent = path_measure.getPosTan(total_length)
            points.append(_jitter_point(last_point))
            tangents.append(last_tangent)

            new_path = skia.Path()
            new_path.moveTo(points[0])
            for i, (point, tangent, next_point, next_tangent) in enumerate(
                zip(
                    points[:-1],
                    tangents[:-1],
                    points[1:],
                    tangents[1:],
                )
            ):
                # The tangents are unit vectors, but we would like actual time
                # derivatives for the conversion below. That's an underspecified
                # problem (the original path may not even have a notion of time.
                # But by scaling with the segment length we at least get
                # a reasonable choice (in particular, this makes the shape of the
                # interpolation invariant to scaling the entire path).
                tangent = tangent * segment_length
                next_tangent = next_tangent * segment_length
                # Compute control points (i.e. convert from Hermite to Bezier curve):
                p1 = point + tangent * 0.333
                p2 = next_point - next_tangent * 0.333

                new_path.cubicTo(p1, p2, next_point)

            return new_path
        except Exception:
            return path

    def binop(self, children):
        op, left, right = children

        _op_table = {
            "+": skia.kUnion_PathOp,
            "-": skia.kDifference_PathOp,
            "^": skia.kIntersect_PathOp,
        }

        builder = skia.OpBuilder()
        builder.add(left, skia.kUnion_PathOp)
        builder.add(right, _op_table[op])

        if op == "+":
            builder.add(left, skia.kUnion_PathOp)

        return builder.resolve()

    def add(self, children):
        return "+"

    def subtract(self, children):
        return "-"

    def intersect(self, children):
        return "^"

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


def _path_to_sketch_path(path: skia.Path) -> skia.Path:
    verbs = []
    n_points = []
    weights = []

    it = iter(path)
    verb, points = it.next()
    while verb != skia.Path.kDone_Verb:
        verbs.append(verb)
        n_points.append(points)
        if verb == skia.Path.kConic_Verb:
            weights.append(it.conicWeight())
        else:
            weights.append(0)

        verb, points = it.next()

    new_path = skia.Path()

    prev_point = None

    for verb, points, weight in zip(verbs, n_points, weights):
        # print(verb, points, weight)
        if verb == skia.Path.kMove_Verb:
            new_path.moveTo(_jitter_point(points[0]))
            prev_point = points[0]
        elif verb == skia.Path.kLine_Verb:
            current_point = points[-1]
            direction = current_point - prev_point

            # Find two points along around 50% and 75% of the direction vector.
            mid_point = prev_point + direction * (0.5 + random.gauss(0, 0.1))
            far_point = prev_point + direction * (0.75 + random.gauss(0, 0.1))

            jittered_last = _jitter_point(current_point)

            new_path.moveTo(_jitter_point(prev_point))
            new_path.cubicTo(
                _jitter_point(mid_point),
                _jitter_point(far_point),
                jittered_last,
            )
            new_path.moveTo(jittered_last if random.random() < 0.5 else current_point)

            prev_point = points[-1]
        elif verb == skia.Path.kConic_Verb:
            new_path.conicTo(points[-2], points[-1], weight)
            prev_point = points[-1]
        elif verb == skia.Path.kCubic_Verb:
            new_path.moveTo(prev_point)
            new_path.cubicTo(points[-3], points[-2], points[-1])
            prev_point = points[-1]
        elif verb == skia.Path.kClose_Verb:
            new_path.close()

    return new_path


class CSG2DHSketchCompiler(Compiler):
    def __init__(self) -> None:
        super().__init__()
        self._expression_to_path = CSG2DHtoSketchPath()

    def compile(self, expression: Tree):
        surface = skia.Surface(_CANVAS_WIDTH, _CANVAS_HEIGHT)
        stroke_width = random.uniform(0.01, 0.3)

        with surface as canvas:
            canvas.clear(skia.ColorWHITE)
            paint = skia.Paint(
                Style=skia.Paint.kStroke_Style,
                StrokeWidth=stroke_width,
                AntiAlias=True,
                Color=skia.ColorBLACK,
            )
            path_orig = self._expression_to_path.transform(expression)
            path = _path_to_sketch_path(path_orig)

            canvas.scale(_SCALE_X, _SCALE_Y)
            canvas.drawPath(path, paint)

        image = surface.makeImageSnapshot()
        array = image.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        return array[:, :, :1] / 255.0


class CSG2DH(Environment):
    def __init__(self) -> None:
        super().__init__()

        self._grammar = Grammar(
            _grammar_spec,
            start="s",
            primitives=["circle", "quad"],
        )

        self._compiler = CSG2DHCompiler()
        self._observation_compiler = CSG2DHSketchCompiler()
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
        return "csg2dh"

    def goal_reached(self, compiledA, compiledB) -> bool:
        return self._goal_checker.goal_reached(compiledA, compiledB)
