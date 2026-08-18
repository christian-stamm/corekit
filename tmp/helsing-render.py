# cook your dish here
''' 
    REQS
    - Render Pixel in 3 Colors
    
    - Point 
        - posX
        - posY
    
    - Pixel
        - Point
        - Color
        
    - Line
        - pointSrc
        - pointDst
        - Color
        
    - Triangle
        - PointA
        - PointB
        - pointC
        - Color
        - Fill
    
    - Screen (Fixed Resolution)
        - setPixel(Pixel)
        - setPixels(PixelList)
        - render()
        - animation(List(PixelList), speed)
        
    - Renderer
        - renderLine(Line) -> PixelList
        - renderTriangle(Triangle) -> PixelList
        - renderAnim(Point) -> List()
'''

from dataclasses import dataclass, field
from enum import Enum
import time

class Color(Enum):
    YELLOW = "\x1b[43m \x1b[0m"
    BLUE = "\x1b[44m \x1b[0m"
    BLACK = "\x1b[40m \x1b[0m"
    NONE = " \x1b[0m"

    def __eq__(self, other: "Color"):
        return self.name == other.name

@dataclass
class Point:
   col: int = 0
   row: int = 0

   def __eq__(self, other: "Point") -> bool:
       return self.col == other.col and self.row == other.row

   def __hash__(self):
        return hash((self.col, self.row))

   def inbounds(self, cols: int, rows: int) -> bool:
       return 0 <= self.col and self.col < cols and 0 <= self.row and self.row < rows
       

@dataclass
class Pixel(Point):
   color: Color = field(default_factory=Color.NONE)

   def __eq__(self, other: "Pixel") -> bool:
       return super().__eq__(other) and self.color == other.color

   def __hash__(self):
        return super().__hash__()

@dataclass
class Line:
   point1: Point = field(default_factory=Point(0,0))
   point2: Point = field(default_factory=Point(0,0))
   color: Color = field(default_factory=Color.NONE)

   def __eq__(self, other: "Line") -> bool:
        return self.point1 == other.point1 and self.point2 == other.point2 and self.color == other.color

@dataclass
class Triangle:
   pointA: Point = field(default_factory=Point(0,0))
   pointB: Point = field(default_factory=Point(0,0))
   pointC: Point = field(default_factory=Point(0,0))
   color: Color = field(default_factory=Color.NONE)
   fill: bool = True

class Screen:

    def __init__(self, cols: int, rows: int):
        self.rows = rows
        self.cols = cols
        self.canvas = None
        self.clear()

    def clear(self):
        import os
        os.system('clear')
        self.canvas = [Color.NONE
            for row in range(self.rows)
            for col in range(self.cols)
        ]

    def setPixel(self, pixel: Pixel):
        self.setPixels([pixel])

    def setPixels(self, pixels: list[Pixel]):
        for p in pixels:
            self.canvas[self._pos_to_idx(p)] = p.color

    def draw_line(self, line: Line) -> list[Pixel]:
        # using pre-set algo
        if line.point1 == line.point2:
            return [Pixel(line.point2.col, line.point2.row, line.color)]

        dcol = abs(line.point1.col - line.point2.col)
        drow = abs(line.point1.row - line.point2.row)
        steps = int(max(dcol, drow))

        baseX = line.point1.col
        baseY = line.point1.row
        pixels = []

        for _ in range(steps):
            dx = (line.point2.col - line.point1.col) / steps
            dy = (line.point2.row - line.point1.row) / steps

            baseX += dx
            baseY += dy

            pixels.append(Pixel(int(baseX), int(baseY), line.color))

        return pixels

    def draw_triangle(self, triangle: Triangle) -> list[Pixel]:
        pixels = list[Pixel]()
        pixels += self.draw_line(Line(triangle.pointA, triangle.pointB, triangle.color))
        pixels += self.draw_line(Line(triangle.pointB, triangle.pointC, triangle.color))
        pixels += self.draw_line(Line(triangle.pointC, triangle.pointA, triangle.color))

        if triangle.fill:
            # scanline algo
            lowerY = min(min(triangle.pointA.row, triangle.pointB.row), triangle.pointC.row)
            upperY = max(max(triangle.pointA.row, triangle.pointB.row), triangle.pointC.row)

            infill = list[Line]()
            for row in range(upperY - lowerY):
                refRow = row + lowerY
                candidates = [pixel 
                    for pixel in pixels
                    if pixel.row == refRow
                ]

                pmin = min(candidates, key=lambda x: x.col)
                pmax = max(candidates, key=lambda x: x.col)

                infill.append(Line(Point(pmin.col, refRow), Point(pmax.col, refRow), triangle.color))


            for line in infill:
                pixels += self.draw_line(line)
            
        return pixels

    def draw_flood(self, entry: Pixel):
        frames = []

        frontier = {entry}
        visited = {entry}

        def expand(pixel: Pixel) -> list[Pixel]:
            candidates = [
                Pixel(pixel.col - 1, pixel.row, pixel.color),
                Pixel(pixel.col + 1, pixel.row, pixel.color),
                Pixel(pixel.col, pixel.row - 1, pixel.color),
                Pixel(pixel.col, pixel.row + 1, pixel.color),
            ]

            return [
                p for p in candidates
                if p.inbounds(self.cols, self.rows)
            ]

        while frontier:
            next_frontier = set()

            for target in frontier:
                for p in expand(target):
                    if p not in visited:
                        next_frontier.add(p)

            visited.update(next_frontier)

            frames.append(list(visited))

            frontier = next_frontier

        return frames


    

    def render(self):
        for row in range(self.rows):
            line = str()
            for col in range(self.cols):
                line += self.canvas[self._pos_to_idx(Point(col, row))].value
            print(line)

    def animate(self, frames: list[list[Pixel]], speed: float = 1):
        for frame in frames:
            self.clear()
            self.setPixels(frame)
            self.render()
            time.sleep(1.0 / speed)

    def _pos_to_idx(self, point: Point) -> int:
        if self.rows < point.row:
            raise OverflowError(f"Pixel Coord X is OOB {point.col} max ({self.cols})")
        if self.cols < point.col:
            raise OverflowError(f"Pixel Coord Y is OOB {point.row} max ({self.rows})")

        return point.col + self.cols * point.row

    def _idx_to_pos(self, idx: int) -> Point:
        if len(self.canvas) < idx:
                raise OverflowError(f"Pixel Index is OOB {idx} max ({len(self.canvas)})")

        return Point(idx % self.cols, idx // self.cols)
       
screen = Screen(cols=72, rows=32)
anim = screen.draw_flood(Pixel(32, 15, Color.BLUE))
screen.animate(anim)


