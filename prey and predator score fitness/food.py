import pygame, random
from tools import *

class Cell(Drawable):
    CELL_COLORS = [
        (80,252,54),
        (36,244,255),
        (4,39,243),
        (254,6,178),
        (255,211,7),
        (216,6,254),
        (145,255,7),
        (7,255,182),
        (147,7,255)
    ]

    def __init__(self, surface, PLATFORM_HEIGHT, PLATFORM_WIDTH, spatial_grid=None):
        super().__init__(surface)
        x_limit = PLATFORM_WIDTH - 20
        y_limit = PLATFORM_HEIGHT - 20
        self.x = random.randint(20, x_limit)
        self.y = random.randint(20, y_limit)
        self.mass = random.randint(2, 10)
        self.color = random.choice(Cell.CELL_COLORS)
        self.grid = spatial_grid
        self.grid.add_object(self)


    def draw(self):
        center = (int(self.x), int(self.y))
        pygame.draw.circle(self.surface, self.color, center, int(self.mass))


class CellList(Drawable):
    def __init__(self, surface, numOfCells, PLATFORM_HEIGHT, PLATFORM_WIDTH, spatial_grid=None):
        super().__init__(surface)
        self.PLATFORM_HEIGHT = PLATFORM_HEIGHT
        self.PLATFORM_WIDTH = PLATFORM_WIDTH
        self.count = numOfCells
        self.list = []
        self.grid = spatial_grid
        for _ in range(self.count):
            self.list.append(Cell(self.surface, PLATFORM_HEIGHT, PLATFORM_WIDTH, self.grid))

    def remove_cell(self, cell):
        if cell in self.list:
            self.list.remove(cell)
            self.count -= 1
            self.grid.remove_object(cell)

    def add_cell(self, new_cells):
        for _ in range(new_cells):
            cell = Cell(self.surface, self.PLATFORM_HEIGHT, self.PLATFORM_WIDTH, self.grid)
            self.list.append(cell)
            self.count += 1
            self.grid.add_object(cell)

    def draw(self):
        for cell in self.list:
            cell.draw()
