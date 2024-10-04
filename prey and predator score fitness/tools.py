import pygame, math




def getDistance(a, b):
    """Calculates Euclidean distance between given points.
    """
    diffX = math.fabs(a[0]-b[0])
    diffY = math.fabs(a[1]-b[1])
    return ((diffX**2)+(diffY**2))**(0.5)

def point_to_segment(A, B, cell):
    # Calculate vectors AB and AC
    AB = (B[0] - A[0], B[1] - A[1])
    AC = (cell[0] - A[0], cell[1] - A[1])

    # Calculate dot products
    dot_AB_AC = AB[0] * AC[0] + AB[1] * AC[1]
    dot_AB_AB = AB[0] * AB[0] + AB[1] * AB[1]

    # Check if C lies approximately on segment AB within the tolerance
        # Calculate the distance from point C to the line AB
    cross = AB[0] * AC[1] - AB[1] * AC[0]
    distance = abs(cross) / math.sqrt(dot_AB_AB)
    return distance

def point_on_segment(A, B, cell, tolerance):
    # Calculate vectors AB and AC
    AB = (B[0] - A[0], B[1] - A[1])
    AC = (cell[0] - A[0], cell[1] - A[1])

    # Calculate dot products
    dot_AB_AC = AB[0] * AC[0] + AB[1] * AC[1]
    dot_AB_AB = AB[0] * AB[0] + AB[1] * AB[1]

    # Check if C lies approximately on segment AB within the tolerance
    if 0 <= dot_AB_AC <= dot_AB_AB:
        # Calculate the distance from point C to the line AB
        cross = AB[0] * AC[1] - AB[1] * AC[0]
        distance = abs(cross) / math.sqrt(dot_AB_AB)
        # Check if the distance is within the tolerance
        if abs(distance) <= tolerance:
            return True
    return False

class Drawable:
    """Used as an abstract base-class for every drawable element.
    """

    def __init__(self, surface):
        self.surface = surface

    def draw(self):
        pass


class Grid(Drawable):
    """Used to represent the backgroun grid.
    """

    def __init__(self, surface):
        super().__init__(surface)
        self.color = (230,240,240)

    def draw(self):
        # A grid is a set of horizontal and prependicular lines
        for i in range(0,2001,25):
            pygame.draw.line(self.surface,  self.color, (0,i), (2001, i), 3)
            pygame.draw.line(self.surface, self.color, (i,0), (i, 2001), 3)


class Painter:
    """Used to organize the drawing/ updating procedure.
    Implemantation based on Strategy Pattern.
    Note that Painter draws objects in a FIFO order.
    Objects added first, are always going to be drawn first.
    """

    def __init__(self):
        self.paintings = []

    def add(self, drawable):
        self.paintings.append(drawable)

    def paint(self):
        for drawing in self.paintings:
            drawing.draw()
            
class SpatialHashGrid(Drawable):
    def __init__(self, cell_size, surface, platform_width, platform_height):
        super().__init__(surface)
        self.cell_size = cell_size
        self.platform_width = platform_width
        self.platform_height = platform_height
        self.cells = {}

    def _hash(self, x, y):
        return (x // self.cell_size, y // self.cell_size)

    def add_object(self, obj):
        cell = self._hash(obj.x, obj.y)
        if cell not in self.cells:
            self.cells[cell] = []
        self.cells[cell].append(obj)

    def remove_object(self, obj):
        cell = self._hash(obj.x, obj.y)
        if cell in self.cells:
            if obj in self.cells[cell]:
                self.cells[cell].remove(obj)
                if not self.cells[cell]:
                    del self.cells[cell]

    def move_object(self, obj, new_x, new_y):
        old_cell = self._hash(obj.x, obj.y)
        new_cell = self._hash(new_x, new_y)
        if old_cell != new_cell:
            self.remove_object(obj)
            obj.x = new_x
            obj.y = new_y
            self.add_object(obj)
        else:
            obj.x = new_x
            obj.y = new_y

    def get_nearby_objects(self, obj):
        cell = self._hash(obj.x, obj.y)
        nearby_objects = []
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                nearby_cell = (cell[0] + dx, cell[1] + dy)
                if nearby_cell in self.cells:
                    nearby_objects.extend(self.cells[nearby_cell])
        return nearby_objects

    def draw(self):
        color = (230, 240, 240)
        for x in range(0, self.platform_width, self.cell_size):
            pygame.draw.line(self.surface, color, (x, 0), (x, self.platform_height))
        for y in range(0, self.platform_height, self.cell_size):
            pygame.draw.line(self.surface, color, (0, y), (self.platform_width, y))
