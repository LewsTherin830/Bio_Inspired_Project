from Agent import Agent
from food import Cell
import pygame
import math
import random
from tools import *
import numpy as np

class Predator(Drawable):
    """Predator class."""
    vision_color = (192, 192, 192)  # Red color for vision lines

    def __init__(self, surface, grid, name="", id=0, PLATFORM_WIDTH=2000, PLATFORM_HEIGHT=2000, font=None):
        super().__init__(surface)
        self.color = (255, 0, 0)  # Red color
        self.outlineColor = (150, 0, 0)  # Darker red for outline
        self.font = font
        self.PLATFORM_WIDTH = PLATFORM_WIDTH
        self.PLATFORM_HEIGHT = PLATFORM_HEIGHT
        self.x = random.randint(100, PLATFORM_WIDTH - 100)
        self.y = random.randint(100, PLATFORM_HEIGHT - 100)
        self.dx = 0
        self.dy = 0
        self.mass = 10
        self.speed = 4
        self.rotation_speed = 30  # Adjust as needed
        self.radius = 150
        self.outlineColor = (int(255 - 255 / 3), 0, 0)
        self.name = name if name else "Anonymous"
        self.score = 0
        self.prev_stat = "up"
        self.id = id
        self.energy = 100  # Initial energy level
        self.energy_depletion_rate = 0.1  # Adjust as needed
        self.alive = True
        self.time_alive = 0
        self.num_vision_lines = 7
        self.lines = [(0, 0) for _ in range(self.num_vision_lines)]
        self.lines_stats = [1 for _ in range(self.num_vision_lines)]
        self.new_direction = (0, 0)  # Initialize new direction
        self.current_direction = (0, 0)  # Initialize current movement direction
        self.transition_frames = 60  # Number of frames for transition
        self.transition_counter = 0.1  # Counter for transition frames
        self.view_angle = 0  # Initial view angle
        self.stack = 0

        self.grid = grid
        self.grid.add_object(self)
        self.dash_multiplier = 1

    def create_vision_lines(self):
        """Creates vision lines for the agent."""
        angle_offset = math.radians(self.view_angle)
        step = 90 // (self.num_vision_lines - 1)
        for i, angle in enumerate(range(-45, 46, step)):
            final_angle = math.radians(angle) + angle_offset
            end_x = self.x + self.radius * math.cos(final_angle)
            end_y = self.y + self.radius * math.sin(final_angle)
            end_point = (int(end_x), int(end_y))
            if i < self.num_vision_lines:
                self.lines[i] = end_point

    def collisionDetection(self, edibles):
            """Detects cells being inside the radius of current player."""
            self.last_lines_stats = [0 for _ in range(self.num_vision_lines)]
            nearby_edibles = self.grid.get_nearby_objects(self)

            for edible in nearby_edibles:
                if isinstance(edible, Cell):
                    dist_food = getDistance((edible.x, edible.y), (self.x, self.y))

                    if dist_food <= self.mass :
                        edibles.remove(edible)
                        self.grid.remove_object(edible)
                        self.score += 1
                        self.energy += 10
                    elif dist_food <= self.radius:
                        for i, line in enumerate(self.lines):
                            if point_on_segment((self.x, self.y), line, (edible.x, edible.y), edible.mass * 3):
                                self.lines_stats[i] = dist_food / self.radius
                                self.last_lines_stats[i] = 1
                                break

            for i in range(self.num_vision_lines):
                if self.last_lines_stats[i] == 0:
                    self.lines_stats[i] = 1



    def move(self, rotate_left=0, rotate_right=0, dash_forward=0, dash_backward=0):
        if not self.alive:
            return
        # Define rotation speed and adjust angle accordingly
        if dash_forward:
            self.dash_multiplier = 5
            self.dx = self.speed * self.dash_multiplier * np.cos(np.deg2rad(self.view_angle))
            self.dy = self.speed * self.dash_multiplier * np.sin(np.deg2rad(self.view_angle))
        elif dash_backward:
            self.dash_multiplier = 5
            # Move backward with dash speed
            self.dx = -self.speed * self.dash_multiplier * np.cos(np.deg2rad(self.view_angle))
            self.dy = -self.speed * self.dash_multiplier * np.sin(np.deg2rad(self.view_angle))


        else:
            if rotate_left:
                self.view_angle += self.rotation_speed
            elif rotate_right:
                self.view_angle -= self.rotation_speed

            # Normalize view_angle to be within 0 to 360 degrees
            self.view_angle %= 360

            # Convert angle to radians
            angle_rad = np.deg2rad(self.view_angle)

            # Calculate new position based on angle and movement speed
            self.dx = self.speed * np.cos(angle_rad)
            self.dy = self.speed * np.sin(angle_rad)

 
    def update(self):
        if not self.alive:
            return
        if self.energy > 0:
            next_x = self.x + self.dx
            next_y = self.y + self.dy
            self.grid.move_object(self, next_x, next_y)


            self.x = next_x
            self.y = next_y
            self.x = max(0, min(self.x, self.PLATFORM_WIDTH))
            self.y = max(0, min(self.y, self.PLATFORM_HEIGHT))
            self.create_vision_lines()
            self.time_alive += 0.1
            self.energy -= self.energy_depletion_rate
            
            if self.dash_multiplier > 1:
                self.dx = self.dx / self.dash_multiplier
                self.dy = self.dy / self.dash_multiplier
                self.dash_multiplier = 1
        else:
            self.die()

    def die(self):
        """Kills the player."""
        if self.alive:
            self.dx = 0
            self.dy = 0
            self.color = (192, 192, 192)  # Change color to gray when energy is depleted
            self.vision_color = (169, 169, 169)  # Change vision line color to dark gray when energy is depleted
            self.outlineColor = (169, 169, 169)  # Change outline color to dark gray when energy is depleted
            self.alive = False
            self.grid.remove_object(self)

    def draw_vision_lines(self):
        """Draws lines representing the agent's vision."""
        for line in self.lines:
            end_point = (int(line[0]), int(line[1]))
            pygame.draw.line(self.surface, self.vision_color, (self.x, self.y), end_point, 1)

    def draw(self):
        center = (int(self.x), int(self.y))
        pygame.draw.circle(self.surface, self.outlineColor, center, int((self.mass / 2 + 3)))
        pygame.draw.circle(self.surface, self.color, center, int(self.mass / 2))
        self.draw_vision_lines()