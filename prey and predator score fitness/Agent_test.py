import pygame
import math
import random
import numpy as np
from tools import *
from food import Cell


class Agent(Drawable):
    """Used to represent the concept of a player."""
    COLOR_LIST = [(0, 255, 0)]

    FONT_COLOR = (50, 50, 50)
    vision_color = (192, 192, 192)

    def __init__(self, surface, grid, name="", id=0, PLATFORM_WIDTH=2000, PLATFORM_HEIGHT=2000, font=None, mass=10, speed=4, view_angle=135, rotation_speed=30):        
        super().__init__(surface)
        self.font = font
        self.x = random.randint(100, PLATFORM_WIDTH - 100)  
        self.y = random.randint(100, PLATFORM_HEIGHT - 100)
        self.dx = 0
        self.dy = 0
        self.mass = mass
        self.max_speed = speed
        self.speed = self.max_speed  - self.mass / 10
        self.radius = 300
        self.color = col = random.choice(Agent.COLOR_LIST)
        self.outlineColor = (
            int(col[0] - col[0] / 3),
            int(col[1] - col[1] / 3),
            int(col[2] - col[2] / 3))
        if name:
            self.name = name
        else:
            self.name = "Anonymous"
        self.score = 0
        self.id = id
        self.PLATFORM_WIDTH = PLATFORM_WIDTH
        self.PLATFORM_HEIGHT = PLATFORM_HEIGHT
        self.num_vision_lines = 14
        self.angle = view_angle
        # self.lines = [(0, 0) for _ in range(self.num_vision_lines )]
        self.food_lines = [(0,0) for _ in range(self.num_vision_lines//2)]
        self.pred_lines = [(0,0) for _ in range(self.num_vision_lines//2)]
        self.lines_stats = [10 for i in range(self.num_vision_lines )]

        # Energy attributes
        self.energy = 200  # Initial energy level
        self.energy_depletion_rate = (self.mass /100 + self.max_speed /100) * 2  # Adjust as needed
        self.alive = True
        self.time_alive = 0
        self.closest_cell = (0, 0)
        self.view_angle = 0  # Initial view angle
        self.rotation_speed = rotation_speed  # Adjust as needed

        # Initialize spatial grid
        self.grid = grid
        self.grid.add_object(self)
        self.place_x = (self.PLATFORM_WIDTH - self.x) / self.PLATFORM_WIDTH
        self.place_y = (self.PLATFORM_HEIGHT - self.y) / self.PLATFORM_HEIGHT
        self.children = 0
    def create_vision_lines(self):
        """Creates vision lines for the agent."""
        # Calculate the angle offset based on the view angle of the predator
        angle_offset = math.radians(self.view_angle)

        # Calculate the step value
        step = int(self.angle) * 2 // self.num_vision_lines

        # Iterate over the vision lines
        for i, angle in enumerate(range(int(-self.angle), int(self.angle), step)):
            # Calculate the final angle for each vision line
            final_angle = math.radians(angle) + angle_offset

            # Calculate the endpoint of the vision line using trigonometry
            end_x = self.x + self.radius * math.cos(final_angle)
            end_y = self.y + self.radius * math.sin(final_angle)
            end_point = (int(end_x), int(end_y))

            if i < self.num_vision_lines and i % 2 == 0:
                self.food_lines[i // 2] = end_point
            elif i < self.num_vision_lines and i % 2 != 0:
                self.pred_lines[i // 2] = end_point


    def collisionDetection(self, edibles, predators):
        if not self.alive:
            return
        self.last_lines_stats = [0] * self.num_vision_lines
        
        nearby_edibles = self.grid.get_nearby_objects(self)
        nearby_predators = self.grid.get_nearby_objects(self)

        for edible in nearby_edibles:
            if edible.mass < self.mass + self.mass/2:
                diff = 1
            else:
                diff = -1
            if isinstance(edible, Cell):
                dist_food = getDistance((edible.x, edible.y), (self.x, self.y))
                if dist_food <= self.mass and edible.mass < self.mass + self.mass/2:
                    edibles.remove_cell(edible)
                    # self.grid.remove_object(edible)
                    self.score += 1
                    self.energy += 20  # Increase energy when eating
                elif dist_food <= self.radius:
                    for i, line in enumerate(self.food_lines):
                        if point_on_segment((self.x, self.y), line, (edible.x, edible.y), edible.mass*4):
                            self.lines_stats[i] = dist_food / self.radius * diff
                            self.last_lines_stats[i] = 1
                            

        for predator in nearby_predators:
            if predator.mass + predator.mass / 2 > self.mass :
                diff = 1
            else:
                diff = -1
            if isinstance(predator, Predator):
                dist_pred = getDistance((predator.x, predator.y), (self.x, self.y))
                if dist_pred <= self.radius:
                    for i, line in enumerate(self.pred_lines):
                        if point_on_segment((self.x, self.y), line, (predator.x, predator.y), predator.mass*4):
                            self.lines_stats[i + self.num_vision_lines //2 ] = dist_pred / self.radius * diff
                            self.last_lines_stats[i + self.num_vision_lines //2 ] = 1
                            

        for i in range(self.num_vision_lines):
            if self.last_lines_stats[i] == 0:
                self.lines_stats[i] = 10

    def move(self, rotate_left=0, rotate_right=0, stay=0):
        if not self.alive:
            return

        elif stay:
            return


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
            if self.x + self.dx < 0 :
                self.x = self.PLATFORM_WIDTH
            elif self.x + self.dx > self.PLATFORM_WIDTH:
                self.x = 0
            if self.y + self.dy < 0:
                self.y = self.PLATFORM_HEIGHT
            elif self.y + self.dy > self.PLATFORM_HEIGHT:
                self.y = 0

            next_x = self.x + self.dx
            next_y = self.y + self.dy
            self.grid.move_object(self, next_x, next_y)


            self.x = next_x
            self.y = next_y
            self.x = max(0, min(self.x, self.PLATFORM_WIDTH))
            self.y = max(0, min(self.y, self.PLATFORM_HEIGHT))
            # Update other attributes and methods as needed
            self.create_vision_lines()
            self.time_alive += 0.1

            self.energy -= self.energy_depletion_rate

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
            self.surface = None
        if self.score == 0:
            self.time_alive = -100

    def drawText(self,message,pos,font,color=(255,255,255)):
        """Blits text to main (global) screen.
        """
        self.surface.blit(font.render(message,1,color),pos)


    def draw_vision_lines(self):
        """Draws lines representing the agent's vision."""
        # Define the colors for the vision lines
        j = 1
        k = 0
        for i,line in enumerate(self.food_lines):
            # Draw the vision line from the agent's position to the endpoint
            end_point = (int(line[0]), int(line[1]))
            pygame.draw.line(self.surface, self.vision_color, (self.x, self.y), end_point, 1)
            # self.drawText(str(j), end_point,self.font, self.FONT_COLOR)
            j += 2
        for i,line in enumerate(self.pred_lines):
            # Draw the vision line from the agent's position to the endpoint
            k += 2
            end_point = (int(line[0]), int(line[1]))
            pygame.draw.line(self.surface, (0,0,255), (self.x, self.y), end_point, 1)
            # self.drawText(str(k), end_point,self.font, self.FONT_COLOR)


    def draw_with_cam(self):

        """Draws the player as an outlined circle."""
        center = (int(self.x), int(self.y))
        # Draw the outline of the player as a darker, bigger circle
        # pygame.draw.circle(self.surface, self.outlineColor, center, int((self.mass / 2 + 3)))
        # Draw the actual player as a circle
        pygame.draw.circle(self.surface, self.color, center, int(self.mass / 2))
        self.draw_vision_lines()

    def draw(self):
        if not self.alive:
            return
        self.draw_with_cam()





class Predator(Drawable):
    """Predator class."""
    vision_color = (192, 192, 192)  # Red color for vision lines

    def __init__(self, surface, grid, name="", id=0, PLATFORM_WIDTH=2000, PLATFORM_HEIGHT=2000, font=None, mass=10, speed=4, view_angle=135, rotation_speed=30):
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
        self.mass = mass
        self.max_speed = speed
        self.speed = self.max_speed - self.mass / 10
        self.rotation_speed = rotation_speed  # Adjust as needed
        self.angle = view_angle
        self.radius = 400
        self.outlineColor = (int(255 - 255 / 3), 0, 0)
        self.name = name if name else "Anonymous"
        self.score = 0
        self.prev_stat = "up"
        self.id = id
        self.energy = 200  # Initial energy level
        self.energy_depletion_rate = (self.mass / 100 + self.max_speed / 100) * 2  # Adjust as needed
        self.alive = True
        self.time_alive = 0
        self.num_vision_lines = 7
        self.lines = [(0, 0) for _ in range(self.num_vision_lines)]
        self.lines_stats = [1 for _ in range(self.num_vision_lines)]

        self.view_angle = 0  # Initial view angle
        self.stack = 0

        self.grid = grid
        self.grid.add_object(self)
        self.dash_multiplier = 1

    def create_vision_lines(self):
        """Creates vision lines for the agent."""
        angle_offset = math.radians(self.view_angle)
        step = int(self.angle) * 2 // self.num_vision_lines
        # Iterate over the vision lines
        for i, angle in enumerate(range(int(-self.angle), int(self.angle), step)):
            final_angle = math.radians(angle) + angle_offset
            end_x = self.x + self.radius * math.cos(final_angle)
            end_y = self.y + self.radius * math.sin(final_angle)
            end_point = (int(end_x), int(end_y))
            if i < self.num_vision_lines:
                self.lines[i] = end_point

    def collisionDetection(self, edibles):
        """Detects cells being inside the radius of current player.
        Those cells are eaten.
        """
        self.last_lines_stats = [0 for _ in range(self.num_vision_lines)]
        for edible in edibles:
            if not edible.alive:
                continue
            if edible.mass < self.mass + self.mass/2:
                diff = 1
            else:
                diff = -1
            dist = getDistance((edible.x, edible.y), (self.x,self.y))
            if(dist <= self.mass and edible.mass  < self.mass + self.mass/2 ):
                #self.mass+=0.5
                edible.die()
                edible.score -= 5 
                edibles.remove(edible)
                self.score +=1
                self.energy += 50  # Increase energy when eating


            elif dist <= self.radius:
                for i,line in enumerate(self.lines):
                    if point_on_segment((self.x,self.y),line,(edible.x, edible.y), edible.mass*3):
                        self.lines_stats[i] = dist/self.radius * diff
                        self.last_lines_stats[i] = 1
                        continue




        for i in range(self.num_vision_lines):
            if self.last_lines_stats[i] == 0:
                self.lines_stats[i] = 1



    def move(self, rotate_left=0, rotate_right=0, dash=0, stay=0):
        if not self.alive:
            return
        # Define rotation speed and adjust angle accordingly
        if dash:
            self.dash_multiplier = 5
            self.dx = self.speed * self.dash_multiplier * np.cos(np.deg2rad(self.view_angle))
            self.dy = self.speed * self.dash_multiplier * np.sin(np.deg2rad(self.view_angle))
        elif stay:
            return


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

            if self.x + self.dx < 0 :
                self.x = self.PLATFORM_WIDTH
            elif self.x + self.dx > self.PLATFORM_WIDTH:
                self.x = 0
            if self.y + self.dy < 0:
                self.y = self.PLATFORM_HEIGHT
            elif self.y + self.dy > self.PLATFORM_HEIGHT:
                self.y = 0

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
                self.energy -= self.energy_depletion_rate * 2
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
            self.surface = None
        if self.score == 0:
            self.time_alive = -100

    def drawText(self,message,pos,font,color=(255,255,255)):
        """Blits text to main (global) screen.
        """
        self.surface.blit(font.render(message,1,color),pos)
        
    def draw_vision_lines(self):
        """Draws lines representing the agent's vision."""
        i=1
        for line in self.lines:
            end_point = (int(line[0]), int(line[1]))
            pygame.draw.line(self.surface, self.vision_color, (self.x, self.y), end_point, 1)
            # self.drawText(str(i), end_point,self.font, (0,0,0))
            i+=1



    def draw(self):
        if not self.alive:
            return
        center = (int(self.x), int(self.y))
        # pygame.draw.circle(self.surface, self.outlineColor, center, int((self.mass / 2 + 3)))
        pygame.draw.circle(self.surface, self.color, center, int(self.mass / 2))
        self.draw_vision_lines()