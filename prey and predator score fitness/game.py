import pygame
from tools import *
from Agent_test import Agent, Predator
from food import CellList


NAME = "agar.io"
VERSION = "0.2"
CELLS_NUMBER = 150
# Pygame initialization
pygame.init()
pygame.display.set_caption("{} - v{}".format(NAME, VERSION))
clock = pygame.time.Clock()

try:
    font = pygame.font.Font("Ubuntu-B.ttf",10)
    big_font = pygame.font.Font("Ubuntu-B.ttf",24)
except:
    print("Font file not found: Ubuntu-B.ttf")
    font = pygame.font.SysFont('Ubuntu',10,True)
    big_font = pygame.font.SysFont('Ubuntu',24,True)


def drawText(message,pos,color=(255,255,255)):
    """Blits text to main (global) screen.
    """
    MAIN_SURFACE.blit(font.render(message,1,color),pos)


class Game:

    def __init__(self, screen_width=800, screen_height=800):
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = (screen_width,screen_height)
        self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT = (screen_width,screen_height)
        self.MAIN_SURFACE = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        self.grid = SpatialHashGrid(cell_size=75, surface = self.MAIN_SURFACE ,platform_width=self.SCREEN_WIDTH, platform_height=self.SCREEN_HEIGHT)

        # self.grid = Grid(self.MAIN_SURFACE)
        # self.number_of_cells = 0
        self.number_of_cells = CELLS_NUMBER
        self.cells = CellList(self.MAIN_SURFACE, self.number_of_cells, self.PLATFORM_HEIGHT, self.PLATFORM_WIDTH, spatial_grid=self.grid)
        self.painter = Painter()
        self.painter.add(self.grid)
        self.painter.add(self.cells)
        






class GameInfo:
    def __init__(self, edibles, agents = []):
        self.edibles = edibles
        self.list_agents = agents
        self.agents = {}
        for agent in self.list_agents:
            self.agents[agent.id] = 0
        self.dead_agents = 0

    def get_agent_by_id(self, id):
        for agent in self.list_agents:
            if agent.id == id:
                return agent
        return None
    
    def update_agents_fitness(self):
        for agent in self.list_agents:
            self.agents[agent.id] = agent.time_alive + agent.score * 10
            if agent.alive == False:
                self.dead_agents += 1
                self.list_agents.remove(agent)

    def update_pred_fitness(self):
        for agent in self.list_agents:
            self.agents[agent.id] = agent.score * 20 + agent.time_alive
            if agent.alive == False:
                self.dead_agents += 1
                self.list_agents.remove(agent)


    def get_agent_fitness(self, genome_id):
        return self.agents[genome_id]
    
    # def get_agent_inputs(self, agent):
    #     """Returns the list of inputs for the agent.
    #     """
    #     return (agent.x, agent.y) + (agent.closest_cell)
    def get_agent_inputs(self, agent):
        """Returns the list of inputs for the agent.
        """
        
        return tuple(agent.lines_stats)    
    
    def game_over(self ):
        """Checks if the game is over.
        """
        if self.list_agents == []:
            return True
        else:
            return False

# Initialize essential entities
if __name__ == "__main__":
    game = Game()

    MAIN_SURFACE = game.MAIN_SURFACE

    blob = Agent(MAIN_SURFACE, game.grid, "GeoVas",font=font, PLATFORM_HEIGHT=game.PLATFORM_HEIGHT, PLATFORM_WIDTH=game.PLATFORM_WIDTH, mass=15,rotation_speed=10)
    agents = [blob]
    pred = Predator(MAIN_SURFACE, game.grid,  "GeoVas",font=font, PLATFORM_HEIGHT=game.PLATFORM_HEIGHT, PLATFORM_WIDTH=game.PLATFORM_WIDTH, mass=15)

    info = GameInfo(game.cells.list, agents)
    game.painter.add(blob)
    game.painter.add(pred)
    blob.energy_depletion_rate = 0
    pred.energy_depletion_rate = 0  
    # Game main loop
    while(True):
        
        clock.tick(60)
        
        for e in pygame.event.get():

            if(e.type == pygame.QUIT):
                pygame.quit()
                quit()
        # Player movement
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_UP:
                    blob.move(dash=1)
                if e.key == pygame.K_DOWN:
                    blob.move(dash_backward=1)
                if e.key == pygame.K_LEFT:
                    blob.move(rotate_left=1)
                    blob.prev_stat = "left"
                if e.key == pygame.K_RIGHT:
                    blob.move(rotate_right=1)   
                    blob.prev_stat = "right"
                if e.key == pygame.K_SPACE:
                    blob.move(stop=1)
                    blob.prev_stat = "stop"
                if e.key == pygame.K_w:
                    pred.move(dash=1)
                if e.key == pygame.K_s:
                    pred.move(dash_backward=1)
                if e.key == pygame.K_a:
                    pred.move(rotate_left=1)
                if e.key == pygame.K_d:
                    pred.move(rotate_right=1)


        blob.update()
        # print("score ",blob.score, "children ",blob.children)
        print("view angle ",blob.view_angle, "rotation speed ",blob.rotation_speed, "friction ", blob.view_angle/blob.rotation_speed)
        if blob.score // 10 - 1 == blob.children and blob.score != 0:
                blob.children += 1
                child = Agent(blob.surface, blob.grid, name=blob.name, id=blob.id, PLATFORM_WIDTH=blob.PLATFORM_WIDTH, PLATFORM_HEIGHT=blob.PLATFORM_HEIGHT, font=blob.font, mass=blob.mass, speed=blob.speed, view_angle=blob.angle, rotation_speed=blob.rotation_speed)
                child.x = blob.x
                child.y = blob.y
                agents.append(child)
                game.painter.add(child)
        pred.update()
        # print("food ",info.get_agent_inputs(blob)[1:blob.num_vision_lines//2 + 1]
        #       , "\npredator ",info.get_agent_inputs(blob)[blob.num_vision_lines //2 + 1:])
              
        # print("predator ",info.get_agent_inputs(pred))
        # print ("blob ",blob.time_alive)
        blob.collisionDetection(game.cells, [pred])
        pred.collisionDetection(agents)
        #cam.update(blob)
        MAIN_SURFACE.fill((242,251,255))
        # Uncomment next line to get dark-theme
        #MAIN_SURFACE.fill((0,0,0))
        game.painter.paint()
        # Start calculating next frame
        pygame.display.flip()
