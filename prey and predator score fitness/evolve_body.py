import pygame
import pygad
import random
import numpy as np
from tools import *
from food import Cell, CellList
from Agent_test import Predator, Agent
import neat
import pickle
import matplotlib.pyplot as plt
import os




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



class Game:

    def __init__(self, screen_width=1400, screen_height=800):
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = (screen_width,screen_height)
        self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT = (screen_width,screen_height)
        self.MAIN_SURFACE = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        self.grid = SpatialHashGrid(cell_size=75, surface = self.MAIN_SURFACE ,platform_width=self.SCREEN_WIDTH, platform_height=self.SCREEN_HEIGHT)

        # self.grid = Grid(self.MAIN_SURFACE)
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
            self.agents[agent.id] = agent.time_alive
            if agent.alive == False:
                self.dead_agents += 1
                self.list_agents.remove(agent)

    def update_pred_fitness(self):
        for agent in self.list_agents:
            self.agents[agent.id] = agent.score * 10
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
        
        return (agent.view_angle/agent.rotation_speed, ) + tuple(agent.lines_stats)    
    
    def game_over(self ):
        """Checks if the game is over.
        """
        if self.list_agents == []:
            return True
        else:
            return False




def simulate_agent(agent, game):

    MAIN_SURFACE = game.MAIN_SURFACE

    agents = [agent]
    with open("prey.pkl", "rb") as f:
        genome = pickle.load(f)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "prey_conf.txt")

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    info = GameInfo(game.cells.list, agents)
    game.painter.add(agent)

    # Game main loop
    show = False
    run = True
    while(run):
        
        # clock.tick(60)
        
        for e in pygame.event.get():

            if(e.type == pygame.QUIT):
                pygame.quit()
                quit()  
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_q:
                    run = False
                    return agent.time_alive
                if e.key == pygame.K_c:
                    show = not show
        int = info.get_agent_inputs(agent)
        out = net.activate(int)
        dec = out.index(max(out))

        if dec == 0:   
            agent.move(rotate_left=1)
        elif dec == 1:
            agent.move(rotate_right=1)
        elif dec == 2:
            agent.move(stay=1)

        if show : 
            print("time ",agent.time_alive)
            print("score ",agent.score)
            print("energy ",agent.energy)
            print("mass ",agent.mass)
            print("speed ",agent.speed)


        agent.update()

        agent.collisionDetection(game.cells, [])
        #cam.update(agent)
        # MAIN_SURFACE.fill((242,251,255))
        # Uncomment next line to get dark-theme

        # MAIN_SURFACE.fill((0,0,0))
        # game.painter.paint()
        # # Start calculating next frame
        # pygame.display.flip()


        if agent.alive == False:
            run = False
            return agent.time_alive

# game = Game()
# agent = Agent(game.MAIN_SURFACE, game.grid, "GeoVas", font=font, PLATFORM_HEIGHT=game.PLATFORM_HEIGHT, PLATFORM_WIDTH=game.PLATFORM_WIDTH)




gene_space = {
    'mass': [5, 20],
    'speed': [1, 10],
    'view_angle': [10, 180]
}

def fitness_func(individual):
    mass, speed, view_angle = individual
    game = Game()
    agent = Agent(game.MAIN_SURFACE, game.grid, "GeoVas", font=font, PLATFORM_HEIGHT=game.PLATFORM_HEIGHT, PLATFORM_WIDTH=game.PLATFORM_WIDTH, mass=mass, speed=speed, view_angle=view_angle)
    return simulate_agent(agent, game)

def create_population(pop_size, gene_space):
    return [
        [random.uniform(min(gene_space[gene]), max(gene_space[gene])) for gene in gene_space]
        for _ in range(pop_size)
    ]

def evaluate_population(population):
    return [fitness_func(individual) for individual in population]

def select_parents(population, fitnesses, num_parents):
    fitness_sum = np.sum(fitnesses)
    probabilities = np.array(fitnesses) / fitness_sum
    parent_indices = np.random.choice(len(population), size=num_parents, p=probabilities, replace=False)
    parents = [population[i] for i in parent_indices]
    return parents

def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual, mutation_rate, gene_space):
    gene_keys = list(gene_space.keys())
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            gene = gene_keys[i]
            individual[i] = random.uniform(min(gene_space[gene]), max(gene_space[gene]))
    return individual

def genetic_algorithm(generations, pop_size, gene_space, mutation_rate, num_parents):
    population = create_population(pop_size, gene_space)
    best_fitnesses = []

    for generation in range(generations):
        fitnesses = evaluate_population(population)
        best_fitnesses.append(max(fitnesses))
        parents = select_parents(population, fitnesses, num_parents)
        next_population = parents.copy()

        while len(next_population) < pop_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            next_population.append(mutate(child1, mutation_rate, gene_space))
            if len(next_population) < pop_size:
                next_population.append(mutate(child2, mutation_rate, gene_space))

        population = next_population
        print(f"Generation {generation + 1}: Best Fitness = {max(fitnesses)}: Best Solution = {population[np.argmax(fitnesses)]}")

    best_solution_idx = np.argmax(fitnesses)
    best_solution = population[best_solution_idx]
    return best_solution, best_fitnesses

# GA parameters
generations = 50
pop_size = 10
mutation_rate = 0.3
num_parents = 5

# Run the GA
best_solution, best_fitnesses = genetic_algorithm(generations, pop_size, gene_space, mutation_rate, num_parents)
print(f"Best solution: {best_solution}")

# Plot the fitness evolution
plt.plot(best_fitnesses, label='Best Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness Evolution Over Generations')
plt.legend()
plt.show()





# simulate_agent(agent, game)

# # CMA-ES implementation
# class CMA_ES:
#     def __init__(self, mean, sigma, popsize, lower_bounds, upper_bounds):
#         self.mean = np.array(mean)
#         self.sigma = sigma
#         self.popsize = popsize
#         self.num_params = len(mean)
#         self.cov = np.identity(self.num_params)
#         self.weights = np.log(self.popsize / 2 + 1) - np.log(np.arange(1, self.popsize + 1))
#         self.weights /= np.sum(self.weights)
#         self.mu_eff = 1 / np.sum(self.weights ** 2)
#         self.cumcov = 2 / ((self.num_params + 1.3) ** 2 + self.mu_eff)
#         self.cs = (self.mu_eff + 2) / (self.num_params + self.mu_eff + 5)
#         self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.num_params + 1)) - 1) + self.cs
#         self.ps = np.zeros(self.num_params)
#         self.pc = np.zeros(self.num_params)
#         self.cc = 4 / (self.num_params + 4)
#         self.ccov = 2 / ((self.num_params + 1.3) ** 2 + self.mu_eff)
#         self.lower_bounds = np.array(lower_bounds)
#         self.upper_bounds = np.array(upper_bounds)

#     def ask(self):
#         solutions = np.array([self.mean + self.sigma * np.random.multivariate_normal(np.zeros(self.num_params), self.cov) for _ in range(self.popsize)])
#         # Clamp solutions to [0, 1] range
#         return np.clip(solutions, 0, 1)

#     def transform(self, solutions):
#         # Transform the solutions to fit within the specified ranges
#         return self.lower_bounds + (self.upper_bounds - self.lower_bounds) * solutions

#     def tell(self, solutions, fitnesses):
#         sorted_indices = np.argsort(fitnesses)[::-1]
#         sorted_solutions = np.array(solutions)[sorted_indices]
#         sorted_weights = self.weights[sorted_indices]
#         new_mean = np.sum(sorted_solutions.T * sorted_weights, axis=1)
#         self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * np.dot(np.linalg.cholesky(self.cov).T, (new_mean - self.mean) / self.sigma)
#         hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (1 / self.popsize))) < (1.4 + 2 / (self.num_params + 1)) * np.sqrt(self.mu_eff)
#         self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * (new_mean - self.mean) / self.sigma
#         y = (sorted_solutions - self.mean) / self.sigma
#         self.cov = (1 - self.ccov) * self.cov + self.ccov * np.dot(y.T, np.diag(self.weights)).dot(y) + (1 - hsig) * self.cc * (2 - self.cc) * self.cov
#         self.sigma *= np.exp((np.linalg.norm(self.ps) - np.sqrt(1 - (1 - self.cs) ** (2 * (1 / self.popsize)))) / self.damps)
#         self.mean = new_mean

# # Parameters for CMA-ES
# initial_mean = [0.5, 0.5, 0.5]  # Initial guess in normalized form [0, 1]
# sigma = 0.5
# popsize = 20
# max_generations = 50

# # Define the ranges for each gene
# lower_bounds = [5, 2, 10]    # Lower bounds for mass, speed, and view_angle
# upper_bounds = [20, 10, 180] # Upper bounds for mass, speed, and view_angle

# # Fitness function
# def fitness(solution):
#     mass, speed, view_angle = solution
#     game = Game()
#     agent = Agent(game.MAIN_SURFACE, game.grid, "GeoVas", font=font, PLATFORM_HEIGHT=game.PLATFORM_HEIGHT, PLATFORM_WIDTH=game.PLATFORM_WIDTH, mass=mass, speed=speed, view_angle=view_angle)
#     time_alive = simulate_agent(agent, game)
#     return time_alive

# # Running CMA-ES
# cma_es = CMA_ES(initial_mean, sigma, popsize, lower_bounds, upper_bounds)
# for generation in range(max_generations):
#     print(f"Generation {generation}")
#     solutions = cma_es.ask()
#     transformed_solutions = cma_es.transform(solutions)
#     fitnesses = [fitness(solution) for solution in transformed_solutions]
#     cma_es.tell(solutions, fitnesses)
#     best_fitness = max(fitnesses)
#     best_solution = transformed_solutions[np.argmax(fitnesses)]
#     print(f"Generation {generation}: Best fitness = {best_fitness}")
#     print(f"Generation {generation}: Best solution = {best_solution}")

# # Output the best solution found
# print(f"Best solution found: {best_solution}")


# # Define gene space for each attribute you want to evolve
# gene_space = {
#     'mass': [5, 20],           # Possible values for mass
#     'speed': [1, 10],          # Possible values for speed
#     'view_angle': [10, 180]     # Possible values for view angle
# }

# # Define the fitness function
# def fitness_func(ga_instance, solution, solution_idx):
#     mass, speed, view_angle = solution
#     game = Game()
#     agent = Agent(game.MAIN_SURFACE, game.grid, "GeoVas", font=font, PLATFORM_HEIGHT=game.PLATFORM_HEIGHT, PLATFORM_WIDTH=game.PLATFORM_WIDTH, mass=mass, speed=speed, view_angle=view_angle)
#     time_alive = simulate_agent(agent, game)
#     return time_alive

# # Genetic Algorithm parameters
# ga_instance = pygad.GA(
#     num_generations=100,
#     num_parents_mating=10,
#     fitness_func=fitness_func,
#     sol_per_pop=50,
#     mutation_probability=0.3,
#     num_genes=3,  # number of attributes to evolve
#     init_range_low=0,
#     init_range_high=1,
#     gene_space=[
#         gene_space['mass'],
#         gene_space['speed'],
#         gene_space['view_angle']
#     ]
# )

# # Run the GA
# ga_instance.run()

# # Get the best solution
# solution, solution_fitness, _ = ga_instance.best_solution()
# print(f"Best solution: {solution}, Fitness: {solution_fitness}")


# # Plot the fitness evolution
# plt.plot(ga_instance.best_solutions_fitness, label='Best Fitness')
# plt.xlabel('Generation')
# plt.ylabel('Fitness')
# plt.title('Fitness Evolution Over Generations')
# plt.legend()
# plt.show()

