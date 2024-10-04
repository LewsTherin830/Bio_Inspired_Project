import neat.checkpoint
import pygame
from game import *
import neat
import os
import pickle
from predator import Predator as the_predator


pygame.init()


pygame.display.set_caption("prey and predator")


MAX_CELL = 100



class Prey :
    def __init__(self):
        self.game = Game()


    def test_ai(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        agent = Predator(self.game.MAIN_SURFACE, id = 0, PLATFORM_HEIGHT=self.game.PLATFORM_HEIGHT, PLATFORM_WIDTH=self.game.PLATFORM_WIDTH)
        self.game_info = GameInfo(self.game.number_of_cells, [agent])
        self.game.painter.add(agent)
        run = True
        clock = pygame.time.Clock()
        runing_time = 0
        # agent.energy_depletion_rate = 0
        while run:
            runing_time +=1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
            clock.tick(60)
            inputs = self.game_info.get_agent_inputs(agent)
            output = net.activate(inputs)
            decision = output.index(max(output))

            if decision == 0:
                agent.move(rotate_left=1)
            elif decision == 1:
                agent.move(rotate_right=1)
            else:
                agent.move(dash_forward=1)

            # if decision == 0:
            #     agent.move(down = 1)
            # elif decision == 1:
            #     agent.move(left = 1)
            # elif decision == 2:
            #     agent.move(right = 1)
            # elif decision == 3:
            #     agent.move(up = 1)
            # elif decision == 4:
            #     agent.move(up_left=1)
            # elif decision == 5:
            #     agent.move(up_right=1)
            # elif decision == 6:
            #     agent.move(down_left=1)
            # elif decision == 7:
                agent.move(down_right=1)
                            
            agent.update()
            self.game_info.update_agents_fitness()
            agent.collisionDetection(self.game.cells.list)

            if runing_time % 50 == 0:
                self.game.cells.add_cell(2)
            self.game.painter.paint()
            pygame.display.flip()

            self.game.MAIN_SURFACE.fill((242,251,255))
            print(agent.energy)

            if self.game_info.game_over():
                print("Game Over")
                run = False
                break


    def train_ai(self, genomes, config):
        nets = []
        self.agents = [Predator(self.game.MAIN_SURFACE,id = genome_id,PLATFORM_HEIGHT=self.game.PLATFORM_HEIGHT, PLATFORM_WIDTH=self.game.PLATFORM_WIDTH) for genome_id, _ in genomes]
        for agent in self.agents:
            self.game.painter.add(agent)
        self.game_info = GameInfo(self.game.number_of_cells, self.agents)

        for _, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append(net)

        run = True
        clock = pygame.time.Clock()
        runing_time = 0
        while run:
            runing_time +=1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            for i, agent in enumerate(self.agents):
                inputs = self.game_info.get_agent_inputs(agent)  # Get inputs for the current agent
                output = nets[i].activate(inputs)  # Get output from neural network
                # Process output and control agent's actions based on the output
                decision = output.index(max(output))

                if decision == 0:
                    agent.move(down = 1)
                elif decision == 1:
                    agent.move(left = 1)
                elif decision == 2:
                    agent.move(right = 1)
                elif decision == 3:
                    agent.move(up = 1)
                elif decision == 4:
                    agent.move(up_left=1)
                elif decision == 5:
                    agent.move(up_right=1)
                elif decision == 6:
                    agent.move(down_left=1)
                elif decision == 7:
                    agent.move(down_right=1)


                agent.update()
                self.game_info.update_agents_fitness()
                agent.collisionDetection(self.game.cells.list)
                if runing_time % 50 == 0:
                    self.game.cells.add_cell(2)

                self.game.painter.paint()
                pygame.display.flip()

            self.game.MAIN_SURFACE.fill((242,251,255))


            if self.game_info.game_over():
                self.calculate_fitness(genomes, self.game_info)
                run = False
                break


    def divide_and_train(self, genome, genome_id, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        agent = the_predator(self.game.MAIN_SURFACE, self.game.grid, id = genome_id, PLATFORM_HEIGHT=self.game.PLATFORM_HEIGHT, PLATFORM_WIDTH=self.game.PLATFORM_WIDTH)
        self.game_info = GameInfo(self.game.number_of_cells, [agent])
        self.game.painter.add(agent)

        run = True
        clock = pygame.time.Clock()
        fitness_threshold = 200

        runing_time = 0
        show = False
        while run:
            runing_time +=1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        if show == True:
                            show = False
                        else:
                            show = True
                            print("score : ", agent.score), print("energy : ", agent.energy)

            # clock.tick(60)
            inputs = self.game_info.get_agent_inputs(agent)  # Get inputs for the current agent
            output = net.activate(inputs)  # Get output from neural network
            # Process output and control agent's actions based on the output
            decision = output.index(max(output))

            if decision == 0:
                agent.move(rotate_left=1)
            elif decision == 1:
                agent.move(rotate_right=1)
            else:
                agent.move(dash_forward=1)



            agent.update()
            self.game_info.update_agents_fitness()
            agent.collisionDetection(self.game.cells.list)
            if runing_time % 100 == 0 and self.game.cells.count < MAX_CELL:
                self.game.cells.add_cell(2)

            # self.game.painter.paint()
            # pygame.display.flip()

            # self.game.MAIN_SURFACE.fill((242,251,255))
            if show:
                self.game.painter.paint()
                pygame.display.flip()

                self.game.MAIN_SURFACE.fill((242,251,255))


            if self.game_info.game_over() or agent.score >= fitness_threshold:
                self.calculate_fitness_genome(genome, genome_id, self.game_info)
                run = False
                break


    def calculate_fitness_genome(self, genome, genome_id, game_info):
        genome.fitness += game_info.get_agent_fitness(genome_id)

    def calculate_fitness(self, genomes, game_info):
        for id,genome in genomes:
            genome.fitness += game_info.get_agent_fitness(id)  # Update fitness based on game info


def eval_genomes(genomes, config):
    for id,genome in genomes:
        genome.fitness = 0

    for genome_id, genome in genomes:
        prey = Prey()
        prey.divide_and_train(genome, genome_id, config)

    # prey= Prey()
    # prey.train_ai(genomes, config)

def run_neat(config):
    # p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-350")
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 200)

    with open("predator.pkl", "wb") as f:
        pickle.dump(winner, f)


def test_ai(config):
    with open("predator.pkl", "rb") as f:
        winner = pickle.load(f)

    WIDTH, HEIGHT = 1000, 800
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))


    prey = Prey()
    prey.test_ai(winner, config )

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "predator_conf.txt")

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    
    run_neat(config)
    # test_ai(config)


    # with open("winner.pkl", "rb") as f:
    #     winner = pickle.load(f)

    # visualise.draw_net(config, winner, True)