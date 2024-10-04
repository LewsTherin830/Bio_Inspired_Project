import neat.checkpoint
import pygame
from game import *
import neat
import os
import pickle


pygame.init()


pygame.display.set_caption("Pong")
FPS  = 60




class Prey :
    def __init__(self, num_agents):
        self.game = Game()
        self.agents = [Agent(self.game.MAIN_SURFACE,self.game.camera,id = i) for i in range(num_agents)]
        self.game_info = GameInfo(self.game.cells.list, self.agents)

    def train_ai(self, genomes, config):
        nets = []
        for _, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append(net)

        run = True
        clock = pygame.time.Clock()

        while run:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            for i, agent in enumerate(self.agents):
                inputs = self.game.get_agent_inputs(agent)  # Get inputs for the current agent
                output = nets[i].activate(inputs)  # Get output from neural network
                # Process output and control agent's actions based on the output
                decision = output.index(max(output))
                if decision == 0:
                    agent.move(up = 1)
                elif decision == 1:
                    agent.move(down = 1)
                elif decision == 2:
                    agent.move(left = 1)
                elif decision == 3:
                    agent.move(right = 1)
                agent.update()
                agent.collisionDetection(game.cells.list)
                game.painter.paint()
                pygame.display.flip()
            MAIN_SURFACE.fill((242,251,255))


            if self.game_info.game_over(self.game):
                self.calculate_fitness(genomes, self.game_info)
                run = False
                break

    def calculate_fitness(self, genomes, game_info):
        for genome_id, genome in genomes:
            genome.fitness += game_info.get_agent_fitness(genome)  # Update fitness based on game info


def eval_genomes(genomes, config):
    WIDTH, HEIGHT = 800, 600
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    num_agents = 10

    prey = Prey(num_agents)
    prey.train_ai(genomes, config)

def run_neat(config):
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 10)

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)




if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    
    run_neat(config)
    #test_ai(config)

    # with open("winner.pkl", "rb") as f:
    #     winner = pickle.load(f)

    # visualise.draw_net(config, winner, True)