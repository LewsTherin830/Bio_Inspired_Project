import neat.checkpoint
import neat.config
import neat.threaded
import pygame
from game import *
from Agent_test import Agent as AG
from Agent_test import Predator as PRED
import neat
import os
import pickle
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
# Initialize Pygame
pygame.init()


pygame.display.set_caption("prey and predator")


MAX_CELL = 150

def plot_CIAO(CIAO):
    plt.imshow(CIAO, cmap='gray_r', interpolation='nearest')
    plt.title("CIAO Matrix")
    plt.xlabel("Prey Generation")
    plt.ylabel("Predator Generation")
    plt.savefig('CIAO_matrix_withthreshhold.png')

def simulate():
    CIAO = np.zeros((100,100))

    for gen in range(100,200):
        file_prey = f"prey_gen{gen}.pkl"
        with open(file_prey, "rb") as f:
            preys_load = pickle.load(f)
            f.close()

        preys_genomes = preys_load["genomes"]
        prey_genes = preys_load["genes"]
        preys_genomes = preys_genomes[:21]
        prey_conf = preys_genomes[0]

        preys_genomes = sorted(preys_genomes[1:], key=lambda g: (g[1].fitness, g[0]), reverse=True)

        for new_gen in range(100,200):
            print(f"Prey Generation {gen} : Predator Generation {new_gen}")
            file_pred = f"pred_gen{gen}.pkl"
            with open(file_pred, "rb") as f:
                predators_load = pickle.load(f)
                f.close()


            predators = predators_load["genomes"]
            pred_genes = predators_load["genes"]
            pred_conf = predators[0]
            predators = predators[:11]
            predators =  sorted(predators[1:11], key=lambda g: (g[1].fitness, g[0]), reverse=True)

            the_preys = preys_genomes[:8]
            the_preds = predators[:4]

            game = Game(screen_width=1000)

            preds_net = []
            preds = []
            prey = []
            prey_net = []


            for pr in range(len(the_preys)):
                mass, speed, view_angle, rotation_speed = prey_genes[the_preys[pr][0]][0]
                prey.append(Agent(game.MAIN_SURFACE,game.grid, id=pr, PLATFORM_HEIGHT=game.PLATFORM_HEIGHT, PLATFORM_WIDTH=game.PLATFORM_WIDTH, mass=mass, speed=speed, view_angle=view_angle, rotation_speed=rotation_speed))
                game.painter.add(prey[pr])
                p = neat.nn.FeedForwardNetwork.create(the_preys[pr][1], prey_conf)
                prey_net.append(p)



            for pred in range(0, len(the_preds)):
                mass, speed, view_angle, rotation_speed = pred_genes[the_preds[pred][0]][0]
                preds.append(Predator(game.MAIN_SURFACE,game.grid, id=pred, PLATFORM_HEIGHT=game.PLATFORM_HEIGHT, PLATFORM_WIDTH=game.PLATFORM_WIDTH, mass=mass, speed=speed, view_angle=view_angle, rotation_speed=rotation_speed))
                game.painter.add(preds[pred])
                p = neat.nn.FeedForwardNetwork.create(the_preds[pred][1], pred_conf)
                preds_net.append(p)


            run = True
            clock = pygame.time.Clock()
            game_info = GameInfo(game.number_of_cells, prey)
            successfull_prey = 0
            runing_time = 0
            while run:
                runing_time +=1
                alive = []
                alive_net = []
                preds_alive = []
                preds_alive_net = []
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit()

                for pred in range(0,len(preds)):
                    if not preds[pred].alive:
                        continue
                    int = game_info.get_agent_inputs(preds[pred])
                    out = preds_net[pred].activate(int)
                    dec = out.index(max(out))

                    if dec == 0:   
                        preds[pred].move(rotate_left=1)
                    elif dec == 1:
                        preds[pred].move(rotate_right=1)
                    elif dec == 2:
                        preds[pred].move(dash=1)
                    else:
                        preds[pred].move(stay=1)



                    preds[pred].update()
                    preds[pred].collisionDetection(prey)

                    if preds[pred].alive:
                        preds_alive.append(preds[pred])
                        preds_alive_net.append(preds_net[pred])
                
                preds = preds_alive
                preds_net = preds_alive_net

                for pr in range(0,len(prey)):
                    if not prey[pr].alive:
                        continue
                    else:
                        int = game_info.get_agent_inputs(prey[pr])
                        out = prey_net[pr].activate(int)
                        dec = out.index(max(out))

                        if dec == 0:   
                            prey[pr].move(rotate_left=1)
                        elif dec == 1:
                            prey[pr].move(rotate_right=1)
                        elif dec == 2:
                            prey[pr].move(stay=1)


                        prey[pr].update()
                        prey[pr].collisionDetection(game.cells, preds)

                        if prey[pr].score > 40:
                            successfull_prey += 1
                            prey[pr].alive = False

                        if prey[pr].alive:
                            alive.append(prey[pr])
                            alive_net.append(prey_net[pr])

                prey = alive
                prey_net = alive_net

                # game.painter.paint()
                # pygame.display.flip()

                # game.MAIN_SURFACE.fill((242,251,255))

                if game.cells.count < game.number_of_cells:
                    game.cells.add_cell(game.number_of_cells - game.cells.count)
                    

                if len(prey) == 0 or len(preds) == 0:
                    if len(prey) <1:
                        CIAO[new_gen - 100 ][gen - 100 ] = 1 - successfull_prey/8
                    # else:
                    #     CIAO[new_gen ][gen ] = max(0, 1 - len(prey)/4)
                    run = False
        plot_CIAO(CIAO)
    plot_CIAO(CIAO)


simulate()

