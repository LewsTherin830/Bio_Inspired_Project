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
import visualise
# Initialize Pygame
pygame.init()


pygame.display.set_caption("prey and predator")


MAX_CELL = 150
ancestors_prey = {}
old_gen_prey = {}
prey_fitness = {}

ancestors_pred = {}
old_gen_pred = {}
pred_fitness = {}



class Prey :
    def __init__(self):
        self.game = Game(screen_width=1400, screen_height=800)

    def train(self, genomes, predators, config):
        preys = []
        preys_net = []

        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            mass, speed, view_angle, rotation_speed = old_gen_prey[genome_id][0]
            agent = AG(self.game.MAIN_SURFACE,self.game.grid, id = genome_id, PLATFORM_HEIGHT=self.game.PLATFORM_HEIGHT, PLATFORM_WIDTH=self.game.PLATFORM_WIDTH, mass=mass, speed=speed, view_angle=view_angle, rotation_speed=rotation_speed)
            preys.append(agent)
            self.game.painter.add(agent)
            preys_net.append(net)

        preds = []
        preds_net = []
        for pred in range(1, len(predators)):
            mass, speed, view_angle, rotation_speed = old_gen_pred[predators[pred][0]][0]
            preds.append(PRED(self.game.MAIN_SURFACE, self.game.grid, id=predators[pred][0], PLATFORM_HEIGHT=self.game.PLATFORM_HEIGHT, PLATFORM_WIDTH=self.game.PLATFORM_WIDTH, mass=mass, speed=speed, view_angle=view_angle, rotation_speed=rotation_speed))
            self.game.painter.add(preds[pred-1])
            p = neat.nn.FeedForwardNetwork.create(predators[pred][1], predators[0])
            preds_net.append(p)

        self.game_info = GameInfo(self.game.number_of_cells, preys)
        self.game.painter.add(agent)
        show = False
        run = True
        clock = pygame.time.Clock()

        children = []
        threshhold = 40        
        runing_time = 0
        while run:
            alive = []
            alive_net = []
            preds_alive = []
            preds_alive_net = []
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



            
            for agent in preys:
                if not agent.alive:
                    continue
                if agent.score > threshhold:
                    agent.time_alive = 1000
                    agent.alive = False
                    continue
                inputs = self.game_info.get_agent_inputs(agent)  # Get inputs for the current agent
                output = net.activate(inputs)  # Get output from neural network
                # Process output and control agent's actions based on the output
                decision = output.index(max(output))

                if decision == 0:
                    agent.move(rotate_left=1)
                elif decision == 1:
                    agent.move(rotate_right=1)
                else:
                    agent.move(stay=1)


                agent.update()
                agent.collisionDetection(self.game.cells, preds)

                # if agent.score // 10 - 1 == agent.children and agent.score != 0:
                #     agent.children += 1
                #     child = AG(agent.surface, agent.grid, name=agent.name, id=agent.id, PLATFORM_WIDTH=agent.PLATFORM_WIDTH, PLATFORM_HEIGHT=agent.PLATFORM_HEIGHT, font=agent.font, mass=agent.mass, speed=agent.speed, view_angle=agent.angle, rotation_speed=agent.rotation_speed)
                #     child.x = agent.x
                #     child.y = agent.y
                #     children.append(child)
                #     self.game.painter.add(child)

                if agent.alive:
                    alive.append(agent)
                    alive_net.append(net)
            
            preys = alive
            preys_net = alive_net



            for pred in range(0,len(preds)):
                if not preds[pred].alive:
                    continue
                int = self.game_info.get_agent_inputs(preds[pred])
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
                preds[pred].collisionDetection(preys)

                if preds[pred].alive:
                    preds_alive.append(preds[pred])
                    preds_alive_net.append(preds_net[pred])
            
            preds = preds_alive
            preds_net = preds_alive_net

            self.game_info.update_agents_fitness()

            if self.game.cells.count < self.game.number_of_cells // 2:
                self.game.cells.add_cell(self.game.number_of_cells - self.game.cells.count)
            

            # self.game.painter.paint()
            # pygame.display.flip()

            # self.game.MAIN_SURFACE.fill((242,251,255))
            if show:
                clock.tick(30)
                self.game.painter.paint()
                pygame.display.flip()

                self.game.MAIN_SURFACE.fill((242,251,255))


            if self.game_info.game_over() :
                self.calculate_fitness(genomes, self.game_info)

                run = False
                break


    def calculate_fitness_genome(self, genome, genome_id, game_info):
        genome.fitness += game_info.get_agent_fitness(genome_id)

    def calculate_fitness(self, genomes, game_info):
        for id,genome in genomes:
            genome.fitness += game_info.get_agent_fitness(id)
            genome.fitness =  max(0, genome.fitness)









class Pred :
    def __init__(self):
        self.game = Game(screen_width=1400, screen_height=800)


    def train(self, genomes, preys, config):
        predators = []
        preds_net = []

        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            mass, speed, view_angle, rotation_speed = old_gen_pred[genome_id][0]
            agent = PRED(self.game.MAIN_SURFACE,self.game.grid, id = genome_id, PLATFORM_HEIGHT=self.game.PLATFORM_HEIGHT, PLATFORM_WIDTH=self.game.PLATFORM_WIDTH, mass=mass, speed=speed, view_angle=view_angle, rotation_speed=rotation_speed)
            predators.append(agent)
            self.game.painter.add(agent)
            preds_net.append(net)
            
        prey = []
        prey_net = []
        for pr in range(1, len(preys)):
            mass, speed, view_angle, rotaion_speed = old_gen_prey[preys[pr][0]][0]
            prey.append(AG(self.game.MAIN_SURFACE,self.game.grid, id=preys[pr][0], PLATFORM_HEIGHT=self.game.PLATFORM_HEIGHT, PLATFORM_WIDTH=self.game.PLATFORM_WIDTH, mass=mass, speed=speed, view_angle=view_angle, rotation_speed=rotaion_speed))
            self.game.painter.add(prey[pr-1])
            p = neat.nn.FeedForwardNetwork.create(preys[pr][1], preys[0])
            prey_net.append(p)

        self.game_info = GameInfo(self.game.number_of_cells, predators)
        self.game.painter.add(agent)
        show = False
        run = True
        clock = pygame.time.Clock()

        children = []        
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
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        if show == True:
                            show = False
                        else:
                            show = True
                            print("score : ", predators[0].score), print("energy : ", predators[0].energy)



            # clock.tick(60)
            for agent in predators:
                if not agent.alive:
                    continue
                inputs = self.game_info.get_agent_inputs(agent)  # Get inputs for the current agent
                output = net.activate(inputs)  # Get output from neural network
            # Process output and control agent's actions based on the output
                decision = output.index(max(output))

                if decision == 0:
                    agent.move(rotate_left=1)
                elif decision == 1:
                    agent.move(rotate_right=1)
                elif decision == 2:
                    agent.move(dash=1)
                else:
                    agent.move(stay=1)


                agent.collisionDetection(prey + children)
                agent.update()
                if agent.alive:
                    preds_alive.append(agent)
                    preds_alive_net.append(net)
            
            predators = preds_alive
            preds_net = preds_alive_net


            for pr in range(0,len(prey)):
                if not prey[pr].alive:
                    continue
                int = self.game_info.get_agent_inputs(prey[pr])
                out = prey_net[pr].activate(int)
                dec = out.index(max(out))

                if dec == 0:   
                    prey[pr].move(rotate_left=1)
                elif dec == 1:
                    prey[pr].move(rotate_right=1)
                else:
                    prey[pr].move(stay=1)


                prey[pr].update()
                prey[pr].collisionDetection(self.game.cells, predators)

                if prey[pr].score == 20 and prey[pr].children == 0:
                    prey[pr].children += 1
                    child = Agent(prey[pr].surface, prey[pr].grid, name=prey[pr].name, id=prey[pr].id, PLATFORM_WIDTH=prey[pr].PLATFORM_WIDTH, PLATFORM_HEIGHT=prey[pr].PLATFORM_HEIGHT, font=prey[pr].font, mass=prey[pr].mass, speed=prey[pr].speed, view_angle=prey[pr].angle, rotation_speed=prey[pr].rotation_speed)
                    child.x = prey[pr].x
                    child.y = prey[pr].y
                    children.append(child)
                    self.game.painter.add(child)

                if prey[pr].score == 30 and prey[pr].children == 1:
                    prey[pr].children += 1
                    child = Agent(prey[pr].surface, prey[pr].grid, name=prey[pr].name, id=prey[pr].id, PLATFORM_WIDTH=prey[pr].PLATFORM_WIDTH, PLATFORM_HEIGHT=prey[pr].PLATFORM_HEIGHT, font=prey[pr].font, mass=prey[pr].mass, speed=prey[pr].speed, view_angle=prey[pr].angle, rotation_speed=prey[pr].rotation_speed)
                    child.x = prey[pr].x
                    child.y = prey[pr].y
                    children.append(child)
                    self.game.painter.add(child)

                if prey[pr].alive:
                    alive.append(prey[pr])
                    alive_net.append(prey_net[pr])

            prey = alive
            prey_net = alive_net



            self.game_info.update_pred_fitness()

            if self.game.cells.count < self.game.number_of_cells:
                self.game.cells.add_cell(self.game.number_of_cells - self.game.cells.count)
            


            # self.game.painter.paint()
            # pygame.display.flip()

            # self.game.MAIN_SURFACE.fill((242,251,255))
            if show:
                clock.tick(60)
                self.game.painter.paint()
                pygame.display.flip()

                self.game.MAIN_SURFACE.fill((242,251,255))


            if self.game_info.game_over():
                self.calculate_fitness(genomes, self.game_info)
                run = False
                break


    def calculate_fitness_genome(self, genome, genome_id, game_info):
        genome.fitness += game_info.get_agent_fitness(genome_id)

    def calculate_fitness(self, genomes, game_info):
        for id,genome in genomes:
            genome.fitness += game_info.get_agent_fitness(id)
            genome.fitness =  max(0, genome.fitness)  # Update fitness based on game info







def simulate():

    # with open("old_preys.pkl", "rb") as f:
    #     preys_load = pickle.load(f)
    #     f.close()
    # with open("old_preds.pkl", "rb") as f:
    #     predators_load = pickle.load(f)
    #     f.close()

    # with open("prey_gen200.pkl", "rb") as f:
    #     preys_load = pickle.load(f)
    #     f.close()

    # with open("pred_gen200.pkl", "rb") as f:
    #     predators_load = pickle.load(f)
    #     f.close()

    with open("prey_hall_of_fame copy.pkl", "rb") as f:
        preys_load = pickle.load(f)
        f.close()
    with open("predator_hall_of_fame copy.pkl", "rb") as f:
        predators_load = pickle.load(f)
        f.close()

    preys = preys_load["genomes"]
    predators = predators_load["genomes"]
    prey_genes = preys_load["genes"]
    pred_genes = predators_load["genes"]
    predators = predators[:10]
    preys = preys[:15]

    game = Game(screen_width=1400)

    preds_net = []
    preds = []
    prey = []
    prey_net = []


    for pr in range(1, len(preys)):
        mass, speed, view_angle, rotation_speed = prey_genes[preys[pr][0]][0]
        prey.append(AG(game.MAIN_SURFACE,game.grid, id=pr, PLATFORM_HEIGHT=game.PLATFORM_HEIGHT, PLATFORM_WIDTH=game.PLATFORM_WIDTH, mass=mass, speed=speed, view_angle=view_angle, rotation_speed=rotation_speed))
        game.painter.add(prey[pr-1])
        p = neat.nn.FeedForwardNetwork.create(preys[pr][1], preys[0])
        prey_net.append(p)



    for pred in range(1, len(predators)):
        mass, speed, view_angle, rotation_speed = pred_genes[predators[pred][0]][0]
        preds.append(PRED(game.MAIN_SURFACE,game.grid, id=pred, PLATFORM_HEIGHT=game.PLATFORM_HEIGHT, PLATFORM_WIDTH=game.PLATFORM_WIDTH, mass=mass, speed=speed, view_angle=view_angle, rotation_speed=rotation_speed))
        game.painter.add(preds[pred-1])
        p = neat.nn.FeedForwardNetwork.create(predators[pred][1], predators[0])
        preds_net.append(p)


    run = True
    clock = pygame.time.Clock()
    game_info = GameInfo(game.number_of_cells, prey)
    
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
        clock.tick(30)

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


                if prey[pr].alive:
                    alive.append(prey[pr])
                    alive_net.append(prey_net[pr])

        prey = alive
        prey_net = alive_net

        game.painter.paint()
        pygame.display.flip()

        game.MAIN_SURFACE.fill((242,251,255))

        if game.cells.count < game.number_of_cells:
            game.cells.add_cell(game.number_of_cells - game.cells.count)
            

        # if game_info.game_over() :
        #     print("Game Over")
        #     run = False
        #     break



def save_best_genomes(genomes, type,  n = 10):
    if type == "pred" :
        m = n * 3 + 1
        filename = "old_preds.pkl"
        hall_filename = "predator_hall_of_fame.pkl"
        genes = old_gen_pred

    elif type == "prey":
        m = n * 4 + 1
        filename = "old_preys.pkl"
        hall_filename = "prey_hall_of_fame.pkl"
        genes = old_gen_prey

    with open(filename, "rb") as f:
        load = pickle.load(f)
        f.close()
    old_ones = load["genomes"]

    with open(hall_filename, "rb") as f:
        load = pickle.load(f)
        f.close()
    hall_of_fame = load["genomes"]
    
    sorted_genomes = sorted(genomes, key=lambda g: g[1].fitness, reverse=True)
    old_ones = old_ones[0:1] + sorted_genomes[:n] + old_ones[1:]
    hall_of_fame = hall_of_fame + old_ones[1:]
    hall_of_fame = hall_of_fame[0:1] + sorted(hall_of_fame[1:], key=lambda g: (g[1].fitness, g[0]), reverse=True)[:n]

    if len(old_ones) > m:
        old_ones = old_ones[:m]

    with open(hall_filename, "wb") as f:
        pickle.dump({"genomes": hall_of_fame, "genes": genes}, f)
        f.close()
    with open(filename, "wb") as f:
        pickle.dump({"genomes": old_ones, "genes": genes}, f)
        f.close()


def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1[0]) - 1)
    child1_genes = parent1[0][:crossover_point] + parent2[0][crossover_point:]
    child1_sigmas = parent1[1][:crossover_point] + parent2[1][crossover_point:]
    child2_genes = parent2[0][:crossover_point] + parent1[0][crossover_point:]
    child2_sigmas = parent2[1][:crossover_point] + parent1[1][crossover_point:]
    child1 = (child1_genes, child1_sigmas)
    child2 = (child2_genes, child2_sigmas)
    return random.choice([child1, child2])

def mutate(individual, mutation_rate, gene_space, parent_fitness, type):
    genes, sigmas = individual
    n = len(genes)
    tau_prime = 1.0 / np.sqrt(2 * n)
    tau = 1.0 / np.sqrt(2 * np.sqrt(n))
    epsilon = 1e-8

    new_genes = []
    new_sigmas = []

    for i, gene_key in enumerate(gene_space.keys()):
        sigma = sigmas[i]

        sigma_prime = sigma * np.exp(tau_prime * np.random.normal() + tau * np.random.normal())
        if sigma_prime < epsilon:
            sigma_prime = epsilon

        gene_value_prime = genes[i] + sigma_prime * np.random.normal()

        # Clamp gene_value_prime to stay within the range specified in gene_space
        min_val, max_val = gene_space[gene_key]
        gene_value_prime = max(min_val, min(max_val, gene_value_prime))

        new_genes.append(gene_value_prime)
        new_sigmas.append(sigma_prime)

    return (new_genes, new_sigmas)


def create_genes(genomes, type):
    global old_gen_prey, ancestors_prey, old_gen_pred, ancestors_pred 
    gene_space = {
    'mass': [5, 30],
    'speed': [1, 15],
    'view_angle': [10, 180],
    'rotation_speed': [10, 60]
        }

    if type == "prey": 
        old_gen = old_gen_prey
        anc = ancestors_prey
    else:
        old_gen = old_gen_pred
        anc = ancestors_pred
    
    if old_gen == {}:
        for id,_ in genomes:
            genes = [random.uniform(min(gene_space[gene]), max(gene_space[gene])) for gene in gene_space]
            sigma = [random.uniform(0.1, 0.5) for _ in range(len(genes))]
            old_gen[id] = (genes,sigma)
        return old_gen
    
    else:
        for id, _ in genomes:
            if id in old_gen:
                continue
            else:
                parent_id = anc[id]
                child = crossover(old_gen[parent_id[0]], old_gen[parent_id[1]])
                child = mutate(child,0.3, gene_space, parent_id, type)
                old_gen[id] = child
        return old_gen

                    

def eval_prey(genomes, config):
    global old_gen_prey, ancestors_prey, prey_fitness, max_fitness
    with open("old_preds.pkl", "rb") as f:
        load = pickle.load(f)
        f.close()
    try:
        old_preds = load["genomes"]
    except:
        old_preds = []


    # Create populations of prey and predator agents
    full_genes = create_genes(genomes, "prey")
    max_fitness = 0

    if len(old_preds) < 30:
        train_genes = []
        for  _, prey_genome in genomes:
            prey_genome.fitness = 0 
            
        for i in range(0,len(genomes),10):
            try:
                train_genes.append([genomes[j] for j in range(i, i+10)])
            except:
                train_genes.append([genomes[i]])

        for prey in train_genes:
            preys = Prey()
            preys.train(prey, old_preds, config)
    else:
        preds1 = old_preds[0:1] + random.sample(old_preds[1: len(old_preds)//3], 3)
        preds2 = old_preds[0:1] + random.sample(old_preds[len(old_preds)//3: 2*len(old_preds)//3], 3)
        preds3 = old_preds[0:1] + random.sample(old_preds[2*len(old_preds)//3:], 3)
        train = [preds1, preds2, preds3]
        fitnesses = [0 for _ in range(len(genomes))]
        for k in range(3):
            train_genes = []
            for  _, prey_genome in genomes:
                prey_genome.fitness = 0 
                
            for i in range(0,len(genomes),10):
                try:
                    train_genes.append([genomes[j] for j in range(i, i+10)])
                except:
                    train_genes.append([genomes[i]])

            for prey in train_genes:
                preys = Prey()
                preys.train(prey, train[k], config)
            for l, (id, genome) in enumerate(genomes):
                fitnesses[l]+=genome.fitness
        for l, (id, genome) in enumerate(genomes):
            genome.fitness = fitnesses[l]/3

    max_fitness = max([(id,genome.fitness) for id, genome in genomes])
    prey_fitness = {id: (genome.fitness, max_fitness[1]) for id, genome in genomes}

    save_best_genomes(genomes, 'prey',  20)


def eval_pred(genomes, config):
    global old_gen_pred, ancestors_pred, pred_fitness, max_fitness
    with open("old_preys.pkl", "rb") as f:
        load = pickle.load(f)
        f.close()
    try:
        old_preys = load["genomes"]
    except:
        old_preys = []
        
    # Create populations of prey and predator agents
    full_genes = create_genes(genomes, "pred")
    max_fitness = 0

    if len(old_preys) < 30:
        train_genes = []
        for  _, pred_genome in genomes:
            pred_genome.fitness = 0 
            
        for i in range(0,len(genomes),2):
            try:
                train_genes.append([genomes[j] for j in range(i, i+2)])
            except:
                train_genes.append([genomes[i]])

        for pred in train_genes:
            preds = Pred()
            preds.train(pred, old_preys, config)
    else:
        preys1 = old_preys[0:1] + random.sample(old_preys[len(old_preys)//4: 2*len(old_preys)//4], 10)
        preys2 = old_preys[0:1] + random.sample(old_preys[2*len(old_preys)//4: 3*len(old_preys)//4], 10)
        preys3 = old_preys[0:1] + random.sample(old_preys[3*len(old_preys)//4:], 10)
        train = [preys1, preys2, preys3]
        fitnesses = [0 for _ in range(len(genomes))]
        for k in range(3):
            train_genes = []
            for  _, pred_genome in genomes:
                pred_genome.fitness = 0 
                
            for i in range(0,len(genomes),2):
                try:
                    train_genes.append([genomes[j] for j in range(i, i+2)])
                except:
                    train_genes.append([genomes[i]])

            for pred in train_genes:
                preds = Pred()
                preds.train(pred, train[k], config)
            for l, (id, genome) in enumerate(genomes):
                fitnesses[l]+=genome.fitness
        for l, (id, genome) in enumerate(genomes):
            genome.fitness = fitnesses[l]/3

    max_fitness = max([(id,genome.fitness) for id, genome in genomes])
    pred_fitness = {id: (genome.fitness, max_fitness[1]) for id, genome in genomes}

    save_best_genomes(genomes, 'pred',  10)


def copy_file(source_file, destination_file):
    with open(source_file, 'rb') as f_source:
        with open(destination_file, 'wb') as f_destination:
            # Read from the source file and write to the destination file
            for line in f_source:
                f_destination.write(line)




def eval_hall_of_fame():
    with open("prey_hall_of_fame.pkl", "rb") as f:
        load = pickle.load(f)
        f.close()
    prey = load["genomes"]
    prey_genes = load["genes"]

    with open("predator_hall_of_fame.pkl", "rb") as f:
        load = pickle.load(f)
        f.close()
    pred = load["genomes"]
    pred_genes = load["genes"]

    with open("old_preys.pkl", "rb") as f:
        load = pickle.load(f)
        f.close()
    old_preys = load["genomes"] + prey[1:]

    with open("old_preds.pkl", "rb") as f:
        load = pickle.load(f)
        f.close()
    old_preds = load["genomes"] + pred[1:]

    # test predators  
    preys1 = old_preys[0:1] + random.sample(old_preys[len(old_preys)//4: 2*len(old_preys)//4], 10)
    preys2 = old_preys[0:1] + random.sample(old_preys[2*len(old_preys)//4: 3*len(old_preys)//4], 10)
    preys3 = old_preys[0:1] + random.sample(old_preys[3*len(old_preys)//4:], 10)
    train = [preys1, preys2, preys3]
    fitnesses = [0 for _ in range(len(pred[1:]))]
    for k in range(3):
        train_genes = []
        for  _, pred_genome in pred[1:]:
            pred_genome.fitness = 0 
            
        for i in range(0,len(pred[1:]),2):
            try:
                train_genes.append([pred[1:][j] for j in range(i, i+2)])
            except:
                train_genes.append([pred[1:][i]])

        for predators in train_genes:
            preds = Pred()
            preds.train(predators, train[k], pred[0])
        for l, (id, genome) in enumerate(pred[1:]):
            fitnesses[l]+=genome.fitness
    for l, (id, genome) in enumerate(pred[1:]):
        genome.fitness = fitnesses[l]/3


    # test preys
    preds1 = old_preds[0:1] + random.sample(old_preds[1: len(old_preds)//3], 3)
    preds2 = old_preds[0:1] + random.sample(old_preds[len(old_preds)//3: 2*len(old_preds)//3], 3)
    preds3 = old_preds[0:1] + random.sample(old_preds[2*len(old_preds)//3:], 3)
    train = [preds1, preds2, preds3]
    fitnesses = [0 for _ in range(len(prey[1:]))]
    for k in range(3):
        train_genes = []
        for  _, prey_genome in prey[1:]:
            prey_genome.fitness = 0 
            
        for i in range(0,len(prey[1:]),10):
            try:
                train_genes.append([prey[1:][j] for j in range(i, i+10)])
            except:
                train_genes.append([prey[1:][i]])

        for prey_portion in train_genes:
            preys = Prey()
            preys.train(prey_portion, train[k], prey[0])
        for l, (id, genome) in enumerate(prey[1:]):
            fitnesses[l]+=genome.fitness
    for l, (id, genome) in enumerate(prey[1:]):
        genome.fitness = fitnesses[l]/3

    with open("predator_hall_of_fame.pkl", "wb") as f:
        pickle.dump({"genomes": pred, "genes": pred_genes}, f)
        f.close()
    
    with open("prey_hall_of_fame.pkl", "wb") as f:
        pickle.dump({"genomes": prey, "genes": prey_genes}, f)
        f.close()
    save_best_genomes(pred[1:], 'pred',  10)
    save_best_genomes(prey[1:], 'prey',  20)






# Run NEAT training

def run_neat():
    global old_gen_prey, ancestors_prey, old_gen_pred, ancestors_pred 
    local_dir = os.path.dirname(__file__)
    prey_config_path = os.path.join(local_dir, "prey_conf.txt")
    pred_config_path = os.path.join(local_dir, "predator_conf.txt")

    prey_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                     prey_config_path)
    
    pred_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                     pred_config_path)
    


    with open("old_preds.pkl", "wb") as f:
        pickle.dump({"genomes":[pred_config]}, f)

    with open("old_preys.pkl", "wb") as f:
        pickle.dump({"genomes":[prey_config]}, f)

    with open("predator_hall_of_fame.pkl", "wb") as f:
        pickle.dump({"genomes":[pred_config]}, f)

    with open("prey_hall_of_fame.pkl", "wb") as f:
        pickle.dump({"genomes":[prey_config]}, f)

    # Create and run NEAT populations for prey and predator
    prey_population = neat.Population(prey_config)
    # prey_population = neat.Checkpointer.restore_checkpoint("prey-checkpoint-182")
    prey_population.add_reporter(neat.StdOutReporter(True))
    prey_stats = neat.StatisticsReporter()
    prey_population.add_reporter(prey_stats)

    pred_population = neat.Population(pred_config)
    # pred_population = neat.Checkpointer.restore_checkpoint("pred-checkpoint-182")
    pred_population.add_reporter(neat.StdOutReporter(True))
    pred_stats = neat.StatisticsReporter()
    pred_population.add_reporter(pred_stats)

    prey_best_fitness = []
    prey_mean_fitness = []
    pred_best_fitness = []
    pred_mean_fitness = []

    for i in range(201):
        print("\n******** Prey Generation: ", i)
        best_prey = prey_population.run(eval_prey, 1)
        prey_best_fitness.append(best_prey.fitness)
        prey_mean_fitness.append(prey_stats.get_fitness_mean()[-1])
        print("Best Prey fitness: ", max_fitness[1])
        print("Best Prey Genome: ", old_gen_prey[max_fitness[0]])
        ancestors_prey = prey_population.reproduction.ancestors 
        


        print("\n******** Predator Generation: ", i)
        best_pred = pred_population.run(eval_pred, 1)
        pred_best_fitness.append(best_pred.fitness)
        pred_mean_fitness.append(pred_stats.get_fitness_mean()[-1])
        print("Best Pred fitness: ", max_fitness[1])
        print("Best Pred Genome: ", old_gen_pred[max_fitness[0]])

        ancestors_pred = pred_population.reproduction.ancestors

        if i % 20 == 0 and i != 0:
            start_time = time.time()
            eval_hall_of_fame()
            end_time = time.time()
            print(f"Time taken by eval_hall_of_fame: {end_time - start_time} seconds")

        try:
            # Plotting the fitness data on the same graph
            plt.figure(figsize=(10, 5))

            plt.plot(prey_best_fitness, label='Best Fitness (Prey)', color='green', linestyle='-')
            plt.plot(prey_mean_fitness, label='Mean Fitness (Prey)', color='green', linestyle='--')
            plt.plot(pred_best_fitness, label='Best Fitness (Predator)', color='red', linestyle='-')
            plt.plot(pred_mean_fitness, label='Mean Fitness (Predator)', color='red', linestyle='--')

            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('Fitness Evolution of Prey and Predators Over Generations')
            plt.legend()
            # Save the plot
            plt.savefig(f'fitness_evolution.png')

            # Plotting the fitness data of prey
            plt.figure(figsize=(10, 5))

            plt.plot(prey_best_fitness, label='Best Fitness (Prey)', color='green', linestyle='-')
            plt.plot(prey_mean_fitness, label='Mean Fitness (Prey)', color='green', linestyle='--')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('Fitness Evolution of Over Generations')
            plt.legend()
            # Save the plot
            plt.savefig(f'fitness_evolution_of_prey.png')
            plt.close("all")  # Close the figure to free up memory
            
            # Plotting the fitness data of predators
            plt.figure(figsize=(10, 5))

            plt.plot(pred_best_fitness, label='Best Fitness (Predator)', color='red', linestyle='-')
            plt.plot(pred_mean_fitness, label='Mean Fitness (Predator)', color='red', linestyle='--')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('Fitness Evolution of Over Generations')
            plt.legend()
            # Save the plot
            plt.savefig(f'fitness_evolution_of_predators.png')

            plt.close("all")  # Close the figure to free up memory

        except Exception as e:
            print(f"Error plotting fitness data: {e}")
            continue

        copy_file('old_preys.pkl', f'prey_gen{i}.pkl')
        copy_file('old_preds.pkl', f'pred_gen{i}.pkl')

    
if __name__ == "__main__":

    
    # run_neat()

    simulate()    
    
#visualize neural network of prey or predator i in the hall of fame
    # with open("prey_hall_of_fame copy.pkl", "rb") as f:
    #     load = pickle.load(f)
    #     f.close()
    # pred = load["genomes"]
    # pred_genes = load["genes"]
    # conf = pred[0]
    # i = 10
    # game = Game()

    # MAIN_SURFACE = game.MAIN_SURFACE
    # mass , speed, view_angle, rotation_speed = pred_genes[pred[i][0]][0]
    # print("mass : ", mass, "speed : ", speed, "view_angle : ", view_angle, "rotation_speed : ", rotation_speed)
    # predator = AG(MAIN_SURFACE, game.grid,  "GeoVas",font=font, PLATFORM_HEIGHT=game.PLATFORM_HEIGHT, PLATFORM_WIDTH=game.PLATFORM_WIDTH, mass=mass, speed=speed, view_angle=view_angle, rotation_speed=rotation_speed)
    # game.painter.add(predator)
    # visualise.draw_net(conf, pred[i][1], True)
    # predator.energy_depletion_rate = 0
    # while True:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             quit()
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_a:
    #                 predator.move(rotate_left=1)
    #             if event.key == pygame.K_d:
    #                 predator.move(rotate_right=1)
    #             if event.key == pygame.K_SPACE:
    #                 predator.dx = 0
    #                 predator.dy = 0
                    
    #     clock.tick(30)
    #     predator.update()

    #     MAIN_SURFACE.fill((242,251,255))
    #     game.painter.paint()
    #     pygame.display.flip()





