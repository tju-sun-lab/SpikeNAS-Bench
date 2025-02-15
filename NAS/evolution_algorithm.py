import numpy as np
import torch

from utils import *
from get_train_log import get_train_log
import random
from evaluator import *
from model_snn_withoutcupy import SNASNet
from foresight.pruners import predictive
from foresight.dataset import *

def check_validity(coding):
    count_vector = coding.ravel()
    con_mat = np.zeros((4, 4))
    position = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    flag2 = 1
    flag3 = 1

    for num, (k0, k1) in enumerate(position):
        con_mat[k0, k1] = count_vector[num]

    neigh2_cnts = con_mat @ con_mat
    neigh3_cnts = neigh2_cnts @ con_mat
    neigh4_cnts = neigh3_cnts @ con_mat
    connection_graph = con_mat + neigh2_cnts + neigh3_cnts + neigh4_cnts

    for node in range(3):
        if connection_graph[node, 3] == 0:  # if any node doesnt send information to the last layer, remove it
            flag2 = 0
        if flag2 == 0: return -1

    for node in range(3):
        if connection_graph[0, node + 1] == 0:  # if any node doesnt get information from the input layer, remove it
            flag3 = 0
        if flag3 == 0: return -1

    # 保证node1到node4一定有连接
    if con_mat[0, 3] == 0: return -1  # ensure direct connection between input=>output for fast information propagation

    return con_mat



def initialize_population(population_size):
    initialized_population = []
    while(len(initialized_population) < population_size):
        individual = np.random.randint(0, 5, size=6)
        flag = check_validity(individual)
        if type(flag) == int:
            continue
        else:
            initialized_population.append(individual)

    return initialized_population

def evaluate_fitness(args, population, evaluator):
    evaluated_fitness = []
    search_time = 0
    if evaluator == 'early_stop_test':
        for idv in population:
            idv_id = encoding_to_id(idv)
            _, _, _, _, _, _, _, test_acc, _, seconds = get_train_log(idv_id, '../data/CIFAR10')
            evaluated_fitness.append(test_acc)
            search_time += seconds

    elif evaluator == 'early_stop_10':
        for idv in population:
            idv_id = encoding_to_id(idv)
            _, _, _, _, _, val_acc, _, _, _, seconds = get_train_log(idv_id, '../data/CIFAR10')
            evaluated_fitness.append(val_acc[-1])
            search_time += seconds * 0.1

    elif evaluator == 'early_stop_40':
        for idv in population:
            idv_id = encoding_to_id(idv)
            _, _, _, _, _, val_acc, _, _, _, seconds = get_train_log(idv_id, '../data/CIFAR10')
            evaluated_fitness.append(val_acc[39])
            search_time += seconds * 0.4

    elif evaluator == 'early_stop_60':
        for idv in population:
            idv_id = encoding_to_id(idv)
            _, _, _, _, _, val_acc, _, _, _, seconds = get_train_log(idv_id, '../data/CIFAR10')
            evaluated_fitness.append(val_acc[59])
            search_time += seconds * 0.6



    return evaluated_fitness, search_time


def generate_offspring(population_size, parent, fitness, pc, pm):
    offspring_cross = []
    offspring = []
    while(len(offspring_cross) < population_size):
        p1 = p2 = np.array([0, 0, 0, 0, 0, 0])
        while((p1 == p2).all()):
            index1 = random.sample(range(0, population_size), 2)
            p1_1 = fitness[index1[0]]
            p1_2 = fitness[index1[1]]
            if p1_1 > p1_2:
                p1 = parent[index1[0]]
            else:
                p1 = parent[index1[1]]

            index2 = random.sample(range(0, population_size), 2)
            p2_1 = fitness[index2[0]]
            p2_2 = fitness[index2[1]]
            if p2_1 > p2_2:
                p2 = parent[index2[0]]
            else:
                p2 = parent[index2[1]]

        r = random.random()
        if r < pc:
            cross_point = np.random.randint(1, 6, 1)[0]
            o1 = np.concatenate((p1[0: cross_point], p2[cross_point:]))
            o2 = np.concatenate((p2[0: cross_point], p1[cross_point:]))
            flag1 = check_validity(o1)
            flag2 = check_validity(o2)
            if type(flag1) == int or type(flag2) == int:
                continue
            else:
                offspring_cross.append(o1)
                offspring_cross.append(o2)
        else:
            offspring_cross.append(p1)
            offspring_cross.append(p2)

    for idv in offspring_cross:
        r = random.random()
        if r < pm:
            flag = -1
            while(type(flag) == int):
                idv_copy = idv.copy()
                mutation_point = np.random.randint(0, 6, 1)[0]
                mutation_operation = np.random.randint(0, 5, 1)[0]
                idv_copy[mutation_point] = mutation_operation
                flag = check_validity(idv_copy)
            offspring.append(idv_copy)
        else:
            offspring.append(idv)

    return offspring

def enviromental_selection(args, population_size, parent, parent_fit, offspring, offspring_fit):
    old_population = parent + offspring
    old_fit = parent_fit + offspring_fit
    idx_max = np.argmax(old_fit)
    p_best = old_population[idx_max]

    new_population = []
    while(len(new_population) < population_size):
        index = random.sample(range(0, len(old_population)), 2)
        p1 = old_fit[index[0]]
        p2 = old_fit[index[1]]
        if p1 > p2:
            p = old_population[index[0]]
            del old_population[index[0]]
            del old_fit[index[0]]
        else:
            p = old_population[index[1]]
            del old_population[index[1]]
            del old_fit[index[1]]
        new_population.append(p)

    new_fit, _ = evaluate_fitness(args, new_population, args.fitness_evaluator)

    is_exist = False
    for idv in new_population:
        if (idv == p_best).all():
            is_exist = True
            break
    if not is_exist:
        idx_min = np.argmin(new_fit)
        new_population[idx_min] = p_best

    return new_population









