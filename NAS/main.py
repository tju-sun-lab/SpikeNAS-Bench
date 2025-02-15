import time

import numpy as np
import utils
from get_train_log import get_train_log
import config
from evolution_algorithm import initialize_population, evaluate_fitness, generate_offspring, enviromental_selection
from utils import *
import logging
import sys
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def seed_torch(seed=640):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

start = time.time()
args = config.get_args()
seed_torch(args.seed)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir, exist_ok=True)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(
    os.path.join(args.save_dir, 'search_{}_log_{}.txt').format(args.fitness_evaluator, time.strftime("%Y%m%d-%H%M%S")))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.getLogger().setLevel(logging.INFO)

for arg, val in args.__dict__.items():
    logging.info(arg + '.' * (60 - len(arg) - len(str(val))) + str(val))

population = []
fitness = []
test_acc = []
search_time_total = 0
initialized_populaiton = initialize_population(args.population_size)
population.append(initialized_populaiton)
for iter in range(args.generation_number):
    print(iter)
    evaluated_fitness, search_time = evaluate_fitness(args, population[iter], args.fitness_evaluator)
    search_time_total += search_time
    fitness.append(evaluated_fitness)
    offspring = generate_offspring(args.population_size, population[iter], evaluated_fitness, args.pc, args.pm)
    offspring_fitness, search_time = evaluate_fitness(args, offspring, args.fitness_evaluator)
    search_time_total += search_time
    new_population = enviromental_selection(args, args.population_size, population[iter], evaluated_fitness, offspring, offspring_fitness)
    population.append(new_population)
last_fitness, search_time = evaluate_fitness(args, population[-1], args.fitness_evaluator)
search_time_total += search_time
fitness.append(last_fitness)

fitness_last = fitness[-1]
population_last = population[-1]
max_idx = np.array(fitness_last).argmax()
best_arch = population_last[max_idx]
best_id = utils.encoding_to_id(best_arch)
_, _, model_param, _, _, val_acc, _, test_acc, _, _ = get_train_log(best_id, '../data/CIFAR10')
logging.info("[architecture_{}] [params = {}] [val_acc = {:.3f}] [test_acc = {:.3f}]"
                 .format(best_id, model_param, val_acc[-1], test_acc))
end = time.time()
search_time_with_SNAS = end - start
search_time_total = search_time_total + end - start
hour = search_time_total // 3600
minute = (search_time_total - hour * 3600) // 60
second = search_time_total - hour * 3600 - minute * 60
logging.info("total search time seconds: {}".format(search_time_total))
logging.info("total search time: {}h {}m {}s".format(hour, minute, second))
logging.info("search time with SNAS: {}s".format(search_time_with_SNAS))














