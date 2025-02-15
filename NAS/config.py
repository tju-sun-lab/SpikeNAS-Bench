import argparse


def get_args():
    parser = argparse.ArgumentParser("NAS")
    parser.add_argument('--exp_name', type=str, default='NAS',  help='experiment name')
    parser.add_argument('--save_dir', type=str, default='./search_result', help='path to the result')
    parser.add_argument('--dataset', type=str, default='cifar10', help='[cifar10]')
    parser.add_argument('--seed', default=640, type=int)
    parser.add_argument('--device', default='cuda:0')

    parser.add_argument('--timestep', type=int, default=5, help='timestep for SNN')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=64, help='epoch')
    parser.add_argument('--tau', type=float, default=4/3, help='neuron decay time factor')
    parser.add_argument('--threshold_low', type=float, default=0.5, help='neuron firing threshold')
    parser.add_argument('--threshold_middle', type=float, default=1.0, help='neuron firing threshold')
    parser.add_argument('--threshold_high', type=float, default=1.5, help='neuron firing threshold')
    parser.add_argument('--celltype', type=str, default='forward', help='[forward, backward]')
    parser.add_argument('--second_avgpooling', type=int, default=2, help='momentum')

    parser.add_argument('--heads', default=6)
    parser.add_argument('--depth', default=4)
    parser.add_argument('--dim', default=80)

    parser.add_argument('--population_size', type=int, default=10)
    parser.add_argument('--generation_number', type=int, default=10)
    parser.add_argument('--fitness_evaluator', type=str, default='early_stop_40', help='early_stop_10, early_stop_40, early_stop_60')
    parser.add_argument('--pc', type=float, default=0.9, help=' the probabity for crossover operation')
    parser.add_argument('--pm', type=float, default=0.2, help=' the probabity for mutation operation')


    args = parser.parse_args()
    print(args)

    return args
