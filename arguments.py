import argparse

def parser():
    parser = argparse.ArgumentParser(description='Parameter Summary')

    parser.add_argument('--numep',default=1000,help='Number of episodes')
    parser.add_argument('--obs_size',default=160,help='Observation size')
    parser.add_argument('--hidd_lay',default=4,help='Hidden layers')
    parser.add_argument('--c_hidd_lay',default=200,help='C_Hidden layers')
    parser.add_argument('--actor_lr',default=0.0001,help='Actor learning rate')
    parser.add_argument('--critic_lr',default=0.0001,help='Critic learning rate')
    parser.add_argument('--gamma',default=0.995,help='Gamma')

    return parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))
