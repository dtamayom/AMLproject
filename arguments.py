import argparse

def parser():
    parser = argparse.ArgumentParser(description='Parameter Summary')

    #General parameters
    parser.add_argument('--num_episodes', default=20000,help='Number of episodes')
    parser.add_argument('--render_environment', default=True, help='Show ALAN (or not)')
    parser.add_argument('--checkpoint_dir', default='./final_model/', help='Location to save models')
    parser.add_argument('--graphs_folder', default='graphs_final', help='Location to save resulting graphs')
    parser.add_argument('--mode_test', default=True, help='Choose False to train, True to test')
    parser.add_argument('--minibatch_size', default=200, help='Batch size')
    parser.add_argument('--num_steps', default=1000, help='Number of step sizes')
    parser.add_argument('--v_obj', default=3, help='Target vel to calculate loss')
    parser.add_argument('--use_v_obj', default=False, help='Target vel or not')

    #Ornstein-Ulhenbeck Process explorer parameters
    parser.add_argument('--mu', default=0.0,help='mu')
    parser.add_argument('--theta', default=0.15,help='theta')
    parser.add_argument('--max_sigma', default=0.3,help='maximum sigma')
    parser.add_argument('--min_sigma', default=0.3,help='minimum sigma')
    parser.add_argument('--decay_period', default=100000,help='decay_period')

    #DDPGagent
    parser.add_argument('--hidden_size', default=300, help='Hidden size')
    parser.add_argument('--actor_lr', default=1e-4, help='Actor learning rate')
    parser.add_argument('--critic_lr', default=1e-3, help='Critic learning rate')
    parser.add_argument('--gamma', default=0.99, help='Discount factor')
    parser.add_argument('--tau', default=1e-2, help='The value of Tau in the soft target update')
    parser.add_argument('--max_memory', default=50000, help='The value of Tau in the soft target update')

    #Test
    parser.add_argument('--test_episodes', default=15)

    # eval_interval=10 ** 5
    # final_exploration_steps=10 ** 6       

    return parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))
