import argparse

def parser():
    parser = argparse.ArgumentParser(description='Parameter Summary')

    parser.add_argument('--num_episodes',default=2000,help='Number of episodes')
    parser.add_argument('--max_episode_length',default=1000,help='Maximum episode length')

    parser.add_argument('--critic_lr', default=1e-3, help='Critic learning rate')
    parser.add_argument('--critic_hidden_layers', default=3, help='Critic Hidden Layers)
    parser.add_argument('--critic_hidden_units', default=300, help='Critic Hidden Units')

    parser.add_argument('--actor_lr', default=1e-4, help='Actor learning rate')
    parser.add_argument('--actor_hidden_layers', default=3, help='Actor Hidden Layers)
    parser.add_argument('--actor_hidden_units', default=300, help='Actor Hidden Units')

    parser.add_argument('--gamma', default=0.995, help='Discount factor')
    parser.add_argument('--minibatch_size', default=128, help='Batch size')

    parser.add_argument('replay_buffer_size', default=5 * 10 ** 5, help='the size of the replay buffer')
    parser.add_argument('replay_start_size', default=5000, help='the size of the replay buffer when the network starts the training step')
    parser.add_argument('number_of_update_times', default=1, help='Number of repetition of update')

    parser.add_argument('target_update_interval', default=1, help='Target update interval in each step')
    parser.add_argument('target_update_method', default='soft', help='the type of update: hard or soft')

    parser.add_argument('soft_update_tau', default=1e-2, help='The value of Tau  in the soft target update')
    parser.add_argument('update_interval', default=4, help='Model update interval in each step')
    parser.add_argument('number_of_eval_runs', default==100)

    # eval_interval=10 ** 5
    # final_exploration_steps=10 ** 6       

    return parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))
