from osim.env import ProstheticsEnv 
from arguments import parser, print_args
import numpy as np
from functions import OUNoise, NormalizedEnv, Memory, Critic, Actor, DDPGagent, make_env, graph_reward, penalties, velocity, shape_rew, step_jor
import torch

args = parser()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = ProstheticsEnv(visualize=True)
env.change_model(model='3D', prosthetic=True, difficulty=0, seed=None)
# phi = lambda x: np.array(x).astype(np.float32, copy=False)
# obs_size = np.asarray(env.observation_space.shape).prod()
# action_space = env.action_space
# action_size = np.asarray(action_space.shape).prod()

# #Critic Network
# q_func = q_functions.FCSAQFunction(
#             160, 
#             action_size,
#             n_hidden_channels=args.critic_hidden_units,
#             n_hidden_layers=args.critic_hidden_layers)

# #Policy Network
# pi = policy.FCDeterministicPolicy(
#             160, 
#             action_size=action_size,
#             n_hidden_channels=args.actor_hidden_units,
#             n_hidden_layers=args.actor_hidden_layers,
#             min_action=action_space.low, 
#             max_action=action_space.high,
#             bound_action=True)

# #Model
# model = DDPGModel(q_func=q_func, policy=pi)
# opt_actor = optimizers.Adam(alpha=args.actor_lr)
# opt_critic = optimizers.Adam(alpha=args.critic_lr)
# opt_actor.setup(model['policy'])
# opt_critic.setup(model['q_function'])
# opt_actor.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_a')
# opt_critic.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_c')
# rbuf = replay_buffer.ReplayBuffer(args.replay_buffer_size)
# ou_sigma = (action_space.high - action_space.low) * 0.2

# #Explorer
# explorer = explorers.AdditiveOU(sigma=ou_sigma)

# #Agent
# agent = DDPG(model, opt_actor, opt_critic, rbuf, gamma=args.gamma,
#                  explorer=explorer, replay_start_size=args.replay_start_size,
#                  target_update_method=args.target_update_method,
#                  target_update_interval=args.target_update_interval,
#                  update_interval=args.update_interval,
#                  soft_update_tau=args.soft_update_tau,
#                  n_times_update=args.number_of_update_times,
#                  phi=phi,minibatch_size=args.minibatch_size
#             )

# Cargar el modelo guardado
agent = DDPGagent(env)
noise = OUNoise(env.action_space)
agent.load_checkpoint("ep_best.pth.tar")

for i in range(args.test_episodes):
    obs = env.reset()
    done = False
    reward_Alan = 0
    t = 0
    state = env.reset(project=True)
    while not done and t < 200:
        # env.render()
        # action = agent.update(obs)
        # obs, r, done, x = env.step(action)
        # reward_Alan += r
        # t += 1

        action = agent.get_action(state)
        action = noise.get_action(action, i)
        new_state, reward, done, _ = env.step(action)#step_jor(env, action)# 
        #state = torch.Tensor([new_state]).to(device)
        reward_Alan += reward
        t += 1
    print('Test episode:', i, 'Reward obtained: ', reward_Alan)
    #agent.stop_episode()