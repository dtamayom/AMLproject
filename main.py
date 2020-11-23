##Codigo tomado de https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
import torch.nn as nn
import torch
import gc
import torch.nn.functional as F
import torch.autograd
import torch.optim as optim
from torch.autograd import Variable
import random
from collections import deque
import numpy as np
import gym
from osim.env import ProstheticsEnv
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
from gym import spaces 
import sys
import os
from arguments import parser, print_args
import pdb
from functions import OUNoise, NormalizedEnv, Memory, Critic, Actor, DDPGagent, make_env, graph_reward, penalties, velocity, shape_rew, step_jor, reward_first, get_observation

cuda = True if torch.cuda.is_available() else False

args = parser()
print_args(args)


env = make_env(test=args.mode_test,render=args.render_environment)
#env = NormalizedEnv(env)
obs_size = np.asarray(env.observation_space.shape).prod()
#action_size = np.asarray(action_space.shape).prod()

agent = DDPGagent(env)
noise = OUNoise(env.action_space)
batch_size = args.minibatch_size
rewards = []
avg_rewards = []
best_reward=-1000

for episode in range(1, args.num_episodes+ 1):
    state = env.reset(project=True)
    noise.reset()
    episode_reward = 0
    
    for step in range(500):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        new_state, reward, done, _ = env.step(action)#env.step(action) 
        reward = shape_rew(env)
        agent.memory.push(state, action, reward, new_state, done)
        if len(agent.memory) > batch_size:
            agent.update(batch_size)        
        
        state = new_state
        episode_reward += reward

        if done:
            sys.stdout.write("Episode: {}, Reward: {}, Average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))
    
    if episode%10==0:
        print('Velocity: ', velocity(env))

    if episode_reward>best_reward:
        best_reward=episode_reward
        if not os.path.exists(args.checkpoint_dir):
           os.makedirs(args.checkpoint_dir)
        agent.save_checkpoint("best")
        print('new best', episode)

    # Save the model every 100 episode.
    if episode%100==0:
        if not os.path.exists(args.checkpoint_dir):
           os.makedirs(args.checkpoint_dir)
        agent.save_checkpoint(episode)

    #generate graph of rewards vs episodes
    if episode%50==0: 
        if not os.path.exists(args.graphs_folder):
           os.makedirs(args.graphs_folder)
        graph_reward(rewards, episode, '_DDPGargs_')    
    
    
print('Good job Alan')

plt.plot(rewards, color='cadetblue')
plt.ylabel('Returns')
plt.xlabel('Number of episodes')
plt.savefig(os.path.join(args.graphs_folder,"Episodes_VS_Returns.png"))

plt.plot(avg_rewards, color='darkgoldenrod')
plt.ylabel('Average of Returns ')
plt.xlabel('Number of episodes')
plt.savefig(os.path.join(args.graphs_folder,"AverageEpisodes_VS_Returns.png"))
