from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import logging
import sys
import gym                       
#gym.undo_logger_setup()  # NOQA
from gym import spaces  
import gym.wrappers
import numpy as np   
import matplotlib   
matplotlib.use('Agg')         
import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import os

from osim.env import ProstheticsEnv   # Environment con la prótesis
import chainer                               
from chainer import optimizers               
from chainerrl.agents.ddpg import DDPG      
from chainerrl.agents.ddpg import DDPGModel  
from chainerrl import explorers              
from chainerrl import misc                  
from chainerrl import policy                 
from chainerrl import q_functions           
from chainerrl import replay_buffer          
from arguments import parser, print_args
import pdb 

seed=0
gpu=0   # GPU device id

#Get argument values and print them
args = parser()
print_args(args)

#Helper functions
def clip_action_filter(a):
    #limit actions between highest and lowest action in space
    return np.clip(a, action_space.low, action_space.high)

def reward_filter(r):
    #Reward value scaled
    return r *1 #1e-2


# def phi(obs):
#     obs=np.array(obs)
#     return obs.astype(np.float32)

phi = lambda x: np.array(x).astype(np.float32, copy=False) #COnverir datatype de la observación


def random_action():
    a = action_space.sample()
    if isinstance(a, np.ndarray):
        a = a.astype(np.float32)
    return a


def make_env(test,render=False):
        
    env = ProstheticsEnv(visualize=render)
    env.change_model(model='3D', prosthetic=True, difficulty=0, seed=None)
    # Use different random seeds for train and test envs
    env_seed = 2 ** 32 - 1 - seed if test else seed
    env.seed(env_seed)
    if isinstance(env.action_space, spaces.Box):
        misc.env_modifiers.make_action_filtered(env, clip_action_filter)
    if not test:
        misc.env_modifiers.make_reward_filtered(env, reward_filter)
    if render and not test:
        misc.env_modifiers.make_rendered(env)
    return env

def graph_reward(reward, eps, saveas):
    name = saveas + str(eps) + '.png'
    episodes = np.linspace(0,eps,eps)
    plt.figure()
    plt.plot(episodes,reward,'cadetblue',label='DDPG')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(os.path.join('graphs',name))
    plt.close()

def penalties(env):
    state_desc = env.get_state_desc()
    penalty = 0.
    penalty += (state_desc["body_vel"]["pelvis"][0] - 3.0) ** 2
    penalty += (state_desc["body_vel"]["pelvis"][2]) ** 2
    penalty += np.sum(np.array(env.osim_model.get_activations()) ** 2) * 0.001
    if state_desc["body_pos"]["pelvis"][1] < 0.70: #falling
        penalty += 10  

    return penalty

# Set a random seed used in ChainerRL
misc.set_random_seed(seed)

# Setup the environment
env = make_env(test=False,render=False)
obs_size = np.asarray(env.observation_space.shape).prod()
action_space = env.action_space
action_size = np.asarray(action_space.shape).prod()

# Critic Network
q_func = q_functions.FCSAQFunction(
            160, 
            action_size,
            n_hidden_channels=args.critic_hidden_units,
            n_hidden_layers=args.critic_hidden_layers)

# Policy Network
pi = policy.FCDeterministicPolicy(
            160, 
            action_size=action_size,
            n_hidden_channels=args.actor_hidden_units,
            n_hidden_layers=args.actor_hidden_layers,
            min_action=action_space.low, 
            max_action=action_space.high,
            bound_action=True)

# The Model
model = DDPGModel(q_func=q_func, policy=pi)
opt_actor = optimizers.Adam(alpha=args.actor_lr)
opt_critic = optimizers.Adam(alpha=args.critic_lr)
opt_actor.setup(model['policy'])
opt_critic.setup(model['q_function'])
opt_actor.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_a')
opt_critic.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_c')

rbuf = replay_buffer.ReplayBuffer(args.replay_buffer_size)
ou_sigma = (action_space.high - action_space.low) * 0.2

#Possible explorers
explorer = explorers.AdditiveOU(sigma=ou_sigma)
#explorer = chainerrl.explorers.ConstantEpsilonGreedy(epsilon=0.2, random_action_func=env.action_space.sample)

# The agent
agent = DDPG(model, opt_actor, opt_critic, rbuf, gamma=args.gamma,
                 explorer=explorer, replay_start_size=args.replay_start_size,
                 target_update_method=args.target_update_method,
                 target_update_interval=args.target_update_interval,
                 update_interval=args.update_interval,
                 soft_update_tau=args.soft_update_tau,
                 n_times_update=args.number_of_update_times,
                 phi=phi,minibatch_size=args.minibatch_size
            )

G=[]
G_mean=[]
best_reward=-1000
for ep in range(1, args.num_episodes+ 1):
    
    obs = env.reset()
    reward = 0
    done = False
    R = 0  # return (sum of rewards)
    t = 0  # time step
    episode_rewards=[]
    while not done and t < args.max_episode_length:
        env.render()
        action = agent.act_and_train(obs, reward)
        penalty = penalties(env) #Penalties calc
        reward -= penalty
        obs, reward, done, _ = env.step(action)
        R += reward
        episode_rewards.append(reward)
        t += 1
        
    if done or t > args.max_episode_length:
            
        # Calculate sum of the rewards
        episode_rewards_sum = sum(episode_rewards)     
        G.append(episode_rewards_sum)
        total_G = np.sum(G)
        maximumReturn = np.amax(G)
        if ep % 1 == 0:    

            print("==========================================")
            print("Episode: ", ep)
            print("Rewards: ", episode_rewards_sum)
            print("Max reward so far: ", maximumReturn)
            # Mean reward
            total_reward_mean = np.divide(total_G, ep+1)
            G_mean.append(total_reward_mean)
            print("Mean Reward", total_reward_mean)
            
    # Save the model every 100 episode.       
    if episode_rewards_sum>best_reward:
        best_reward=episode_rewards_sum
        agent.save("DDPG_best_model")
        print('new best', ep)

    #generate graph of rewards vs episodes
    if ep%50==0:
        graph_reward(G, ep, 'DDPGargs')    
    agent.stop_episode_and_train(obs, reward, done)
    
    
print('Good job Alan')

plt.plot(G, color='cadetblue')
plt.ylabel('Returns')
plt.xlabel('Number of episodes')
plt.savefig("DDPG_Prosthetic_Episodes_VS_Returns.png")

plt.plot(G_mean, color='darkgoldenrod')
plt.ylabel('Average of Returns ')
plt.xlabel('Number of episodes')
plt.savefig("DDPG_Prosthetic_AverageEpisodes_VS_Returns.png")