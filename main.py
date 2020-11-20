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

cuda = True if torch.cuda.is_available() else False

args = parser()
print_args(args)

# Ornstein-Ulhenbeck Process explorer to add noise to the action output
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space):
        self.mu           = args.mu
        self.theta        = args.theta
        self.sigma        = args.max_sigma
        self.max_sigma    = args.max_sigma
        self.min_sigma    = args.min_sigma
        self.decay_period = args.decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)
        

#Experience replay buffer class
class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

#Standard Actor-Critic Architecture
class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        if input_size==177:
            input_size=179
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4):
        super(Actor, self).__init__()
        if input_size==158: 
            input_size=160
        self.linear1 = nn.Linear(input_size, hidden_size) #input_size
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x


class DDPGagent:
    def __init__(self, env):
        # Params
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = args.gamma

        #Params from args______________
        self.hidden_size = args.hidden_size
        self.actor_learning_rate = args.actor_lr
        self.critic_learning_rate = args.critic_lr
        self.tau = args.tau
        #________________________________

        # Network and target network initialization
        self.actor = Actor(self.num_states, self.hidden_size, self.num_actions)
        self.actor_target = Actor(self.num_states, self.hidden_size, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, self.hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, self.hidden_size, self.num_actions)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.memory = Memory(args.max_memory)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        if cuda:
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic.cuda()
            self.critic_target.cuda()
            self.critic_criterion.cuda()
    
    def get_action(self, state):
        state = Variable(torch.from_numpy(np.array(state)).float().unsqueeze(0))
        if cuda:
            state= state.cuda()
        action = self.actor.forward(state)
        action = action.detach().cpu().numpy()[0,0]
        return action
    
    def update(self, batch_size):
        #pdb.set_trace()
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        if cuda:
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            next_states = next_states.cuda()
    
        # Critic loss        
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime) #MAximized MSE between original and updated Q values

        # Actor loss: mean of the sum of gradients calculated from mini-batch
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        
        # update networks with ADAM optimizer
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks via soft updates (copy targetand make it track learned network)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        
     #Adapted from https://github.com/MoritzTaylor/ddpg-pytorch/blob/master/ddpg.py
    def save_checkpoint(self, last_timestep):
        """
        Saving the networks and all parameters to a file in 'checkpoint_dir'
        Arguments:
            last_timestep:  Last timestep in training before saving
            replay_buffer:  Current replay buffer
        """
        checkpoint_name = args.checkpoint_dir + '/ep_{}.pth.tar'.format(last_timestep)
        #logger.info('Saving checkpoint...')
        checkpoint = {
            'last_timestep': last_timestep,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'replay_buffer': self.memory,
        }
        #logger.info('Saving model at timestep {}...'.format(last_timestep))
        torch.save(checkpoint, checkpoint_name)
        gc.collect()
        #logger.info('Saved model at timestep {} to {}'.format(last_timestep, self.checkpoint_dir))



def make_env(test,render):
    seed=0
    env = ProstheticsEnv(visualize=render)
    env.change_model(model='3D', prosthetic=True, difficulty=0, seed=None)
    # Use different random seeds for train and test envs
    env_seed = 2 ** 32 - 1 - seed if test else seed
    env.seed(env_seed)
    return env

def graph_reward(reward, eps, saveas):
    name = saveas + str(eps) + '.png'
    episodes = np.linspace(0,eps,eps)
    plt.figure()
    plt.plot(episodes, reward,'cadetblue',label='DDPG')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(args.graphs_folder,name))
    plt.close()

def penalties(env, tra_curr):
    if tra_curr == 1:
        target_v = 4
    if tra_curr == 2:
        target_v = 2
    state_desc = env.get_state_desc()
    penalty = 0
    penalty += np.sum(np.array(env.osim_model.get_activations()) ** 2) * 0.001
    vel_penalty = ((state_desc["body_vel"]["pelvis"][0] - target_v)**2
                       + (state_desc["body_vel"]["pelvis"][2] - 0)**2)
    if state_desc["body_pos"]["pelvis"][1] < 0.70: #falling
        penalty += 10  
    if velocity(env) < tra_curr:
        penalty += 10
    return penalty

def velocity(env):
    state_desc = env.get_state_desc()
    cur_vel_x = state_desc['body_vel']['pelvis'][0]
    cur_vel_z = state_desc['body_vel']['pelvis'][2]
    cur_vel = (cur_vel_x**2 + cur_vel_z**2)**0.5
    return cur_vel
    

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
        new_state, reward, done, _ = env.step(action) 
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
    
    if episode%2==0:
        print('Velocity: ', velocity(env))

    if episode_reward>best_reward:
        best_reward=episode_reward
        #agent.save("DDPG_best_model2")
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
plt.savefig("DDPG_Prosthetic_Episodes_VS_Returns.png")

plt.plot(avg_rewards, color='darkgoldenrod')
plt.ylabel('Average of Returns ')
plt.xlabel('Number of episodes')
plt.savefig("DDPG_Prosthetic_AverageEpisodes_VS_Returns.png")
