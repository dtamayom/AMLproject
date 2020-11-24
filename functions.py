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
import time

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
    
    def update(self, batch_size, env):
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
        critic_loss = self.critic_criterion(Qvals, Qprime) #MAximized MSE between original and updated Q values)

        # Actor loss: mean of the sum of gradients calculated from mini-batch
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        # Velocity loss
        #velocity_loss = self.critic_criterion(velocity(env), v_obj)
        #pdb.set_trace()
        if args.use_v_obj:
            loss = nn.MSELoss()
            velocity_loss = loss(torch.as_tensor(velocity(env)), torch.as_tensor(args.v_obj))
            velocity_loss.requires_grad = True
            
        # update networks with ADAM optimizer
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()  
        self.critic_optimizer.step()

        if args.use_v_obj:
            self.critic_optimizer.zero_grad()
            velocity_loss.backward() 
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
    
    def load_checkpoint(self, checkpoint_path):
        """
        Carga un modelo guardado en archivo .pth.tar
        Arguments:
            checkpoint_path:    File to load the model from
        """
        checkpoint_path  =os.path.join(args.checkpoint_dir, checkpoint_path)
        if os.path.isfile(checkpoint_path):
            #logger.info("Loading checkpoint...({})".format(checkpoint_path))
            print('Loading checkpoint...')
            key = 'cuda' if torch.cuda.is_available() else 'cpu'
            checkpoint = torch.load(checkpoint_path, map_location=key)
            start_timestep = checkpoint['last_timestep'] #+ 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            replay_buffer = checkpoint['replay_buffer']

            gc.collect()
            #logger.info('Loaded model at timestep {} from {}'.format(start_timestep, checkpoint_path))
            print('Model',checkpoint_path,'loaded')
            return start_timestep, replay_buffer
        else:
            raise OSError('Checkpoint not found')



def make_env(test,render):
    seed=np.random.seed()
    env = ProstheticsEnv(visualize=render)
    env.change_model(model='3D', prosthetic=True, difficulty=0, seed=None)
    # Use different random seeds for train and test envs
    env_seed = 2 ** 32 - 1 - seed if test else seed
    env.seed(env_seed)
    return env


def graph_reward(reward, eps, avg_reward, saveas):
    name = saveas + str(eps) + '.png'
    episodes = np.linspace(0,eps,eps)
    plt.figure()
    plt.plot(episodes, reward,'cadetblue',label='Standard')
    plt.xlabel("Number of Episodes")
    plt.ylabel("Reward")
    plt.plot(avg_reward, color='darkgoldenrod', label='Average')
    plt.legend()
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

# def shape_rew(env):
#         state_desc = env.get_state_desc()
#         prev_state_desc = env.get_prev_state_desc()
#         if not prev_state_desc:
#             return 0
#         penalty = 0.
#         penalty += (state_desc["body_vel"]["pelvis"][0] - 3.0) ** 2
#         penalty += (state_desc["body_vel"]["pelvis"][2]) ** 2
#         penalty += np.sum(np.array(env.osim_model.get_activations()) ** 2) * 0.001
#         if state_desc["body_pos"]["pelvis"][1] < 0.70:
#             penalty += 10  # penalize falling more

#         # Reward for not falling
#         reward = 10.0

#         return reward - penalty


def step_jor(env, action):
        reward = 0.
        for _ in range(500):
            env.prev_state_desc = env.get_state_desc()
            env.osim_model.actuate(action)
            env.osim_model.integrate()
            done = env.is_done()
            rewards = shape_rew(env)
            if done:
                break

        obs = env.get_state_desc()

        return obs, rewards, done, {'r': reward}

def shape_rew(env):
        state_desc = env.get_state_desc()
        prev_state_desc = env.get_prev_state_desc()
        if not prev_state_desc:
            return 0
        penalty = 0.
        penalty += (state_desc["body_vel"]["pelvis"][0] - 3.0) ** 2
        penalty += (state_desc["body_vel"]["pelvis"][2]) ** 2
        penalty += np.sum(np.array(env.osim_model.get_activations()) ** 2) * 0.001
        if state_desc["body_pos"]["pelvis"][1] < 0.70:
            penalty += 10  # penalize falling more

        # Reward for not falling
        reward = 10.0

        return reward - penalty

def reward_first(env):
        state_desc = env.get_state_desc()
        prev_state_desc = env.get_prev_state_desc()
        if not prev_state_desc:
            return 0
        return 9.0 - (state_desc["body_vel"]["pelvis"][0] - 3.0)**2

def get_observation(env):
        state_desc = env.get_state_desc()
        return project_obs(state_desc, proj=self.project_mode, prosthetic=self.prosthetic)

# def step_jor(env, action):
#         reward = 0.
#         rewardb = 0
#         for _ in range(500):
#             env.prev_state_desc = env.get_state_desc()
#             start_time = time.perf_counter()
#             env.osim_model.actuate(action)
#             env.osim_model.integrate()
#             step_time = time.perf_counter() - start_time
#             done = env.is_done()

#             if step_time > 15.:
#                 reward += -10
#                 done = True
#             else:
#                 reward += shape_rew(env)
#             rewardb += reward_first(env)
#             #rewards = shape_rew(env)
#             if done:
#                 break
#         print(reward)
#         print(rewardb)

#         #obs = env.get_state_desc()
#         obs = env.get_observation()

#         #print(obs)

#         return obs, reward, done, {'r': reward}
    

# def reset(self):
# self.time_step = 0

# if self.episodes % self.ep2reload == 0:
#     self.env = ProstheticsEnv(
#         visualize=self.visualize, integrator_accuracy=1e-3)
#     self.env.change_model(
#         model=self.model, prosthetic=True, difficulty=0,
#         seed=np.random.randint(200))

# state_desc = self.env.reset(project=False)
# if self.randomized_start:
#     state = get_simbody_state(state_desc)
#     noise = np.random.normal(scale=0.1, size=72)
#     noise[3:6] = 0
#     state = (np.array(state) + noise).tolist()
#     simbody_state = self.env.osim_model.get_state()
#     obj = simbody_state.getY()
#     for i in range(72):
#         obj[i] = state[i]
#     self.env.osim_model.set_state(simbody_state)

# observation = preprocess_obs(state_desc)
# if self.observe_time:
#     observation.append(-1.0)

# return observation

def is_done(env, observation, time_step, frame_skip):
    max_episode_length=args.num_steps
    pelvis_y = observation["body_pos"]["pelvis"][1]
    if time_step * frame_skip > max_ep_length:
        return True
    elif pelvis_y < 0.6:
        return True
    return False

def shape_reward_s(env, reward, time_step, frame_skip):
    state_desc = env.get_state_desc()
    max_ep_length=args.num_steps
    death_penalty=0.0
    living_bonus=0.1
    side_dev_coef=0.1
    cross_legs_coef=0.1
    bending_knees_coef = 0.1
    side_step_penalty = False
    # death penalty
    if time_step * frame_skip < max_ep_length:
        reward -= death_penalty
    else:
        reward += living_bonus

    # deviation from forward direction penalty
    vy, vz = state_desc['body_vel']['pelvis'][1:]
    side_dev_penalty = (vy ** 2 + vz ** 2)
    reward -= side_dev_coef * side_dev_penalty

    # crossing legs penalty
    pelvis_xy = np.array(state_desc['body_pos']['pelvis'])
    left = np.array(state_desc['body_pos']['toes_l']) - pelvis_xy
    right = np.array(state_desc['body_pos']['pros_foot_r']) - pelvis_xy
    axis = np.array(state_desc['body_pos']['head']) - pelvis_xy
    cross_legs_penalty = np.cross(left, right).dot(axis)
    if cross_legs_penalty > 0:
        cross_legs_penalty = 0.0
    reward += cross_legs_coef * cross_legs_penalty

    # bending knees bonus
    r_knee_flexion = np.minimum(state_desc['joint_pos']['knee_r'][0], 0.)
    l_knee_flexion = np.minimum(state_desc['joint_pos']['knee_l'][0], 0.)
    bend_knees_bonus = np.abs(r_knee_flexion + l_knee_flexion)
    reward += bending_knees_coef * bend_knees_bonus

    # side step penalty
    #if side_step_penalty:
        #rx, ry, rz = state_desc['body_pos_rot']['pelvis']
        #R = euler_angles_to_rotation_matrix([rx, ry, rz])
        #reward *= (1.0 - math.fabs(R[2, 0]))

    return reward

def step_s(env, action):
    reward = 0
    reward_origin = 0
    reward_scale=0.1
    frame_skip=1
    time_step = 0

    #action = lambda x: x
    action = np.clip(action, 0.0, 1.0)

    for i in range(frame_skip):
        observation, r, _, info = env.step(action)
        reward_origin += r
        done = env.is_done()
        #reward += shape_reward_s(env, r, time_step, frame_skip)
        reward += shape_rew(env)
        if done:
            break

    observation = env.get_observation()
    reward *= reward_scale
    info["reward_origin"] = reward_origin
    time_step += 1

    return observation, reward, done, info

# def is_done(env, observation):
#     pelvis_y = observation["body_pos"]["pelvis"][1]
#     if time_step * frame_skip > max_ep_length:
#         return True
#     elif pelvis_y < 0.6:
#         return True
#     return False

PROJ_FULL = 0
PROJ_NORMAL = 1
PROJ_SIMPLE = 2

def project_obs(state_desc, proj=PROJ_FULL, prosthetic=True):
    res = []

    if proj == PROJ_SIMPLE:
        pelvis = state_desc["body_pos"]["pelvis"][0:3]
        # pelvis_vel = state_desc["body_vel"]["pelvis"][0:3]
        # pelvis_acc = state_desc["body_acc"]["pelvis"][0:3]
        res += pelvis[1:2]  # + pelvis_vel[:] + pelvis_acc[:]
        for bp in ["talus_l", "pros_foot_r"]:
            bp_pos = state_desc["body_pos"][bp].copy()
            bp_pos[0] = bp_pos[0] - pelvis[0]
            bp_pos[2] = bp_pos[2] - pelvis[2]
            res += bp_pos
    else:
        pelvis = None
        for body_part in ["pelvis", "head", "torso", "toes_l", "toes_r", "talus_l", "talus_r"]:
            if prosthetic and body_part in ["toes_r", "talus_r"]:
                if proj == PROJ_FULL:
                    res += [0] * 12
                continue
            cur = []
            cur += state_desc["body_pos"][body_part][0:3]
            cur += state_desc["body_vel"][body_part][0:3]
            cur += state_desc["body_acc"][body_part][0:3]
            cur += state_desc["body_pos_rot"][body_part][2:]
            cur += state_desc["body_vel_rot"][body_part][2:]
            cur += state_desc["body_acc_rot"][body_part][2:]
            if body_part == "pelvis":
                pelvis = cur.copy()
                res += pelvis[1:2] + pelvis[3:]
            else:
                cur_upd = cur.copy()
                cur_upd[:3] = [cur[i] - pelvis[i] for i in range(3)]
                cur_upd[9:10] = [cur[i] - pelvis[i] for i in range(9, 10)]
                res += cur_upd

    for joint in ["ankle_l", "ankle_r", "back", "hip_l", "hip_r", "knee_l", "knee_r"]:
        res += state_desc["joint_pos"][joint]
        res += state_desc["joint_vel"][joint]
        res += state_desc["joint_acc"][joint]

    for muscle in sorted(state_desc["muscles"].keys()):
        res += [state_desc["muscles"][muscle]["activation"]]
        res += [state_desc["muscles"][muscle]["fiber_length"]]
        res += [state_desc["muscles"][muscle]["fiber_velocity"]]

    cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(3)]
    cm_vel = state_desc["misc"]["mass_center_vel"]
    cm_acc = state_desc["misc"]["mass_center_acc"]
    res = res + cm_pos + cm_vel + cm_acc

    return np.array(res)


class MyProstheticsEnv(ProstheticsEnv):

    def __init__(self, visualize=False, integrator_accuracy=1e-4, difficulty=0, seed=0, frame_skip=0):
        self.project_mode = PROJ_FULL
        super(MyProstheticsEnv, self).__init__(
            visualize=visualize,
            integrator_accuracy=integrator_accuracy,
            difficulty=difficulty,
            seed=seed)
        if difficulty == 0:
            self.time_limit = 600  # longer time limit to reduce likelihood of diving strategy
        self.spec.timestep_limit = self.time_limit
        np.random.seed(seed)
        self.frame_times = deque(maxlen=100)
        self.frame_count = 0
        self.frame_skip = frame_skip
        self.debug = False

    def get_observation(self):
        state_desc = self.get_state_desc()
        return project_obs(state_desc, proj=self.project_mode, prosthetic=self.prosthetic)

    def get_observation_space_size(self):
        if self.prosthetic:
            if self.project_mode == PROJ_SIMPLE:
                return 106
            elif self.project_mode == PROJ_FULL:
                return 181
            else:
                return 157
        return 167

    def is_done(self):
        state_desc = self.get_state_desc()
        return state_desc["body_pos"]["pelvis"][1] < 0.65

    def my_reward_round1(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0

        penalty = 0.
        penalty += (state_desc["body_vel"]["pelvis"][0] - 3.0) ** 2
        penalty += (state_desc["body_vel"]["pelvis"][2]) ** 2
        penalty += np.sum(np.array(self.osim_model.get_activations()) ** 2) * 0.001
        if state_desc["body_pos"]["pelvis"][1] < 0.70:
            penalty += 10  # penalize falling more

        # Reward for not falling
        reward = 10.0

        return reward - penalty

    def reward_round1(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0
        return 9.0 - (state_desc["body_vel"]["pelvis"][0] - 3.0)**2


    def step(self, action, project=True):
        reward = 0.
        rewardb = 0.
        done = False

        if self.frame_skip:
            num_steps = self.frame_skip
        else:
            num_steps = 1

        for _ in range(num_steps):
            self.prev_state_desc = self.get_state_desc()

            start_time = time.perf_counter()
            self.osim_model.actuate(action)
            self.osim_model.integrate()
            step_time = time.perf_counter() - start_time

            # track some step stats across resets
            self.frame_times.append(step_time)
            self.frame_count += 1

            if self.debug and self.frame_count % 1000 == 0:
                frame_mean = np.mean(self.frame_times)
                frame_min = np.min(self.frame_times)
                frame_max = np.max(self.frame_times)
                print('Steps {}, duration mean, min, max: {:.3f}, {:.3f}, {:.3f}'.format(
                    self.frame_count, frame_mean, frame_min, frame_max))

            done = self.is_done() or self.osim_model.istep >= self.spec.timestep_limit
            if step_time > 15.:
                reward += -10
                done = True
            else:
                reward += self.my_reward_round1()
            rewardb += self.reward_round1()

            if done:
                break

        if project:
            obs = self.get_observation()
        else:
            obs = self.get_state_desc()

        return obs, reward, done, {'rb': rewardb}

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
