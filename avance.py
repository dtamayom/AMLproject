from osim.env import L2M2019Env #Ambiente
import chainer #Librería Chainer para el DQN Agent
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
from osim.env import ProstheticsEnv

env = ProstheticsEnv(visualize=True)

env = L2M2019Env(visualize=True)

def phi(obs):
    """ Convert the data type of the observation to float-32
    Input: observation (obs)
    Output:  the processed observation 
    """ 
    obs=np.array(obs)
    return obs.astype(np.float32)

#DQN Agent
q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
    18, 3,
    n_hidden_layers=2, n_hidden_channels=50)

optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)

# Set the discount factor that discounts future rewards.
gamma = 0.95

# Use epsilon-greedy for exploration
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

# Since observations from CartPole-v0 is numpy.float64 while
# Chainer only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
#phi = lambda x: x.astype(np.float32, copy=False)

# Now create an agent that will interact with the environment.
agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, update_interval=1,
    target_update_interval=100, phi=phi)

n_episodes = 200
max_episode_len = 200
for i in range(1, n_episodes + 1):
    obs = env.reset()
    reward = 0
    done = False
    R = 0  # return (sum of rewards)
    t = 0  # time step
    while not done and t < max_episode_len:
        # Uncomment to watch the behaviour
        env.render()
        action = agent.act_and_train(obs, reward)
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
    if i % 10 == 0:
        print('episode:', i,
              'R:', R,
              'statistics:', agent.get_statistics())
    agent.stop_episode_and_train(obs, reward, done)
print('Finished.')


# observation = env.reset()
# for i in range(200):
#     observation, reward, done, info = env.step(env.action_space.sample())  #env.action_space.sample() returns a random vector for muscle activations (red active)

# #construct a controller, i.e. a function from the state space (current positions, velocities and accelerations of joints) to action space (muscle excitations), that will enable to model to travel as far as possible in a fixed amount of time.
# total_reward = 0.0
# for i in range(200):
#     # make a step given by the controller and record the state and the reward
#     observation, reward, done, info = env.step(my_controller(observation))
#     total_reward += reward
#     if done:
#         break

# # Your reward is
# print("Total reward %f" % total_reward)
