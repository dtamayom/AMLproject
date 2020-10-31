import chainer #Librería Chainer para el DQN Agent
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
from osim.env import ProstheticsEnv


env = ProstheticsEnv(visualize=True)
observation = env.reset(project = True)
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample(), project=True)  #env.action_space.sample() returns a random vector for muscle activations (red active)

'''construct a controller, i.e. a function from the state space (current positions, velocities and accelerations of joints) to action space (muscle excitations), that will enable to model to travel as far as possible in a fixed amount of time.
total_reward = 0.0
for i in range(200):
    # make a step given by the controller and record the state and the reward
    observation, reward, done, info = env.step(my_controller(observation))
    total_reward += reward
    if done:
        break

# Your reward is
print("Total reward %f" % total_reward)