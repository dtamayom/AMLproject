from osim.env import ProstheticsEnv 
from arguments import parser, print_args
import numpy as np
from functions import OUNoise, NormalizedEnv, Memory, Critic, Actor, DDPGagent, make_env, graph_reward, penalties, velocity, shape_rew, step_jor
from functions import MyProstheticsEnv
import torch

args = parser()

#Check mode_test and render_environment arguments!!!!
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#env = make_env(args.mode_test, args.render_environment)
env = MyProstheticsEnv(integrator_accuracy=1e-4)
nv.seed=2 ** 32 - 1

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