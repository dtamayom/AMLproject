from osim.env import ProstheticsEnv
from chainerrl.agents.ddpg import DDPG      
from chainerrl.agents.ddpg import DDPGModel  
from arguments import parser, print_args
from chainerrl import explorers              
from chainerrl import misc                  
from chainerrl import policy                 
from chainerrl import q_functions           
from chainerrl import replay_buffer 
import chainer                               
from chainer import optimizers 
import numpy as np
args = parser()

env = ProstheticsEnv(visualize=True)
env.change_model(model='3D', prosthetic=True, difficulty=0, seed=2 ** 32 - 1)
phi = lambda x: np.array(x).astype(np.float32, copy=False)
obs_size = np.asarray(env.observation_space.shape).prod()
action_space = env.action_space
action_size = np.asarray(action_space.shape).prod()

#Critic Network
q_func = q_functions.FCSAQFunction(
            160, 
            action_size,
            n_hidden_channels=args.hidden_size,
            n_hidden_layers=args.hidden_size)

#Policy Network
pi = policy.FCDeterministicPolicy(
            160, 
            action_size=action_size,
            n_hidden_channels=args.hidden_size,
            n_hidden_layers=args.hidden_size,
            min_action=action_space.low, 
            max_action=action_space.high,
            bound_action=True)

#Model
model = DDPGModel(q_func=q_func, policy=pi)
opt_actor = optimizers.Adam(alpha=args.actor_lr)
opt_critic = optimizers.Adam(alpha=args.critic_lr)
opt_actor.setup(model['policy'])
opt_critic.setup(model['q_function'])
opt_actor.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_a')
opt_critic.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_c')
rbuf = replay_buffer.ReplayBuffer(5 * 10 ** 5)
ou_sigma = (action_space.high - action_space.low) * 0.2

#Explorer
explorer = explorers.AdditiveOU(sigma=ou_sigma)

#Agent
agent = DDPG(model, opt_actor, opt_critic, rbuf, gamma=args.gamma,
                 explorer=explorer, replay_start_size=5000,
                 target_update_method='soft',
                 target_update_interval=1,
                 update_interval=4,
                 soft_update_tau=1e-2,
                 n_times_update=1,
                 phi=phi,minibatch_size=128 #args.minibatch_size
            )

# Cargar el modelo guardado
agent.load("DDPG_last_model_rew8k")

for i in range(args.test_episodes):
    obs = env.reset()
    done = False
    reward_Alan = 0
    t = 0
    while not done and t < 200:
        #env.render()
        #agent.load("DDPG_best_model")
        action = agent.act(obs)
        obs, r, done, x = env.step(action)
        reward_Alan += r
        t += 1
    print('Test episode:', i, 'Reward obtained: ', reward_Alan)
    #agent.stop_episode()
