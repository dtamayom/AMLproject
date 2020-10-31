import chainer #Librería Chainer para el DQN Agent
import chainer.functions as F
import chainerrl
from chainer import optimizers               
from chainerrl.agents.ddpg import DDPG      
from chainerrl.agents.ddpg import DDPGModel 
from chainerrl import explorers              
from chainerrl import misc                   
from chainerrl import policy                
from chainerrl import q_functions            
from chainerrl import replay_buffer 
import numpy as np
from osim.env import ProstheticsEnv


env = ProstheticsEnv(visualize=True)
env.change_model(model='3D', prosthetic=True, difficulty=0, seed=None) #Se empieza con dificultad 0, prótesis y 3D
#El action space para el modelo escogido es vector de lenght 19
observation = env.reset(project = True)

#Argumentos
num_ep = 200
obs_size = 160#np.asarray(env.observation_space.shape).prod() #158
print(obs_size)
action_size = np.asarray(env.action_space.shape).prod() #19
hidd_lay = 2
c_hidd_lay = 30
actor_lr = 0.0001
critic_lr = 0.0001
gamma=0.995 

# Función Q
q_func = q_functions.FCSAQFunction(obs_size, action_size, n_hidden_channels=hidd_lay, n_hidden_layers=c_hidd_lay)

# Policy 
pi = policy.FCDeterministicPolicy(obs_size, action_size=action_size, n_hidden_channels=hidd_lay, n_hidden_layers=c_hidd_lay, min_action=env.action_space.low, max_action=env.action_space.high, bound_action=True)


# El Modelo

model = DDPGModel(q_func=q_func, policy=pi)
opt_actor = optimizers.Adam(alpha=actor_lr)
opt_critic = optimizers.Adam(alpha=critic_lr)
opt_actor.setup(model['policy'])
opt_critic.setup(model['q_function'])
opt_actor.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_a')
opt_critic.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_c')
rbuf = replay_buffer.ReplayBuffer(5 * 10 ** 5 )
ou_sigma = (env.action_space.high - env.action_space.low) * 0.2
explorer = explorers.AdditiveOU(sigma=ou_sigma)

def phi(obs):
    """ Convert the data type of the observation to float-32
    Input: observation (obs)
    Output:  the processed observation 
    """ 
    obs=np.array(obs)
    return obs.astype(np.float32)


# Agent
agent = DDPG(model, opt_actor, opt_critic, rbuf, gamma=gamma,
                 explorer=explorer, replay_start_size=5000,
                 target_update_method='soft',
                 target_update_interval=1,
                 update_interval=4,
                 soft_update_tau=1e-2,
                 n_times_update=1,
                 phi=phi,minibatch_size=128
            )


# construct a controller, i.e. a function from the state space (current positions, velocities and accelerations of joints) to action space (muscle excitations), that will enable to model to travel as far as possible in a fixed amount of time.
total_reward = 0.0
for ep in range(1, num_ep+ 1):
    # make a step given by the controller and record the state and the reward
    obs = env.reset()
    reward = 0
    done = False
    while not done:
        env.render()
        action = agent.act_and_train(obs, reward)
        observation, reward, done, info = env.step(action, project = True)
        total_reward += reward
        print('Episodio: ', ep)
        print('Reward', reward)

# Your reward is
print("Total reward %f" % total_reward)