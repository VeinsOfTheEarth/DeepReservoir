# -*- coding: utf-8 -*-
"""
Code to simulate reservoir control using PPO algorithm: Toy problem. 
Under continuous upgradation.
"""

import os
import io
import time
import numpy as np
import gymnasium
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, plot_results,ts2xy

class reservoir(Env):
  def __init__(self):
    self.observation_space=Box(low=np.array([0.0,0.0]),high=np.array([4.5,1.0]),dtype=np.float64)
    self.action_space=Box(low=0.0,high=1.0,dtype=np.float64)
    self.height =np.random.uniform(low=0.0,high=4.5)
    self.outflow =np.random.uniform(low=0.08,high=0.35)
    self.state=np.array([self.height,self.outflow])
    self.episode_length=30
    self.maximum_height=3.5
    self.maximum_outflow=0.1
    
  def reset(self,seed=None):
    self.height =np.random.uniform(low=0.0,high=4.5)
    self.outflow =np.random.uniform(low=0.08,high=0.35)
    self.state=np.array([self.height,self.outflow])
    self.episode_length=30
    info={}
    return self.state,info

  def step(self,action):
    self.state[0]=self.state[0]*action
    self.state[1]=self.state[1]*action
    self.episode_length-=1
    if self.state[1]<=self.maximum_outflow:
      reward =1
    elif self.state[1]>self.maximum_outflow:
      reward=-1
    elif self.state[0]>self.maximum_height:
      reward=-10
    if self.episode_length<=0:
      done = True
    else:
      done=False
    truncated=False
    info={}
    return self.state,reward,done,truncated,info

  def render(self):
      pass

  def close(self):
    pass

  def seed(self):
    pass

#%%
env=reservoir()
check_env(env)
episodes =10
for episode in range(1,episodes+1):
  state, info=env.reset()
  print(state)
  done=False
  tot_reward=0
  while done==False:
    action=env.action_space.sample()
    state, reward, done, truncated, info=env.step(action)
    print(reward)
    tot_reward+=reward
  print('Episode: {}, Total reward: {}'.format(episode,tot_reward))
  
#%%
env=reservoir()
log_path=os.path.join('runs')
# env=Monitor(env, log_path)
agent=PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)
agent.learn(total_timesteps=20000)

#%%
agent_path=os.path.join('runs', 'PPO')
agent.save(agent_path)
del agent
agent=PPO.load(agent_path,env=env)
evaluate_policy(agent,env,n_eval_episodes=10,render=False)

#%%
# plot_results([log_path], 1e5, results_plotter.X_TIMESTEPS, "reservoir")