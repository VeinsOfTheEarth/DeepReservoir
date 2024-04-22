# -*- coding: utf-8 -*-
"""
Code to simulate Navajo reservoir control using PPO algorithm.

Dependencies:
- time
- numpy
- pandas
- gymnasium
- stable-baselines3
- tensorflow
- pytorch

Note: Please install the required libraries before executing the code.

"""

# Importing required libraries
import os
import io
import time
import random
import numpy as np
import pandas as pd
import torch as th
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

# Loading the navajo reservoir dataset 
df_train=pd.read_csv('C:\\Users\\368649\\Desktop\\NAVAJORESERVOIR.csv')
df_train=df_train.iloc[:,0:6]
data_train=df_train.values
data_train=data_train[0:22500,0:6]

class reservoir(Env):
    """
    Observation Space:
    - Continuous Box space with shape (1, 5) and dtype float64. Represents the states of the reservoir.   
    
    Action Space:
    - Box space with continuous actions in the range [0,1]. Represents the percentage of valve opening to control outflow.

    Methods:
    - __init__(): Initializes the environment states.
    - reset(): Resets the enviornment to the initial states and returns the corresponding state values.
    - step(): Takes an action in the environment and returns the next state, reward, done flag, truncated flag, auxiliary info.
    - render(): Used to visualize the agnet-environment interaction.
    - seed(): Sets the seed value.
    - close(): Closes the simulation window.

    Attributes:
    - time: Number of seconds lapsed in between data sample collection.
    - area: Area of the reservoir (acre).
    - data: Training data (sampling frequency -daily).
    - storage: Storage volume of the reservoir (acre-feet).
    - inflow: Volume rate of inflow into the reservoir (cubic feet per second).
    - outflow: Volume rate of flow out of the reservoir (cubic feet per second).
    - evaporation: Volume of water evaporated from the reservoir (acre-feet).
    - height: Height of water in the reservoir (feet).
    -maximum_height: Maximum allowable height of water in the reservoir (feet).

    """
    def __init__(self,data_train)->None:
        lm=10000000.0
        self.time=86400.0
        self.area=15610.0
        self.data=data_train
        self.random=random
        self.data_idx=random.randint(0, len(self.data)-1)
        self.observation_space=Box(low=np.array([-lm,-lm,-lm,-lm,-lm]),high=np.array([lm,lm,lm,lm,lm]),dtype=np.float64)
        self.action_space=Box(low=0.5,high=1.5,dtype=np.float64)
        self.storage=self.data[self.data_idx,1]
        self.inflow=self.data[self.data_idx,3]
        self.outflow=self.data[self.data_idx,5]
        self.evaporation=self.data[self.data_idx,2]
        self.height=(self.storage+(self.inflow-self.outflow)*self.time-self.evaporation)/self.area
        self.state =np.array([self.storage,self.inflow,self.outflow,self.evaporation,self.height])
        self.episode_length=100
        self.maximum_height=20.5
        
    def reset(self, seed=None)->float:
        """
        Resets the enviornment to the initial states and returns the corresponding state values.

        Returns:
        - state (numpy array of floating point numbers): The initial state values.
        """
        self.data_idx=random.randint(0, len(self.data)-1)
        self.storage=self.data[self.data_idx,1]
        self.inflow=self.data[self.data_idx,3]
        self.outflow=self.data[self.data_idx,5]
        self.evaporation=self.data[self.data_idx,2]
        self.height=(self.storage+(self.inflow-self.outflow)*self.time-self.evaporation)/self.area
        self.state =np.array([self.storage,self.inflow,self.outflow,self.evaporation,self.height])
        self.episode_length=100
        info={}
        return self.state,info
    
    def step(self,action):
        """
        Takes an action in the environment and returns the next state, reward, done flag, truncated flag, and auxiliary info.

        Parameters:
        - action (float): The action taken by the agent.

        Returns:
        - state (numpy array of floating point numbers): The next state of the environment.
        - reward (int): The reward returned by the enviornment.
        - done (bool): Flag indicating whether the episode is completed.
        - truncated (bool): Flag indicating whether the episode was truncated due to a
          reason not defined as the part of the MDP.
        - info (dict): Auxiliary information.
        """
        self.state[2]=self.state[2]*action
        self.state[4]=(self.state[0]+(self.state[1]-self.state[2])*self.time-self.state[3])/self.area
        self.episode_length-=1
        self.data_idx+=1
        if self.state[4]<=self.maximum_height:
            reward=1
        elif self.state[4]>self.maximum_height:
            reward=-2
        if self.episode_length<=0:
            done=True
        else:
            done=False
        truncated=False
        info={}
        return self.state,reward,done,truncated,info
    
    def render(self)->None:
        pass
    
    def seed(self)->None:
        pass
    
    def close(self)->None:
        pass
    

#%%
env=reservoir(data_train)
check_env(env)

#%%
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
log_path=os.path.join('runs_navajo')
# env=Monitor(env, log_path)
agent=PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)
agent.learn(total_timesteps=20000)

#%%
agent_path=os.path.join('runs_navajo', 'PPO')
agent.save(agent_path)
del agent
agent=PPO.load(agent_path,env=env)
evaluate_policy(agent,env,n_eval_episodes=10,render=False)

#%%
# plot_results([log_path], 1e5, results_plotter.X_TIMESTEPS, "reservoir")  