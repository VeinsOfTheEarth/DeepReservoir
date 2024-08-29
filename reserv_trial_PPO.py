# -*- coding: utf-8 -*-
"""
@Shubhendu

Code to simulate Navajo reservoir control using PPO algorithm. This code creates an AI agent to imitate the reservoir operator.

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
import os
import io
import time
import numpy as np
import pandas as pd
import torch as th
import gymnasium
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, plot_results,ts2xy
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Define the environment
class Reservoir(Env):
    """
    Observation Space:
    - Continuous Box space with shape (4, ) and dtype float64. Represents the states of the reservoir.   
    
    Action Space:
    - Box space with continuous actions in the range [0,1]. Represents the amount of water released.

    Methods:
    - __init__(): Initializes the environment states.
    - reset(): Resets the enviornment to the initial states and returns the corresponding state values.
    - step(): Takes an action in the environment and returns the next state, reward, done flag, truncated flag, auxiliary info.
    - render(): Used to visualize the agnet-environment interaction.
    - seed(): Sets the seed value.
    - close(): Closes the simulation window.
    """
    
    def __init__(self, data, scaler, target_scaler)->None:
        super(Reservoir, self).__init__()

        self.data = data
        self.scaler = scaler
        self.target_scaler = target_scaler
        self.current_step = 0

        # Define action and observation space
        self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Observations: Elevation, Storage, Evaporation, Inflow
        self.observation_space = Box(low=0, high=1, shape=(4,), dtype=np.float32)

    def reset(self, seed=None)->None:
        """
        Resets the enviornment to the initial states and returns the corresponding state values.

        Returns:
        - state (numpy array of floating point numbers): The initial state values.
        """
        self.current_step = 0
        info={}
        return self._get_observation(), info

    def _get_observation(self):
        obs = self.data[self.current_step, :-1]
        return obs

    def step(self, action):
        """
        Takes an action in the environment and returns the next state, reward, done flag, truncated flag, and auxiliary info.

        Parameters:
        - action (float): The action taken by the agent.

        Returns:
        - state (numpy array of floating point numbers): The next state of the environment.
        - reward (int): The reward returned by the enviornment.
        - done (bool): Flag indicating whether the episode is completed.
        - truncated (bool): Flag indicating whether the episode was truncated due to a
          reason not defined as part of the MDP.
        - info (dict): Auxiliary information.
        """
        release_scaled = action[0]
        target_release_scaled = self.data[self.current_step, -1]

        # reward = -abs(target_release_scaled - release_scaled )
        error = abs(target_release_scaled - release_scaled)
        reward = -np.square(error)

        # Update step
        self.current_step += 1

        # Check if we reached the end of the data
        done = self.current_step >= len(self.data)

        # Get next observation
        if not done:
            observation = self._get_observation()
        else:
            observation = np.zeros(4)
        truncated=False
        info={}
        return observation, reward, done, truncated, info

    def render(self, mode='human', close=False)->None:
        pass

    def seed(self)->None:
        pass
    
    def close(self)->None:
        pass

# Load the data
data = pd.read_csv('/content/NAVAJORESERVOIR08-18-2024T16.48.23.csv')

# Extract columns
elevation = data['Elevation (feet)'].values
storage = data['Storage (af)'].values
evaporation = data['Evaporation (af)'].values
inflow = data['Inflow** (cfs)'].values
release = data['Total Release (cfs)'].values

# Stack columns into a single dataset
dataset = np.column_stack((elevation, storage, evaporation, inflow, release))

# Scale the inputs and outputs between 0 and 1
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)

# Separate features and target for scaling
features = dataset[:, :-1]
target = dataset[:, -1].reshape(-1, 1)

# Scale features and target
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

target_scaler = MinMaxScaler()
target_scaled = target_scaler.fit_transform(target)

# Combine scaled features and target
scaled_data = np.column_stack((features_scaled, target_scaled))

# Create the environment
env = Reservoir(scaled_data, scaler, target_scaler)
env = DummyVecEnv([lambda: env])

# Instantiate the agent
# Custom policy network
policy_kwargs = dict(
    net_arch=[dict(pi=[64, 64, 64], vf=[64, 64, 64])]
)

# Instantiate the agent with custom policy and hyperparameters
agent = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, learning_rate=0.0001, n_steps=2048, batch_size=64, n_epochs=10)

# Train the agent
agent.learn(total_timesteps=200000)

# Save the trained model
agent.save("reserv_agent")

# Load the trained model
agent = PPO.load("reserv_agent")

# Test the trained agent
obs = env.reset()
for i in range(len(scaled_data)):
    action, _states = agent.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        break

print("Test completed.")

scaled_actual_releases = []
scaled_predicted_releases = []

# Test the trained agent and collect scaled predictions
for i in range(len(features_scaled)):
    # Predict the action (release) for the current observation
    action, _states = agent.predict(features_scaled[i])

    # The action is already scaled between 0 and 1, so we can directly use it
    release_predicted_scaled = action[0]

    # Get the actual scaled release value from the data (last element of the observation)
    actual_release_scaled = target_scaled[i]

    # Store the results
    scaled_predicted_releases.append(release_predicted_scaled)
    scaled_actual_releases.append(actual_release_scaled)

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(scaled_actual_releases, label='Actual Total Release (cfs)', color='blue')
plt.plot(scaled_predicted_releases, label='Predicted Total Release (cfs)', color='red')
plt.xlabel('Time Step')
plt.ylabel('Total Release (cfs)')
plt.title('Agent Predicted Releases vs Actual Releases')
plt.legend()
plt.show()
