# -*- coding: utf-8 -*-
"""
@author: Shubhendu
"""
import os
import numpy as np
import pandas as pd
import torch as th
import gymnasium
from gymnasium import Env
from gymnasium.spaces import Box
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Define storage and release limits
min_storage_af = 500000
max_storage_af = 1900000
initial_storage_af = 7000
navajo_min_release = 250 * 1.98211
navajo_max_release = 5000 * 1.98211

class ReservoirEnv(Env):
    def __init__(self, data, scaler):
        super(ReservoirEnv, self).__init__()
        self.data = data
        self.scaler = scaler
        self.current_step = 0
        self.storage_history = []
        self.predicted_release_history = []
        self.reward_history = []
        self.q_values = []

        # Unscaled data to track actual storage and other variables
        unscaled_data = self.scaler.inverse_transform(self.data)
        self.actual_storage_history = unscaled_data[:, 0].tolist()
        self.actual_release_history = unscaled_data[:, 3].tolist()
        self.evaporation_history = unscaled_data[:, 1].tolist()
        self.inflow_history = unscaled_data[:, 2].tolist()

        self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=0, high=1, shape=(3,), dtype=np.float32)

    def reset(self, seed=None):
        self.current_step = 0
        self.storage_history.clear()
        self.predicted_release_history.clear()
        self.reward_history.clear()
        self.q_values.clear()

        # Start storage as actual storage and log it
        self.current_storage = initial_storage_af
        self.storage_history.append(self.current_storage)
        info = {}
        return self._get_observation(), info

    def _get_observation(self):
        return self.data[self.current_step, :3]

    def step(self, action):
        release_scaled = action[0]
        release_af = release_scaled * (navajo_max_release - navajo_min_release)
        self.predicted_release_history.append(release_af)

        inflow_af = self.scaler.inverse_transform(self.data[self.current_step:self.current_step+1, :])[0, 2]
        evaporation_af = self.scaler.inverse_transform(self.data[self.current_step:self.current_step+1, :])[0, 1]

        # Update storage considering inflow, evaporation, and release
        self.current_storage = self.current_storage - release_af + inflow_af - evaporation_af
        self.storage_history.append(self.current_storage)

        # Reward for keeping storage within range
        if min_storage_af <= self.current_storage <= max_storage_af:
            storage_reward = 1
        else:
            storage_reward = -1

        reward = storage_reward
        self.reward_history.append(reward)

        # Collect Q-values for visualization
        with th.no_grad():
            obs_tensor = th.as_tensor(self._get_observation(), dtype=th.float32)
            q_values = model.critic(obs_tensor.unsqueeze(0), th.tensor([[release_af]], dtype=th.float32))[0]
            self.q_values.append(q_values.numpy().mean())

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        observation = self._get_observation() if not done else np.zeros(3)
        truncated = False
        info = {}

        return observation, reward, done, truncated, info

    def render(self, mode='human', close=False):
        pass

# Load data and scale
data = pd.read_csv('/content/NAVAJORESERVOIR08-18-2024T16.48.23.csv')
storage = data['Storage (af)'].values
evaporation = data['Evaporation (af)'].values
inflow_cfs = data['Inflow** (cfs)'].values
inflow = inflow_cfs * 1.98211
release_cfs = data['Total Release (cfs)'].values
release = release_cfs * 1.98211

dataset = np.column_stack((storage, evaporation, inflow, release))
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)

env = ReservoirEnv(scaled_data, scaler)
env = DummyVecEnv([lambda: env])

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.15 * np.ones(n_actions))
model = TD3("MlpPolicy", env, action_noise=action_noise)

total_timesteps = 68000
model.learn(total_timesteps=total_timesteps)

# Save and reload
model.save("td3_reservoir_agent")
model = TD3.load("td3_reservoir_agent")

# Generate plots
plt.figure(figsize=(12, 6))
plt.plot(env.envs[0].storage_history, label="Predicted Storage", color='blue')
plt.plot(env.envs[0].actual_storage_history, label="Actual Storage", color='red', linestyle='--')
plt.xlabel('Days')
plt.ylabel('Storage (af)')
plt.title('Predicted vs Actual Storage')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(env.envs[0].evaporation_history, label="Evaporation", color='green', linestyle=':')
plt.plot(env.envs[0].inflow_history, label="Inflow", color='orange', linestyle='-.')
plt.xlabel('Days')
plt.ylabel('Inflow/Evaporation (af)')
plt.title('Inflow and Evaporation over Training Period')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(env.envs[0].reward_history, label="Daily Rewards", color='purple')
plt.xlabel('Days')
plt.ylabel('Reward')
plt.title('Rewards over Training Period')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(env.envs[0].q_values, label="Critic Q-values", color='darkcyan')
plt.xlabel('Timesteps')
plt.ylabel('Q-value')
plt.title('Critic Q-values over Training')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(env.envs[0].predicted_release_history, label="Predicted Release", color='purple')
plt.plot(env.envs[0].actual_release_history, label="Actual Release", color='orange', linestyle='--')
plt.axhline(y=navajo_min_release, color='gray', linestyle='--', label="Min Release Limit")
plt.axhline(y=navajo_max_release, color='gray', linestyle='--', label="Max Release Limit")
plt.xlabel('Days')
plt.ylabel('Release (af)')
plt.title('Predicted vs Actual Release')
plt.legend()
plt.show()