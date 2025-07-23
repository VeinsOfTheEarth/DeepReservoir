# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 08:49:38 2025

@author: Shubhendu
"""
from datetime import date
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Setup paths
hydropower_model_path = "/content/drive/MyDrive/hydropower_model.py"
parameters_path = "/content/drive/MyDrive/parameters.pkl"
main_data_path = "/content/Clipped_NAVAJORESERVOIR08-18-2024T16.48.23.csv"

# Copy hydropower model and parameter file into working directory
import shutil
shutil.copy(hydropower_model_path, "./hydropower_model.py")
shutil.copy(parameters_path, "./parameters.pkl")

# Now start actual modeling
import os
import numpy as np
import pandas as pd
import torch as th
import tensorflow as tf
import gymnasium
from gymnasium import Env
from gymnasium.spaces import Box
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from hydropower_model import navajo_power_generation_model

# Load dataset
data = pd.read_csv('/content/Clipped_NAVAJORESERVOIR08-18-2024T16.48.23.csv')
storage = data['Storage (af)'].values
evaporation = data['Evaporation (af)'].values
inflow_af = data['Inflow** (cfs)'].values * 1.98211  # Convert to AF
release_af = data['Total Release (cfs)'].values * 1.98211  # Convert to AF
elevation = data['Elevation (feet)'].values
date = data['Date']

parsed_dates = pd.to_datetime(data['Date'], format="%d-%b-%y")
day_of_year = parsed_dates.dt.dayofyear.values
day_of_year_normalized = day_of_year / 366.0  # Simple normalization

# Train/Test Split
train_storage = storage[:18000]
train_evaporation = evaporation[:18000]
train_inflow = inflow_af[:18000]
train_release = release_af[:18000]
train_elevation = elevation[:18000]

# Calculate Mean and Std for Training Data
means = np.array([np.mean(train_storage), np.mean(train_evaporation), np.mean(train_inflow), np.mean(train_release), np.mean(train_elevation)])
stds = np.array([np.std(train_storage), np.std(train_evaporation), np.std(train_inflow), np.std(train_release), np.std(train_elevation)])

# Normalize Function
def normalize(data, mean, std):
    return (data - mean) / std

# Normalize All Data
storage = normalize(storage, means[0], stds[0])
evaporation = normalize(evaporation, means[1], stds[1])
inflow = normalize(inflow_af, means[2], stds[2])
release = normalize(release_af, means[3], stds[3])

# Combined Dataset
dataset = np.column_stack([storage, evaporation, inflow, release, day_of_year_normalized])
train_data = dataset[:18000]
test_data = dataset[18000:]

# Define storage and release limits
min_storage_af = normalize(500000, means[0], stds[0])
max_storage_af = normalize(1731750, means[0], stds[0])
min_release_af = normalize(0*1.98211, means[3], stds[3])
max_release_af = normalize(5000*1.98211, means[3], stds[3])

def storage_to_elevation(storage_val):
    min_elev, max_elev = 5900, 6100
    min_s, max_s = 500000, 1731750
    s_af = storage_val * stds[0] + means[0]
    return np.clip(min_elev + (max_elev - min_elev) * ((s_af - min_s) / (max_s - min_s)), min_elev, max_elev)

# Leap year check
def is_leap(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

# Agriculture demand function: constant from day 100–250, zero elsewhere
def generate_agriculture_demand(leap=False):
    days = 366 if leap else 365
    demand = np.zeros(days)
    daily = 250000 / 151  # constant for days 100–250 inclusive
    demand[99:250] = daily
    return demand, np.cumsum(demand)

agriculture_lookup_365, cumulative_365 = generate_agriculture_demand(False)
agriculture_lookup_366, cumulative_366 = generate_agriculture_demand(True)

# Define the custom Reservoir environment
class ReservoirEnv(Env):
    def __init__(self, dataset, min_storage_af, max_storage_af, min_release_af, max_release_af, episode_length):
        super(ReservoirEnv, self).__init__()
        self.data = dataset
        self.episode_length = episode_length
        self.min_storage_af = min_storage_af
        self.max_storage_af = max_storage_af
        self.min_release_af = min_release_af
        self.max_release_af = max_release_af
        self.reward_history, self.episode_reward = [], []
        self.storage_history, self.mean_storage_history = [], []
        self.release_history, self.mean_release_history = [], []
        self.current_step, self.episode_step_count = 0, 0
        self.cumulative_ag_release = 0
        self.last_doy = None

        # Define action space and observation space
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=np.array([-np.inf,-np.inf,-np.inf,-np.inf]),
                                     high=np.array([np.inf,np.inf,np.inf,np.inf]),
                                     dtype=np.float32)
        self.storage_rewards = []
        self.fisheries_rewards = []
        self.hydropower_rewards = []
        self.agriculture_rewards = []

        self.episode_storage_rewards = []
        self.episode_fisheries_rewards = []
        self.episode_hydropower_rewards = []
        self.episode_agriculture_rewards = []

    def reset(self, seed=None):
        self.current_step = np.random.randint(0, len(self.data) - self.episode_length)  # Random episode start
        self.episode_step_count = 0
        self.reward_history.clear()
        self.storage_history.clear()
        self.release_history.clear()
        self.cumulative_ag_release = 0
        self.last_doy = None
        self.current_storage = self.data[self.current_step, 0]
        self.prev_release = None  # Reset previous release

        self.storage_rewards.clear()
        self.fisheries_rewards.clear()
        self.hydropower_rewards.clear()
        self.agriculture_rewards.clear()

        return self._get_observation(), {}

    def _get_observation(self):
        return np.array([self.current_storage, self.data[self.current_step, 1], self.data[self.current_step, 2], self.data[self.current_step, 4]])

    def step(self, action):
        # Extract actions
        normalized_regular_release = (action[0] + 1) / 2
        normalized_divergent_release = (action[1] + 1) / 2

        # Convert normalized actions to actual release values
        regular_release_af = self.min_release_af + normalized_regular_release * (self.max_release_af - self.min_release_af)
        divergent_release_af = self.min_release_af + normalized_divergent_release * (self.max_release_af - self.min_release_af)

        # Total release
        total_release_af = regular_release_af + divergent_release_af

        inflow_af = self.data[self.current_step, 2]
        evaporation_af = self.data[self.current_step, 1]
        self.current_storage = (self.current_storage*stds[0]+means[0])+((inflow_af*stds[2]+means[2]) - (evaporation_af*stds[1]+means[1]) - (total_release_af*stds[3]+means[3]))
        self.current_storage = (self.current_storage-means[0])/stds[0]

        # Predict elevation from storage
        predicted_elevation_rescaled = storage_to_elevation(self.current_storage)

        # Calculate hydropower
        date_str = date[self.current_step]
        release_hydropower_cfs = (divergent_release_af * stds[3] + means[3])/ 1.98211
        hydropower_generation = navajo_power_generation_model([pd.to_datetime(date_str, format="%d-%b-%y")],[release_hydropower_cfs], [predicted_elevation_rescaled])[0]

        # Compute Storage Reward
        if self.min_storage_af <= self.current_storage <= self.max_storage_af:
          target_storage = (self.max_storage_af + self.min_storage_af) / 2
          r_storage = 1 - abs(self.current_storage - target_storage) / (self.max_storage_af - self.min_storage_af)
        else:
          r_storage = -1
        self.storage_rewards.append(r_storage)
        reward = r_storage # base reward

        # Penalize for total release exceeding 5000 cfs
        if total_release_af >= max_release_af:
            reward -= 0.8
        else:
            reward += 0.5

        # Penalize for fisheries releasing below 350 cfs
        fisheries_threshold_af = 350 * 1.98211
        fisheries_threshold_normalized = (fisheries_threshold_af - means[3]) / stds[3]
        # Apply reward/penalty only to fisheries release
        if divergent_release_af >= fisheries_threshold_normalized:
            r_fisheries = 0.5
        else:
            r_fisheries = -0.2

        self.fisheries_rewards.append(r_fisheries)
        reward += r_fisheries

        min_mwh = 288
        max_mwh = 768

        if hydropower_generation < min_mwh:
            r_hydro = -0.5  # not enough generation

        elif hydropower_generation > max_mwh:
            r_hydro = -0.5  # unrealistic, turbine capacity exceeded
        else:
            r_hydro = (hydropower_generation - min_mwh) / (max_mwh - min_mwh)  # normalized bonus

        self.hydropower_rewards.append(r_hydro)
        reward += r_hydro

        # Agricultural Reward
        dt = pd.to_datetime(date_str, format="%d-%b-%y")
        leap = is_leap(dt.year)
        doy = dt.timetuple().tm_yday
        ag_curve = agriculture_lookup_366 if leap else agriculture_lookup_365
        cum_curve = cumulative_366 if leap else cumulative_365
        today_demand = ag_curve[doy - 1]
        cum_demand = cum_curve[doy - 1]
        reg_release_af = regular_release_af * stds[3] + means[3]

        # Reset cumulative release if new year starts
        r_ag = 0
        if 100 <= doy <= 250:
          self.cumulative_ag_release += reg_release_af
          delta = cum_demand - self.cumulative_ag_release
          r_ag= 1 - abs(delta) / 250000
          reward += r_ag
        else:
          self.cumulative_ag_release = 0
        self.agriculture_rewards.append(r_ag)

        self.prev_release = total_release_af
        self.reward_history.append(reward)
        self.storage_history.append(self.current_storage)
        self.current_step += 1
        self.episode_step_count += 1
        done = self.current_step >= len(self.data) or (self.episode_step_count >= self.episode_length)

        if done:
            self.episode_reward.append(sum(self.reward_history))
            self.mean_storage_history.append(np.mean(self.storage_history))
            self.release_history.append(total_release_af)
            self.mean_release_history.append(np.mean(self.release_history))

            self.episode_storage_rewards.append(np.mean(self.storage_rewards))
            self.episode_fisheries_rewards.append(np.mean(self.fisheries_rewards))
            self.episode_hydropower_rewards.append(np.mean(self.hydropower_rewards))
            self.episode_agriculture_rewards.append(np.mean(self.agriculture_rewards))

        observation = self._get_observation() if not done else np.zeros(4)
        return observation, reward, done, False, {}

# Set up environment
episode_length = 3600
env = ReservoirEnv(train_data, min_storage_af, max_storage_af, min_release_af, max_release_af, episode_length)

# Initialize model
model = PPO("MlpPolicy", env,  verbose=1)

# Train the model
n_episodes = 500
total_timesteps = n_episodes * episode_length
model.learn(total_timesteps=total_timesteps)

# Save model
model.save("ppo_reservoir_agent")

# Save cumulative reward values to CSV
pd.DataFrame(env.episode_reward).to_csv("cumulative_reward_per_episode.csv", index=False)

# Plot cumulative rewards per episode
plt.figure(figsize=(12, 6))
plt.plot(env.episode_reward, label="Cumulative Reward per Episode", color="blue")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward per Episode vs Episode Number")
plt.grid()
plt.legend()
plt.savefig("cumulative_rewards_per_episode.png")
plt.show()

# Plot mean storage per episode
plt.figure(figsize=(12, 6))
plt.plot(env.mean_storage_history, label="Mean Storage per Episode", color="blue")
plt.xlabel("Episode")
plt.ylabel("Mean Storage")
plt.title("Mean Storage per Episode vs Episode Number")
plt.grid()
plt.legend()
plt.savefig("mean_storage_per_episode.png")
plt.show()

# Plot mean release per episode
plt.figure(figsize=(12, 6))
plt.plot(env.mean_release_history, label="Mean Release per Episode", color="blue")
plt.xlabel("Episode")
plt.ylabel("Mean Release")
plt.title("Mean Release per Episode vs Episode Number")
plt.grid()
plt.legend()
plt.savefig("mean_release_per_episode.png")
plt.show()

# Plot all reward components per episode
plt.figure(figsize=(14, 7))
plt.plot(env.episode_reward, label="Total Reward", linewidth=2, color='black')
plt.plot(env.episode_storage_rewards, label="Storage Reward", linestyle="--")
plt.plot(env.episode_fisheries_rewards, label="Fisheries Reward", linestyle="--")
plt.plot(env.episode_hydropower_rewards, label="Hydropower Reward", linestyle="--")
plt.plot(env.episode_agriculture_rewards, label="Agriculture Reward", linestyle="--")
plt.xlabel("Episode")
plt.ylabel("Mean Reward per Episode")
plt.title("Mean Episode Reward Components")
plt.grid(True)
plt.legend()
plt.savefig("component_rewards_per_episode.png")
plt.show()

print("Training complete. Model, Storage, Release, and Reward plots saved.")
