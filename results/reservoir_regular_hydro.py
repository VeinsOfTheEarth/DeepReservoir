# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 23:54:38 2025

@author: Shubhendu
"""
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
import gymnasium
from gymnasium import Env
from gymnasium.spaces import Box
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from hydropower_model import navajo_power_generation_model

# Load dataset
data = pd.read_csv(main_data_path)
storage = data['Storage (af)'].values
evaporation = data['Evaporation (af)'].values
inflow_af = data['Inflow** (cfs)'].values * 1.98211
release_af = data['Total Release (cfs)'].values * 1.98211

# Split training data
train_storage = storage[:18000]
train_evaporation = evaporation[:18000]
train_inflow = inflow_af[:18000]
train_release = release_af[:18000]

# Normalization
means = np.array([np.mean(train_storage), np.mean(train_evaporation), np.mean(train_inflow), np.mean(train_release)])
stds = np.array([np.std(train_storage), np.std(train_evaporation), np.std(train_inflow), np.std(train_release)])
def normalize(data, mean, std): return (data - mean) / std
storage = normalize(storage, means[0], stds[0])
evaporation = normalize(evaporation, means[1], stds[1])
inflow = normalize(inflow_af, means[2], stds[2])
release = normalize(release_af, means[3], stds[3])
dataset = np.column_stack([storage, evaporation, inflow, release])
train_data, test_data = dataset[:18000], dataset[18000:]

min_storage_af = normalize(500000, means[0], stds[0])
max_storage_af = normalize(1731750, means[0], stds[0])
min_release_af = normalize(0, means[3], stds[3])
max_release_af = normalize(5000 * 1.98211, means[3], stds[3])

def storage_to_elevation(storage_val):
    min_elev, max_elev = 5900, 6100
    min_s, max_s = 500000, 1731750
    s_af = storage_val * stds[0] + means[0]
    return np.clip(min_elev + (max_elev - min_elev) * ((s_af - min_s) / (max_s - min_s)), min_elev, max_elev)

class ReservoirEnv(Env):
    def __init__(self, dataset, min_s, max_s, min_r, max_r, episode_length):
        super().__init__()
        self.data = dataset
        self.min_s, self.max_s = min_s, max_s
        self.min_r, self.max_r = min_r, max_r
        self.episode_length = episode_length
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=np.array([-np.inf]*3), high=np.array([np.inf]*3), dtype=np.float32)
        self.reset()

    def reset(self, seed=None):
        self.step_idx = np.random.randint(0, len(self.data) - self.episode_length)
        self.ep_step, self.current_storage = 0, self.data[self.step_idx, 0]
        self.rewards, self.storages, self.releases = [], [], []
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.current_storage, self.data[self.step_idx, 1], self.data[self.step_idx, 2]])

    def step(self, action):
        reg_r = (action[0] + 1) / 2
        hyd_r = (action[1] + 1) / 2
        regular_release_af = self.min_r + reg_r * (self.max_r - self.min_r)
        hydropower_release_af = self.min_r + hyd_r * (self.max_r - self.min_r)
        total_release_af = regular_release_af + hydropower_release_af
        inflow_af = self.data[self.step_idx, 2]
        evaporation_af = self.data[self.step_idx, 1]
        self.current_storage = (self.current_storage*stds[0]+means[0])+((inflow_af*stds[2]+means[2]) - (evaporation_af*stds[1]+means[1]) - (total_release_af*stds[3]+means[3]))
        self.current_storage = (self.current_storage - means[0]) / stds[0]

        reward = -1
        if self.min_s <= self.current_storage <= self.max_s:
            tgt = (self.max_s + self.min_s) / 2
            reward = 1 - abs(self.current_storage - tgt) / (self.max_s - self.min_s)

        if total_release_af >= max_release_af:
            reward -= 0.8
        else:
            reward += 0.5

        # Optional energy reward
        elev_ft = storage_to_elevation(self.current_storage)
        cfs_flow = (hydropower_release_af * stds[3] + means[3]) / 1.98211
        date = pd.Timestamp("2008-01-01") + pd.Timedelta(days=self.step_idx)
        energy = navajo_power_generation_model([date], [cfs_flow], [elev_ft])[0]
        reward += energy / 100.0

        self.step_idx += 1
        self.ep_step += 1
        self.rewards.append(reward)
        self.storages.append(self.current_storage)
        self.releases.append(total_release_af)
        done = self.ep_step >= self.episode_length or self.step_idx >= len(self.data)
        return (self._get_obs() if not done else np.zeros(3)), reward, done, False, {}

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

print("Training complete. Model, Storage, Release, and Reward plots saved.")

# Load trained model
model = PPO.load("ppo_reservoir_agent")

# Lists to store results
predicted_storage, actual_storage = [], []
predicted_release, actual_release = [], []
regular_releases = []
fisheries_releases = []

# Initialize current storage
current_storage = test_data[0,0]
print(current_storage)

# Low release thershold value
low_release_threshold = 350*1.98211

# Perform predictions
for i in range(len(test_data)):
    # Get the observation from the dataset (without scaling)
    obs = np.array([current_storage, test_data[i, 1], test_data[i, 2]])
    # Predict the action using the trained agent
    action, _ = model.predict(obs)

    # Extract and denormalize actions
    normalized_regular_release = (action[0] + 1) / 2
    normalized_fisheries_release = (action[1] + 1) / 2

    regular_release_af = min_release_af + normalized_regular_release * (max_release_af - min_release_af)
    fisheries_release_af = min_release_af + normalized_fisheries_release * (max_release_af - min_release_af)

    # Total release
    total_release_af = regular_release_af + fisheries_release_af

    # Extract inflow and evaporation directly from the dataset (unscaled)
    inflow_af = test_data[i, 2]
    evaporation_af = test_data[i, 1]

    # Update the current storage using the predicted release, inflow, and evaporation
    current_storage = (current_storage*stds[0]+means[0])+((inflow_af*stds[2]+means[2]) - (evaporation_af*stds[1]+means[1]) - (total_release_af*stds[3]+means[3]))
    current_storage = (current_storage-means[0])/stds[0]

    # Store results for analysis
    predicted_storage.append(current_storage)
    regular_releases.append(regular_release_af)
    fisheries_releases.append(fisheries_release_af)
    predicted_release.append(total_release_af)
    actual_storage.append(test_data[i, 0])  # Actual storage from the dataset
    actual_release.append(test_data[i, 3])  # Actual release from the dataset
    # print(f"Step {i}: Observation: {observation}, Predicted Release: {predicted_release_af}, Current Storage: {current_storage}")

# Save results to CSV files
predicted_vs_actual_storage = pd.DataFrame({
    "Predicted Storage (af)": predicted_storage,
    "Actual Storage (af)": actual_storage
})
predicted_vs_actual_release = pd.DataFrame({
    "Predicted Release (af)": predicted_release,
    "Actual Release (af)": actual_release
})

regular_vs_fisheries_release = pd.DataFrame({
    'Regular Release (af)': regular_releases,
    'Fisheries Release (af)': fisheries_releases
})

# Save the results to CSV files
predicted_vs_actual_storage.to_csv("test_predicted_vs_actual_storage_complete_data.csv", index=False)
predicted_vs_actual_release.to_csv("test_predicted_vs_actual_release_complete_data.csv", index=False)
regular_vs_fisheries_release.to_csv("test_regular_vs_fisheries_release_complete_data.csv", index=False)
pd.DataFrame(dataset).to_csv("test_complete_dataset.csv", index=False)

print("Prediction completed and saved to CSV files.")

# Plot Predicted vs Actual Storage
plt.figure(figsize=(12, 6))
plt.plot(actual_storage, label="Actual Storage (af)", color="blue")
plt.plot(predicted_storage, label="Predicted Storage (af)", color="orange")
plt.xlabel("Timestep")
plt.ylabel("Storage (af)")
plt.title("Predicted vs Actual Storage")
plt.legend()
plt.grid()
plt.savefig("test_predicted_vs_actual_storage.png")
plt.show()

# Plot Predicted vs Actual Release
plt.figure(figsize=(12, 6))
plt.plot(actual_release, label="Actual Release (af)", color="blue")
plt.plot(predicted_release, label="Predicted Release (af)", color="orange")
plt.xlabel("Timestep")
plt.ylabel("Release (af)")
plt.title("Predicted vs Actual Release")
plt.legend()
plt.grid()
plt.savefig("test_predicted_vs_actual_release.png")
plt.show()

# Rescaled plots
# Read the CSV file
df = pd.read_csv('/content/test_predicted_vs_actual_storage_complete_data.csv')

column_1 = df['Actual Storage (af)']
column_2 = df['Predicted Storage (af)']

column_1 = column_1 * stds[0] + means[0]
column_2 = column_2 * stds[0] + means[0]

# Rescaled Storage
plt.figure(figsize=(10, 6))
plt.plot(df.index, column_1, label='Actual Storage(af)', color='blue')
plt.plot(df.index, column_2, label='Predicted Storage (af)', color='orange')
plt.axhline(y=500000, color="r", linestyle="--", label=f"Min Storage Limit {500000} AF")
plt.axhline(y=1731750, color="g", linestyle="--", label=f"Max Storage Limit {1731750} AF")
plt.xlabel("Timestep")
plt.ylabel("Storage (af)")
plt.title("Predicted vs Actual Storage")
plt.legend()
plt.grid()
plt.savefig("test_predicted_vs_actual_storage_rescaled.png")
plt.show()

df_1 = pd.read_csv('test_predicted_vs_actual_release_complete_data.csv')

column_3 = df_1['Actual Release (af)']
column_4 = df_1['Predicted Release (af)']

column_3 = column_3 * stds[3] + means[3]
column_4 = column_4 * stds[3] + means[3]

# Rescaled Release
plt.figure(figsize=(10, 6))
plt.plot(df_1.index, column_3, label='Actual Release (af)', color='blue')
plt.plot(df_1.index, column_4, label='Predicted Release (af)', color='orange')
plt.xlabel("Timestep")
plt.ylabel("Release (af)")
plt.title("Predicted vs Actual Release")
plt.legend()
plt.grid()
plt.savefig("test_predicted_vs_actual_release_rescaled.png")
plt.show()

df_2 = pd.read_csv('test_regular_vs_fisheries_release_complete_data.csv')

column_5 = df_2['Regular Release (af)']
column_6 = df_2['Fisheries Release (af)']

column_5 = column_5 * stds[3] + means[3]
column_6 = column_6 * stds[3] + means[3]

# Rescaled Release
plt.figure(figsize=(10, 6))
plt.plot(df_2.index, column_5, label='Regular Release (af)', color='blue')
plt.plot(df_2.index, column_6, label='Fisheries Release (af)', color='orange')
plt.xlabel("Timestep")
plt.ylabel("Release (af)")
plt.title("Regular vs Fisheries Release")
plt.legend()
plt.grid()
plt.savefig("test_regular_vs_fisheries_release_rescaled.png")
plt.show()