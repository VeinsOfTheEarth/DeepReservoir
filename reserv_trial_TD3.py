# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:06:41 2024

@author: Shubhendu
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
from stable_baselines3 import TD3, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, plot_results,ts2xy
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Define the advanced environment with a custom reward function
class ReservoirEnv(Env):
    def __init__(self, data, scaler, target_scaler):
        super(ReservoirEnv, self).__init__()

        self.data = data
        self.scaler = scaler
        self.target_scaler = target_scaler
        self.current_step = 0

        # Define action and observation space
        self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=0, high=1, shape=(4,), dtype=np.float32)

    def reset(self, seed=None):
        self.current_step = 0
        info={}
        return self._get_observation(),info

    def _get_observation(self):
        obs = self.data[self.current_step, :-1]
        return obs

    def step(self, action):
        release_scaled = action[0]
        # release = self.target_scaler.inverse_transform([[release_scaled]])[0, 0]

        target_release_scaled = self.data[self.current_step, -1]
        # target_release = self.target_scaler.inverse_transform([[target_release_scaled]])[0, 0]

        # Reward function: Reward if release is within a certain range
        acceptable_range = 0.05 * target_release_scaled  # Allow 5% deviation
        if abs(release_scaled - target_release_scaled) <= acceptable_range:
            reward = 1.0  # High reward for being within the range
        else:
            reward = -np.square(release_scaled - target_release_scaled)  # Penalize based on squared error

        self.current_step += 1
        done = self.current_step >= len(self.data)
        if not done:
            observation = self._get_observation()
        else:
            observation = np.zeros(4)
        truncated=False
        info={}
        return observation, reward, done, truncated, info

    def render(self, mode='human', close=False):
        pass

# Load the data
data = pd.read_csv('/content/NAVAJORESERVOIR08-18-2024T16.48.23.csv')
elevation = data['Elevation (feet)'].values
storage = data['Storage (af)'].values
evaporation = data['Evaporation (af)'].values
inflow = data['Inflow** (cfs)'].values
release = data['Total Release (cfs)'].values

dataset = np.column_stack((elevation, storage, evaporation, inflow, release))

# Scale the inputs and outputs between 0 and 1
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)

features = dataset[:, :-1]
target = dataset[:, -1].reshape(-1, 1)
features_scaled = scaler.fit_transform(features)
target_scaler = MinMaxScaler()
target_scaled = target_scaler.fit_transform(target)

scaled_data = np.column_stack((features_scaled, target_scaled))

# Create the environment
env = ReservoirEnv(scaled_data, scaler, target_scaler)
env = DummyVecEnv([lambda: env])  # Wrap the environment

# Create a noise object for exploration
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Instantiate the TD3 model with verbose output
model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the model
model.save("td3_reservoir_agent")
model = TD3.load("td3_reservoir_agent")

# Test the model (optional)
obs = env.reset()
for i in range(len(scaled_data)):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        break

print("Test completed.")

#%%

scaled_actual_releases = []
scaled_predicted_releases = []

# Test the trained agent and collect scaled predictions
for i in range(len(features_scaled)):
    # Predict the action (release) for the current observation
    action, _states = model.predict(features_scaled[i])

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
plt.ylabel('Scaled Total Release (cfs)')
plt.title('Agent Predicted Releases vs Actual Releases')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Assuming 'rescaled_predicted_releases' and 'rescaled_actual_releases' contain the rescaled predicted and actual releases

# Convert lists to numpy arrays for metric calculations
scaled_predicted_releases = np.array(scaled_predicted_releases)
scaled_actual_releases = np.array(scaled_actual_releases)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(scaled_actual_releases, scaled_predicted_releases)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(scaled_actual_releases, scaled_predicted_releases)
print(f"Mean Absolute Error (MAE): {mae}")

# Additional metric: Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((scaled_actual_releases - scaled_predicted_releases) / scaled_actual_releases)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Optional: R-squared (Coefficient of Determination)
ss_res = np.sum((scaled_actual_releases - scaled_predicted_releases) ** 2)
ss_tot = np.sum((scaled_actual_releases - np.mean(scaled_actual_releases)) ** 2)
r2_score = 1 - (ss_res / ss_tot)
print(f"R-squared (R2): {r2_score:.4f}")

#%%

# Initialize lists to store the rescaled actual and predicted releases
rescaled_actual_releases = []
rescaled_predicted_releases = []

# Test the trained agent and collect predictions
for i in range(len(scaled_data)):
    # Predict the action (release) for the current observation
    action, _states = model.predict(obs)

    # The action is in scaled form; rescale it to the original value
    # Remove the extra brackets to pass a 2D array
    release_predicted = target_scaler.inverse_transform([action[0]])[0, 0]

    # Get the actual release value from the scaled data (last element of the observation)
    # Extract the single value from the nested array
    actual_release_scaled = obs[0][-1]
    actual_release = target_scaler.inverse_transform([[actual_release_scaled]])[0, 0]

    # Store the rescaled results
    rescaled_predicted_releases.append(release_predicted)
    rescaled_actual_releases.append(actual_release)

    # Step the environment
    obs, rewards, done, info = env.step(action)
    if done:
        break

# Plot the rescaled predicted vs. actual releases
plt.figure(figsize=(14, 7))
plt.plot(rescaled_actual_releases, label='Actual Total Release (cfs)', color='blue')
plt.plot(rescaled_predicted_releases, label='Predicted Total Release (cfs)', color='red')
plt.xlabel('Time Step')
plt.ylabel('Total Release (cfs)')
plt.title('Rescaled Agent Predicted Releases vs Rescaled Actual Releases')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Assuming 'rescaled_predicted_releases' and 'rescaled_actual_releases' contain the rescaled predicted and actual releases

# Convert lists to numpy arrays for metric calculations
rescaled_predicted_releases = np.array(rescaled_predicted_releases)
rescaled_actual_releases = np.array(rescaled_actual_releases)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(rescaled_actual_releases, rescaled_predicted_releases)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(rescaled_actual_releases, rescaled_predicted_releases)
print(f"Mean Absolute Error (MAE): {mae}")

# Additional metric: Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((rescaled_actual_releases - rescaled_predicted_releases) / rescaled_actual_releases)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Optional: R-squared (Coefficient of Determination)
ss_res = np.sum((rescaled_actual_releases - rescaled_predicted_releases) ** 2)
ss_tot = np.sum((rescaled_actual_releases - np.mean(rescaled_actual_releases)) ** 2)
r2_score = 1 - (ss_res / ss_tot)
print(f"R-squared (R2): {r2_score:.4f}")
