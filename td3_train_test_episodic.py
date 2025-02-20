# Import necessary libraries
import os
import numpy as np
import pandas as pd
import torch as th
import gymnasium
from gymnasium import Env
from gymnasium.spaces import Box
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import matplotlib.pyplot as plt

# Define storage and release limits
min_storage_af = 500000
max_storage_af = 1800000
min_release_af = 500 * 1.98211  # Minimum release in acre-feet
max_release_af = 4000 * 1.98211  # Maximum release in acre-feet

# Load and preprocess data
data = pd.read_csv('Clipped_NAVAJORESERVOIR08-18-2024T16.48.23.csv')
storage = data['Storage (af)'].values
evaporation = data['Evaporation (af)'].values
inflow_cfs = data['Inflow** (cfs)'].values
release_cfs = data['Total Release (cfs)'].values
dataset = np.column_stack((storage, evaporation, inflow_cfs * 1.98211, release_cfs * 1.98211))

# Split dataset
train_data = dataset[:18000]  # First 18,000 samples for training
test_data = dataset[18000:]   # Remaining samples for testing

# Define the custom Reservoir environment
class ReservoirEnv(Env):
    def __init__(self, dataset, min_storage_af, max_storage_af, min_release_af, max_release_af, episode_length, td3_model):
        super(ReservoirEnv, self).__init__()
        self.data = dataset
        self.episode_length = episode_length
        self.min_storage_af = min_storage_af
        self.max_storage_af = max_storage_af
        self.min_release_af = min_release_af
        self.max_release_af = max_release_af
        self.td3_model = td3_model  # To compute Q-values
        self.reward_history, self.episode_reward = [], []
        self.storage_history, self.mean_storage_history = [], []
        self.q_values, self.cumulative_q_values = [], []
        self.release_history, self.mean_release_history = [], []
        self.current_step, self.episode_step_count = 0, 0

        # Define action space and observation space
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=np.array([self.min_storage_af, np.min(self.data[:, 1]), np.min(self.data[:, 2])]),
                                     high=np.array([self.max_storage_af, np.max(self.data[:, 1]), np.max(self.data[:, 2])]),
                                     dtype=np.float32)

    def reset(self, seed=None):
        self.current_step = np.random.randint(0, len(self.data) - self.episode_length)  # Random episode start
        self.episode_step_count = 0
        self.reward_history.clear()
        self.q_values.clear()
        self.storage_history.clear()
        self.release_history.clear()
        self.current_storage = self.data[self.current_step, 0]
        return self._get_observation(), {}

    def _get_observation(self):
        return np.array([self.current_storage, self.data[self.current_step, 1], self.data[self.current_step, 2]])

    def step(self, action):
        normalized_release = (action[0] + 1) / 2
        release_af_value = self.min_release_af + normalized_release * (self.max_release_af - self.min_release_af)

        inflow_af = self.data[self.current_step, 2]
        evaporation_af = self.data[self.current_step, 1]
        self.current_storage += inflow_af - evaporation_af - release_af_value

        # Compute Reward
        reward = -abs(self.current_storage - (self.max_storage_af + self.min_storage_af) / 2) / self.max_storage_af

        # Compute Q-value
        obs_tensor = th.tensor(self._get_observation(), dtype=th.float32).unsqueeze(0).to(self.td3_model.device)
        action_tensor = th.tensor(action, dtype=th.float32).unsqueeze(0).to(self.td3_model.device)
        with th.no_grad():
            q_value = self.td3_model.critic(obs_tensor, action_tensor)[0].mean().item()
        self.q_values.append(q_value)

        self.reward_history.append(reward)
        self.storage_history.append(self.current_storage)
        self.current_step += 1
        self.episode_step_count += 1
        done = self.episode_step_count >= self.episode_length
        done = self.current_step >= len(self.data) or (self.episode_step_count >= self.episode_length)

        if done:
            self.episode_reward.append(sum(self.reward_history))
            self.mean_storage_history.append(np.mean(self.storage_history))
            self.cumulative_q_values.append(np.sum(self.q_values))
            self.release_history.append(release_af_value)
            self.mean_release_history.append(np.mean(self.release_history))

        return self._get_observation() if not done else np.zeros(3), reward, done, False, {}

# Set up environment
episode_length = 120
env = ReservoirEnv(train_data, min_storage_af, max_storage_af, min_release_af, max_release_af, episode_length, None)

# Initialize model
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
model = TD3("MlpPolicy", env, policy_kwargs={"net_arch": [400, 300]}, action_noise=action_noise, verbose=1)

# Pass the TD3 model to the environment
env.td3_model = model  

# Train the model
n_episodes = 1000
total_timesteps = n_episodes * episode_length
model.learn(total_timesteps=total_timesteps)

# Save model
model.save("td3_reservoir_agent")

# Save cumulative Q-values to CSV
pd.DataFrame(env.cumulative_q_values, columns=["Cumulative Q-Value"]).to_csv("cumulative_q_values_per_episode.csv", index=False)

# Save cumulative reward values to CSV
pd.DataFrame(env.episode_reward).to_csv("cumulative_reward_per_episode.csv", index=False)

# Plot cumulative Q-values per episode
plt.figure(figsize=(12, 6))
plt.plot(env.cumulative_q_values, label="Cumulative Q-Value per Episode", color="orange")
plt.xlabel("Episode")
plt.ylabel("Cumulative Q-Value")
plt.title("Cumulative Q-Value per Episode vs Episode Number")
plt.grid()
plt.legend()
plt.savefig("cumulative_q_values_per_episode.png")
plt.show()

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

print("Training complete. Model, Q values, release, and reward plots saved.")

# Load trained model
model = TD3.load("td3_reservoir_agent")

# Set up test environment
test_env = ReservoirEnv(test_data, min_storage_af, max_storage_af, min_release_af, max_release_af, episode_length, model)

# Lists to store results
predicted_storage, actual_storage = [], []
predicted_release, actual_release = [], []

# Perform predictions on sampled 120-day episodes from test data
num_episodes = len(test_data) // episode_length  # Maximum possible episodes

for _ in range(num_episodes):
    obs, _ = test_env.reset()
    
    for _ in range(episode_length):
        action, _ = model.predict(obs, deterministic=True)  # Use deterministic actions during testing
        obs, reward, done, _, _ = test_env.step(action)

        # Store results
        predicted_storage.append(obs[0])  # Predicted storage (first element of obs)
        
        # Access the release value after calling step
        normalized_release = (action[0] + 1) / 2
        predicted_release_af = test_env.min_release_af + normalized_release * (test_env.max_release_af - test_env.min_release_af) # Calculate predicted release
        predicted_release.append(predicted_release_af)
        
        actual_storage.append(test_env.data[test_env.current_step, 0])  # True storage from dataset
        actual_release.append(test_env.data[test_env.current_step, 3])  # True release from dataset

        if done:
            break  # End episode if max steps reached

# Save results
pd.DataFrame({"Predicted Storage (af)": predicted_storage, "Actual Storage (af)": actual_storage}).to_csv("predicted_vs_actual_storage.csv", index=False)
pd.DataFrame({"Predicted Release (af)": predicted_release, "Actual Release (af)": actual_release}).to_csv("predicted_vs_actual_release.csv", index=False)

print("Testing complete.")

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

# # Load trained model
# model = TD3.load("td3_reservoir_agent")

# # Set up variables for predictions
# predicted_storage, actual_storage = [], []
# predicted_release, actual_release = [], []

# # Set up testing parameters
# episode_length = 120
# num_episodes = len(test_data) // episode_length  # Maximum number of episodes possible

# # Perform predictions on sampled 120-day episodes from test data
# for _ in range(num_episodes):
#     start_idx = np.random.randint(0, len(test_data) - episode_length)
#     current_storage = test_data[start_idx, 0]

#     for i in range(start_idx, start_idx + episode_length):
#         observation = np.array([current_storage, test_data[i, 1], test_data[i, 2]], dtype=np.float32)

#         action, _ = model.predict(observation)
#         normalized_release = (action[0] + 1) / 2
#         predicted_release_af = min_release_af + normalized_release * (max_release_af - min_release_af)

#         inflow_af = test_data[i, 2]
#         evaporation_af = test_data[i, 1]
#         current_storage = current_storage + inflow_af - evaporation_af - predicted_release_af

#         predicted_storage.append(current_storage)
#         predicted_release.append(predicted_release_af)
#         actual_storage.append(test_data[i, 0])
#         actual_release.append(test_data[i, 3])

# # Save results
# pd.DataFrame({"Predicted Storage (af)": predicted_storage, "Actual Storage (af)": actual_storage}).to_csv("predicted_vs_actual_storage.csv", index=False)
# pd.DataFrame({"Predicted Release (af)": predicted_release, "Actual Release (af)": actual_release}).to_csv("predicted_vs_actual_release.csv", index=False)

# print("Testing complete.")

# # Plot Predicted vs Actual Storage
# plt.figure(figsize=(12, 6))
# plt.plot(actual_storage, label="Actual Storage (af)", color="blue")
# plt.plot(predicted_storage, label="Predicted Storage (af)", color="orange")
# plt.xlabel("Timestep")
# plt.ylabel("Storage (af)")
# plt.title("Predicted vs Actual Storage")
# plt.legend()
# plt.grid()
# plt.savefig("Predicted vs Actual Storage.png")
# plt.show()

# # Plot Predicted vs Actual Release
# plt.figure(figsize=(12, 6))
# plt.plot(actual_release, label="Actual Release (af)", color="blue")
# plt.plot(predicted_release, label="Predicted Release (af)", color="orange")
# plt.xlabel("Timestep")
# plt.ylabel("Release (af)")
# plt.title("Predicted vs Actual Release")
# plt.legend()
# plt.grid()
# plt.savefig("Predicted vs Actual Release.png")
# plt.show()