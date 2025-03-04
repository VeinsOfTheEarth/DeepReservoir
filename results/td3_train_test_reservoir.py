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
min_storage_af = 500000/1731750
max_storage_af = 1731750/1731750
min_release_af = (49 * 1.98211)/(5546*1.98211)  # Minimum release in acre-feet
max_release_af = (3000 * 1.98211)/(5546*1.98211)  # Maximum release in acre-feet

# Load and preprocess data
data = pd.read_csv('Clipped_NAVAJORESERVOIR08-18-2024T16.48.23.csv')
storage = data['Storage (af)'].values
evaporation = data['Evaporation (af)'].values
inflow_cfs = data['Inflow** (cfs)'].values
release_cfs = data['Total Release (cfs)'].values
dataset = np.column_stack((storage/1731750, evaporation/183, (inflow_cfs* 1.98211)/(15398*1.98211), (release_cfs * 1.98211)/(5546 * 1.98211)))

# Split dataset
train_data = dataset[:18000]  # Samples for training
test_data = dataset[18000:]   # Remaining samples for testing

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

        # Define action space and observation space
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=np.array([0,0,0]),
                                     high=np.array([1,1,1]),
                                     dtype=np.float32)

    def reset(self, seed=None):
        self.current_step = np.random.randint(0, len(self.data) - self.episode_length)  # Random episode start
        self.episode_step_count = 0
        self.reward_history.clear()
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
        self.current_storage = self.current_storage*1731750+(inflow_af*15398 - evaporation_af*183 - release_af_value*5546)
        self.current_storage = self.current_storage/1731750

        # Compute Reward
        if self.min_storage_af <= self.current_storage <= self.max_storage_af:
            reward = 1
        else:
            reward = -1

        self.reward_history.append(reward)
        self.storage_history.append(self.current_storage)
        self.current_step += 1
        self.episode_step_count += 1
        done = self.current_step >= len(self.data) or (self.episode_step_count >= self.episode_length)

        if done:
            self.episode_reward.append(sum(self.reward_history))
            self.mean_storage_history.append(np.mean(self.storage_history))
            self.release_history.append(release_af_value)
            self.mean_release_history.append(np.mean(self.release_history))
        
        observation = self._get_observation() if not done else np.zeros(3)
        return observation, reward, done, False, {}

# Set up environment
episode_length = 3600
env = ReservoirEnv(train_data, min_storage_af, max_storage_af, min_release_af, max_release_af, episode_length)

# Initialize model
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)

# Train the model
n_episodes = 500
total_timesteps = n_episodes * episode_length
model.learn(total_timesteps=total_timesteps)

# Save model
model.save("td3_reservoir_agent")

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
model = TD3.load("td3_reservoir_agent")

# Lists to store results
predicted_storage, actual_storage = [], []
predicted_release, actual_release = [], []

# Initialize current storage
current_storage = test_data[0,0]
print(current_storage)

# Perform predictions
for i in range(len(test_data)):
    # Get the observation from the dataset (without scaling)
    obs = np.array([current_storage, test_data[i, 1], test_data[i, 2]])
    # Predict the action using the trained agent
    action, _ = model.predict(obs)

    # Map the action to the corresponding release value in acre-feet
    normalized_release = (action[0] + 1) / 2
    predicted_release_af = min_release_af + normalized_release * (max_release_af - min_release_af)

    # Extract inflow and evaporation directly from the dataset (unscaled)
    inflow_af = test_data[i, 2]
    evaporation_af = test_data[i, 1]

    # Update the current storage using the predicted release, inflow, and evaporation
    current_storage = current_storage*1731750+(inflow_af*15398 - evaporation_af*183 - predicted_release_af*5546)
    current_storage = current_storage/1731750

    # Store results for analysis
    predicted_storage.append(current_storage)
    predicted_release.append(predicted_release_af)
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

# Save the results to CSV files
predicted_vs_actual_storage.to_csv("test_predicted_vs_actual_storage_complete_data.csv", index=False)
predicted_vs_actual_release.to_csv("test_predicted_vs_actual_release_complete_data.csv", index=False)
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
df = pd.read_csv('test_predicted_vs_actual_storage_complete_data.csv')

column_1 = df['Actual Storage (af)']
column_2 = df['Predicted Storage (af)']

column_1 = column_1 * 1731750
column_2 = column_2 * 1731750

# Rescaled storage plot
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

column_3 = column_3 * 5546
column_4 = column_4 * 5546

# Rescaled Rlease
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