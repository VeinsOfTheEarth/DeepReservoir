# -*- coding: utf-8 -*-
"""
@author: Shubhendu
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch as th
import tensorflow as tf
import gymnasium
from datetime import date
from gymnasium import Env
from gymnasium.spaces import Box
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from hydropower_model import navajo_power_generation_model
from niip_demand import niip_daily_demand

# Load main data
import pandas as pd
from importlib import reload
# reload(loader)
from deepreservoir.data import loader
data = loader.NavajoData()

# one-shot load
everything = data.load_all(include_cont_streamflow=False, model_data=True)
model_data = everything["model_data"]





storage = data['Storage (af)'].values
evaporation = data['Evaporation (af)'].values
inflow_af = data['Inflow** (cfs)'].values * 1.98211
release_af = data['Total Release (cfs)'].values * 1.98211
elevation = data['Elevation (feet)'].values
date = data['Date']

parsed_dates = pd.to_datetime(date, format="%d-%b-%y")
day_of_year = parsed_dates.dt.dayofyear.values
day_of_year_normalized = day_of_year / 366.0

# Train/Test Split
train_storage = storage[:18000]
train_evaporation = evaporation[:18000]
train_inflow = inflow_af[:18000]
train_release = release_af[:18000]
train_elevation = elevation[:18000]

# Calculate Mean and Std for Training Data
means = np.array([np.mean(train_storage), np.mean(train_evaporation),
                  np.mean(train_inflow), np.mean(train_release), np.mean(train_elevation)])
stds = np.array([np.std(train_storage), np.std(train_evaporation),
                 np.std(train_inflow), np.std(train_release), np.std(train_elevation)])

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
min_release_af = normalize(0 * 1.98211, means[3], stds[3])
max_release_af = normalize(5000 * 1.98211, means[3], stds[3])

def storage_to_elevation(storage_val):
    min_elev, max_elev = 5900, 6100
    min_s, max_s = 500000, 1731750
    s_af = storage_val * stds[0] + means[0]
    return np.clip(min_elev + (max_elev - min_elev) * ((s_af - min_s) / (max_s - min_s)), min_elev, max_elev)

# Leap year check
def is_leap(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

# Define the custom Reservoir environment
class ReservoirEnv(Env):
    def __init__(self, dataset, min_storage_af, max_storage_af, min_release_af, max_release_af, episode_length):
        super().__init__()
        self.data = dataset
        self.episode_length = episode_length
        self.min_storage_af = min_storage_af
        self.max_storage_af = max_storage_af
        self.min_release_af = min_release_af
        self.max_release_af = max_release_af
        self.reward_history, self.episode_reward = [], []
        self.storage_history, self.mean_storage_history = [], []
        self.release_history, self.mean_release_history = [], []

        # Define action space and observation space
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=np.array([-np.inf,-np.inf,-np.inf,-np.inf]),
                                     high=np.array([np.inf,np.inf,np.inf,np.inf]),
                                     dtype=np.float32)

        self.storage_rewards, self.fisheries_rewards = [], []
        self.hydropower_rewards, self.niip_rewards = [], []
        self.episode_storage_rewards, self.episode_fisheries_rewards = [], []
        self.episode_hydropower_rewards, self.episode_niip_rewards = [], []

    def reset(self, seed=None):
        self.current_step = np.random.randint(0, len(self.data) - self.episode_length)
        self.episode_step_count = 0
        self.reward_history.clear()
        self.storage_history.clear()
        self.release_history.clear()
        self.current_storage = self.data[self.current_step, 0]

        self.storage_rewards.clear()
        self.fisheries_rewards.clear()
        self.hydropower_rewards.clear()
        self.niip_rewards.clear()

        return self._get_observation(), {}

    def _get_observation(self):
        return np.array([self.current_storage, self.data[self.current_step, 1],
                         self.data[self.current_step, 2], self.data[self.current_step, 4]])

    def step(self, action):
        normalized_q1_release = (action[0] + 1) / 2
        normalized_q2_release = (action[1] + 1) / 2
        q1_release_af = self.min_release_af + normalized_q1_release * (self.max_release_af - self.min_release_af)
        q2_release_af = self.min_release_af + normalized_q2_release * (self.max_release_af - self.min_release_af)
        total_release_af = q1_release_af + q2_release_af

        inflow_af = self.data[self.current_step, 2]
        evaporation_af = self.data[self.current_step, 1]
        self.current_storage = (self.current_storage * stds[0] + means[0]) + \
                               ((inflow_af * stds[2] + means[2]) - (evaporation_af * stds[1] + means[1]) - (total_release_af * stds[3] + means[3]))
        self.current_storage = (self.current_storage - means[0]) / stds[0]

        # Predict elevation from storage
        predicted_elevation_rescaled = storage_to_elevation(self.current_storage)

        # Calculate hydropower
        date_str = date[self.current_step]
        dt = pd.to_datetime(date_str, format="%d-%b-%y")
        doy = dt.timetuple().tm_yday
        release_hydropower_cfs = (q2_release_af * stds[3] + means[3])/ 1.98211
        # hydropower_generation = navajo_power_generation_model(release_hydropower_cfs, predicted_elevation_rescaled)
        hydropower_generation = navajo_power_generation_model([pd.to_datetime(date_str, format="%d-%b-%y")],[release_hydropower_cfs], [predicted_elevation_rescaled])[0]

        # Compute Storage Reward
        if self.min_storage_af <= self.current_storage <= self.max_storage_af:
          target_storage = (self.max_storage_af + self.min_storage_af) / 2
          r_storage = 1 - abs(self.current_storage - target_storage) / (self.max_storage_af - self.min_storage_af)
        else:
          r_storage = -1

        # Penalize for fisheries releasing below 350 cfs
        fisheries_threshold_af = 350 * 1.98211
        fisheries_threshold_normalized = (fisheries_threshold_af - means[3]) / stds[3]
        # Apply reward/penalty only to fisheries release
        if q2_release_af >= fisheries_threshold_normalized:
            r_fisheries = 0.5
        else:
            r_fisheries = -0.2

        # Compute Hydropower Reward
        min_mwh = 288
        max_mwh = 768

        if hydropower_generation < min_mwh:
            r_hydro = -0.5  # not enough generation

        # elif hydropower_generation > max_mwh:
        #     r_hydro = -0.5  # unrealistic, turbine capacity exceeded
        else:
            r_hydro = (hydropower_generation-min_mwh) / (max_mwh-min_mwh)  # normalized bonus

        # Compute NIIP Reward
        r_niip = 0
        if 50 <= doy <= 300:
            daily_demand = niip_daily_demand(doy) * 1.98211

            # Compute total area only over actual demand period (DOY 50 to 300)
            total_niip = np.sum(niip_daily_demand(np.arange(50, 301))) * 1.98211

            # Daily NIIP release
            daily_release = q1_release_af * stds[3] + means[3]

            # # Calculte delta
            # delta = daily_demand - daily_release

            # r_niip = 1 - abs(delta) / total_niip

            if daily_release >= daily_demand:
                r_niip = 1.0
            else:
                r_niip = daily_release/ daily_demand

        reward = r_storage + r_fisheries + r_hydro + r_niip

        # Penalize for total release exceeding 5000 cfs
        if total_release_af >= self.max_release_af:
            reward -= 0.8
        else:
            reward += 0.5

        self.storage_rewards.append(r_storage)
        self.fisheries_rewards.append(r_fisheries)
        self.hydropower_rewards.append(r_hydro)
        self.niip_rewards.append(r_niip)
        self.reward_history.append(reward)
        self.storage_history.append(self.current_storage)
        self.release_history.append(total_release_af)
        self.episode_step_count += 1
        self.current_step += 1

        done = self.current_step >= len(self.data) or self.episode_step_count >= self.episode_length
        if done:
            self.episode_reward.append(sum(self.reward_history))
            self.mean_storage_history.append(np.mean(self.storage_history))
            self.mean_release_history.append(np.mean(self.release_history))
            self.episode_storage_rewards.append(sum(self.storage_rewards))
            self.episode_fisheries_rewards.append(sum(self.fisheries_rewards))
            self.episode_hydropower_rewards.append(sum(self.hydropower_rewards))
            self.episode_niip_rewards.append(sum([r for r in self.niip_rewards if r != 0]) if any(self.niip_rewards) else 0)

        return (self._get_observation() if not done else np.zeros(4)), reward, done, False, {}

# Set up environment
episode_length = 3600
env = ReservoirEnv(train_data, min_storage_af, max_storage_af, min_release_af, max_release_af, episode_length)

# Initialize model
model = PPO("MlpPolicy", env,  verbose=1)

# Train the model
n_episodes = 400
total_timesteps = n_episodes * episode_length
model.learn(total_timesteps=total_timesteps)

# Save model
model.save("ppo_reservoir_agent")

# Save and Plot
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
plt.figure(figsize=(12, 6))
plt.plot(env.episode_reward, label="Total Reward", color='black')
plt.plot(env.episode_storage_rewards, label="Storage", linestyle="--")
plt.plot(env.episode_fisheries_rewards, label="Fisheries", linestyle="--")
plt.plot(env.episode_hydropower_rewards, label="Hydropower", linestyle="--")
plt.plot(env.episode_niip_rewards, label="NIIP", linestyle="--")
plt.xlabel("Episode")
plt.ylabel("Mean Reward")
plt.title("Reward Components per Episode")
plt.legend()
plt.grid()
plt.savefig("component_rewards_per_episode.png")
plt.show()

# Plot all reward components per episode with maxima marked
plt.figure(figsize=(12, 6))

# Total Reward
plt.plot(env.episode_reward, label="Total Reward", color='black')
max_total = max(env.episode_reward)
ep_total = np.argmax(env.episode_reward)
plt.scatter(ep_total, max_total, color='black', marker='o')
plt.text(ep_total, max_total, f"Max={max_total:.1f}", fontsize=8, color='black')

# Storage Reward
plt.plot(env.episode_storage_rewards, label="Storage", linestyle="--")
max_storage = max(env.episode_storage_rewards)
ep_storage = np.argmax(env.episode_storage_rewards)
plt.scatter(ep_storage, max_storage, color='C0', marker='o')
plt.text(ep_storage, max_storage, f"Max={max_storage:.1f}", fontsize=8, color='C0')

# Fisheries Reward
plt.plot(env.episode_fisheries_rewards, label="Fisheries", linestyle="--")
max_fish = max(env.episode_fisheries_rewards)
ep_fish = np.argmax(env.episode_fisheries_rewards)
plt.scatter(ep_fish, max_fish, color='C1', marker='o')
plt.text(ep_fish, max_fish, f"Max={max_fish:.1f}", fontsize=8, color='C1')

# Hydropower Reward
plt.plot(env.episode_hydropower_rewards, label="Hydropower", linestyle="--")
max_hydro = max(env.episode_hydropower_rewards)
ep_hydro = np.argmax(env.episode_hydropower_rewards)
plt.scatter(ep_hydro, max_hydro, color='C2', marker='o')
plt.text(ep_hydro, max_hydro, f"Max={max_hydro:.1f}", fontsize=8, color='C2')

# NIIP Reward
plt.plot(env.episode_niip_rewards, label="NIIP", linestyle="--")
max_niip = max(env.episode_niip_rewards)
ep_niip = np.argmax(env.episode_niip_rewards)
plt.scatter(ep_niip, max_niip, color='C3', marker='o')
plt.text(ep_niip, max_niip, f"Max={max_niip:.1f}", fontsize=8, color='C3')

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward Components per Episode")

# Legend with max values
plt.legend(title=f"Max Values:\n"
                 f"Total={max_total:.1f}, "
                 f"Storage={max_storage:.1f}, "
                 f"Fisheries={max_fish:.1f}, "
                 f"Hydro={max_hydro:.1f}, "
                 f"NIIP={max_niip:.1f}")

plt.grid()
plt.savefig("component_rewards_per_episode_with_maximum.png")
plt.show()

print("Training complete. Model, Storage, Release, and Reward plots saved.")

#------Testing--------------

# Load trained model
from stable_baselines3 import PPO
from hydropower_model import navajo_power_generation_model
from niip_demand import niip_daily_demand
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model = PPO.load("ppo_reservoir_agent")

# Leap year check
def is_leap(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

# Test Data
test_date = date[18000:].reset_index(drop=True)

# Storage for results
predicted_storage, actual_storage = [], []
predicted_release, actual_release = [], []
q1_releases = []
q2_releases = []
hydropower_generation_test = []
hydropower_generation_1300 = []

niip_daily_release_list = []
niip_daily_demand_list = []
niip_deltas = []
doy_list = []  # for DOY plotting
year_list = []

# Initialize storage
current_storage = test_data[0, 0]

for i in range(len(test_data)):
    dt = pd.to_datetime(test_date[i], format="%d-%b-%y")
    doy = dt.timetuple().tm_yday
    doy_list.append(doy)  # store DOY for plotting
    year = dt.year
    leap = is_leap(year)
    year_list.append(year)

    # Observation
    obs = np.array([current_storage, test_data[i, 1], test_data[i, 2], test_data[i, 4]])
    action, _ = model.predict(obs)

    normalized_q1_release = (action[0] + 1) / 2
    normalized_q2_release = (action[1] + 1) / 2

    q1_release_af = min_release_af + normalized_q1_release * (max_release_af - min_release_af)
    q2_release_af = min_release_af + normalized_q2_release * (max_release_af - min_release_af)
    total_release_af = q1_release_af + q2_release_af

    inflow_af = test_data[i, 2]
    evaporation_af = test_data[i, 1]

    current_storage = (current_storage * stds[0] + means[0]) + \
                      ((inflow_af * stds[2] + means[2]) - (evaporation_af * stds[1] + means[1]) - (total_release_af * stds[3] + means[3]))
    current_storage = (current_storage - means[0]) / stds[0]

    predicted_elevation_rescaled = storage_to_elevation(current_storage)
    release_hydropower_cfs = (q2_release_af * stds[3] + means[3]) / 1.98211
    # hydropower_generation = navajo_power_generation_model(release_hydropower_cfs, predicted_elevation_rescaled)
    hydropower_generation = navajo_power_generation_model([dt], [release_hydropower_cfs], [predicted_elevation_rescaled])[0]
    hydropower_generation_test.append(hydropower_generation)

    # NEW: Hydropower at fixed 1300 cfs, using the predicted elevation
    # hydropower_fixed_1300 = navajo_power_generation_model(1300.0, predicted_elevation_rescaled)
    hydropower_fixed_1300 = navajo_power_generation_model([dt], [1300], [predicted_elevation_rescaled])[0]
    hydropower_generation_1300.append(hydropower_fixed_1300)

    # NIIP daily tracking (DOY 50–300)
    reg_release_af = q1_release_af * stds[3] + means[3]
    if 50 <= doy <= 300:
        daily_demand = niip_daily_demand(doy) * 1.98211
        total_niip = np.sum(niip_daily_demand(np.arange(50, 301))) * 1.98211
        niip_release_delta = daily_demand - reg_release_af
    else:
        daily_demand = 0
        niip_release_delta = 0

    niip_daily_release_list.append(reg_release_af)
    niip_daily_demand_list.append(daily_demand)
    niip_deltas.append(niip_release_delta)

    predicted_storage.append(current_storage)
    q1_releases.append(q1_release_af)
    q2_releases.append(q2_release_af)
    predicted_release.append(total_release_af)
    actual_storage.append(test_data[i, 0])
    actual_release.append(test_data[i, 3])

# ---------------- Rewards During Testing ----------------
storage_rewards_test = []
fisheries_rewards_test = []
hydropower_rewards_test = []
niip_rewards_test = []
total_rewards_test = []

for i in range(len(test_data)):
    dt = pd.to_datetime(test_date[i], format="%d-%b-%y")
    doy = dt.timetuple().tm_yday

    q1_release_af = q1_releases[i]
    q2_release_af = q2_releases[i]
    total_release_af = predicted_release[i]
    hydropower_generation = hydropower_generation_test[i]

    # Storage reward
    if min_storage_af <= predicted_storage[i] <= max_storage_af:
        target_storage = (max_storage_af + min_storage_af) / 2
        r_storage = 1 - abs(predicted_storage[i] - target_storage) / (max_storage_af - min_storage_af)
    else:
        r_storage = -1

    # Fisheries reward
    fisheries_threshold_af = 350 * 1.98211
    fisheries_threshold_normalized = (fisheries_threshold_af - means[3]) / stds[3]
    if q2_release_af >= fisheries_threshold_normalized:
        r_fisheries = 0.5
    else:
        r_fisheries = -0.2

    # Hydropower reward
    min_mwh, max_mwh = 288, 768
    if hydropower_generation < min_mwh:
        r_hydro = -0.5
    else:
        r_hydro = (hydropower_generation - min_mwh) / (max_mwh - min_mwh)

    # NIIP reward
    r_niip = 0
    if 50 <= doy <= 300:
        daily_demand = niip_daily_demand(doy) * 1.98211
        total_niip = np.sum(niip_daily_demand(np.arange(50, 301))) * 1.98211
        daily_release = q1_release_af * stds[3] + means[3]
        # delta = daily_demand - daily_release
        # r_niip = 1 - abs(delta) / total_niip
        if daily_release >= daily_demand:
            r_niip = 1
        else:
            r_niip = daily_release / daily_demand

    reward = r_storage + r_fisheries + r_hydro + r_niip

    total_rewards_test.append(reward)
    storage_rewards_test.append(r_storage)
    fisheries_rewards_test.append(r_fisheries)
    hydropower_rewards_test.append(r_hydro)
    niip_rewards_test.append(r_niip)

pd.DataFrame({
    "Total Reward": total_rewards_test,
    "Storage Reward": storage_rewards_test,
    "Fisheries Reward": fisheries_rewards_test,
    "Hydropower Reward": hydropower_rewards_test,
    "NIIP Reward": niip_rewards_test,
    "DOY": doy_list
}).to_csv("test_reward_components.csv", index=False)

# Save results to CSV files
pd.DataFrame({
    "Predicted Storage (af)": predicted_storage,
    "Actual Storage (af)": actual_storage
}).to_csv("test_predicted_vs_actual_storage_complete_data.csv", index=False)

pd.DataFrame({
    "Predicted Release (af)": predicted_release,
    "Actual Release (af)": actual_release
}).to_csv("test_predicted_vs_actual_release_complete_data.csv", index=False)

pd.DataFrame({
    'Q1 Release (af)': q1_releases,
    'Q2 Release (af)': q2_releases
}).to_csv("test_q1_vs_q2_release_complete_data.csv", index=False)

pd.DataFrame(dataset).to_csv("test_complete_dataset.csv", index=False)

pd.DataFrame({
    "NIIP_Daily_Release (af)": niip_daily_release_list,
    "NIIP_Daily_Demand (af)": niip_daily_demand_list,
    "NIIP_Delta (af)": niip_deltas,
    "Year": year_list,
    "DOY": doy_list
}).to_csv("test_niip_release_tracking.csv", index=False)

# ---------------- Fulfillment Metrics ----------------
df_niip = pd.DataFrame({
    "Year": year_list,
    "DOY": doy_list,
    "Demand": niip_daily_demand_list,
    "Release": niip_daily_release_list
})

metrics = []
for year, group in df_niip.groupby("Year"):
    demand_days = group[group["Demand"] > 0]
    if len(demand_days) == 0:
        continue
    fulfilled_days = (demand_days["Release"] >= demand_days["Demand"]).sum()
    percentage_fulfilled = (fulfilled_days / len(demand_days)) * 100

    # Volume-based metric
    total_release_vol = demand_days["Release"].sum()
    total_demand_vol = demand_days["Demand"].sum()
    vol_percentage = (total_release_vol / total_demand_vol) * 100 if total_demand_vol > 0 else 0

    metrics.append({
        "Year": year,
        "Days_with_Demand": len(demand_days),
        "Fulfilled_Days": fulfilled_days,
        "Fulfillment_%": percentage_fulfilled,
        "Total_Demand_Volume": total_demand_vol,
        "Total_Release_Volume": total_release_vol,
        "Volume_Fulfillment_%": vol_percentage
    })

df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv("niip_fulfillment_metrics.csv", index=False)
fully_met_years = (df_metrics["Fulfillment_%"] == 100).sum()
print(f"Years fully meeting NIIP demand: {fully_met_years}/{len(df_metrics)}")
print(df_metrics)

# ---------------- Plots ----------------

# Storage
df = pd.read_csv('test_predicted_vs_actual_storage_complete_data.csv')
column_1 = df['Actual Storage (af)'] * stds[0] + means[0]
column_2 = df['Predicted Storage (af)'] * stds[0] + means[0]

plt.figure(figsize=(10, 6))
plt.plot(df.index, column_1, label='Actual Storage(af)', color='blue')
plt.plot(df.index, column_2, label='Predicted Storage (af)', color='orange')
plt.axhline(y=500000, color="r", linestyle="--", label="Min Storage Limit 500000 AF")
plt.axhline(y=1731750, color="g", linestyle="--", label="Max Storage Limit 1731750 AF")
plt.xlabel("Timestep")
plt.ylabel("Storage (af)")
plt.title("Predicted vs Actual Storage")
plt.legend()
plt.grid()
plt.savefig("test_predicted_vs_actual_storage_rescaled.png")
plt.show()

# Release
df_1 = pd.read_csv('test_predicted_vs_actual_release_complete_data.csv')
column_3 = df_1['Actual Release (af)'] * stds[3] + means[3]
column_4 = df_1['Predicted Release (af)'] * stds[3] + means[3]

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

# Q1 vs Q2
df_2 = pd.read_csv('test_q1_vs_q2_release_complete_data.csv')
column_5 = df_2['Q1 Release (af)'] * stds[3] + means[3]
column_6 = df_2['Q2 Release (af)'] * stds[3] + means[3]

plt.figure(figsize=(10, 6))
plt.plot(df_2.index, column_5, label='Q1 Release (af)', color='blue')
plt.plot(df_2.index, column_6, label='Q2 Release (af)', color='orange')
plt.xlabel("Timestep")
plt.ylabel("Release (af)")
plt.title("Q1 vs Q2 Release")
plt.legend()
plt.grid()
plt.savefig("test_q1_vs_q2_release_rescaled.png")
plt.show()

# Hydropower
pd.DataFrame(hydropower_generation_test, columns=["Hydropower (MWh)"]).to_csv("test_hydropower_generation.csv", index=False)
plt.figure(figsize=(10, 6))
plt.plot(hydropower_generation_test, label="Hydropower Generation (MWh)", color="green")
plt.xlabel("Timestep")
plt.ylabel("Hydropower (MWh)")
plt.title("Hydropower Generation During Testing")
plt.legend()
plt.grid()
plt.savefig("test_hydropower_generation.png")
plt.show()

# Hydropower — vs timestep (ADD: 768 MWh line + 1300 cfs curve)
pd.DataFrame(hydropower_generation_test, columns=["Hydropower (MWh)"]).to_csv("test_hydropower_generation.csv", index=False)
plt.figure(figsize=(10, 6))
plt.plot(hydropower_generation_test, label="Hydropower Generation (MWh)", color="green")
plt.plot(hydropower_generation_1300, label="Hydropower @ 1300 cfs (MWh)")
plt.axhline(y=768, color="red", linestyle="--", label="Max possible 768 MWh")
plt.xlabel("Timestep")
plt.ylabel("Hydropower (MWh)")
plt.title("Hydropower Generation During Testing")
plt.legend()
plt.grid()
plt.savefig("test_hydropower_generation_with_max_hydropower.png")
plt.show()

# Reward components per timestep
plt.figure(figsize=(12, 6))
plt.plot(total_rewards_test, label="Total Reward", color="black")
plt.plot(storage_rewards_test, label="Storage", linestyle="--")
plt.plot(fisheries_rewards_test, label="Fisheries", linestyle="--")
plt.plot(hydropower_rewards_test, label="Hydropower", linestyle="--")
plt.plot(niip_rewards_test, label="NIIP", linestyle="--")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.title("Reward Components During Testing")
plt.legend()
plt.grid()
plt.savefig("test_reward_components_timestep.png")
plt.show()

# ---------------- Clean DOY percentile plots ----------------
def plot_doy_percentiles(doy_vals, val1, val2, labels, ylabel, filename):
    df_temp = pd.DataFrame({"DOY": doy_vals * 2,
                            "Value": list(val1) + list(val2),
                            "Type": [labels[0]] * len(val1) + [labels[1]] * len(val2)})
    plt.figure(figsize=(10, 6))
    for t, grp in df_temp.groupby("Type"):
        grouped = grp.groupby("DOY")["Value"]
        median = grouped.median()
        q25 = grouped.quantile(0.25)
        q75 = grouped.quantile(0.75)
        plt.plot(median.index, median.values, label=f"{t} Median")
        plt.fill_between(median.index, q25, q75, alpha=0.3, label=f"{t} IQR")
    plt.xlabel("Day of Year"); plt.ylabel(ylabel)
    plt.title(f"{labels[0]} vs {labels[1]} - Seasonal Distribution")
    plt.legend(); plt.grid()
    plt.savefig(filename); plt.show()

plot_doy_percentiles(doy_list, column_1, column_2, ["Actual Storage", "Predicted Storage"], "Storage (af)", "percentiles_storage_doy.png")
plot_doy_percentiles(doy_list, column_3, column_4, ["Actual Release", "Predicted Release"], "Release (af)", "percentiles_release_doy.png")
plot_doy_percentiles(doy_list, column_5, column_6, ["Q1 Release", "Q2 Release"], "Release (af)", "percentiles_q1_q2_doy.png")

# ---------------- Clean DOY percentile plots ----------------
def plot_doy_percentiles(doy_vals, val1, val2, labels, ylabel, filename):
    df_temp = pd.DataFrame({
        "DOY": doy_vals * 2,
        "Value": list(val1) + list(val2),
        "Type": [labels[0]] * len(val1) + [labels[1]] * len(val2)
    })
    plt.figure(figsize=(10, 6))
    for t, grp in df_temp.groupby("Type"):
        grouped = grp.groupby("DOY")["Value"]
        median = grouped.median()
        q25 = grouped.quantile(0.25)
        q75 = grouped.quantile(0.75)
        plt.plot(median.index, median.values, label=f"{t} Median")
        plt.fill_between(median.index, q25, q75, alpha=0.3, label=f"{t} IQR")
    plt.xlabel("Day of Year"); plt.ylabel(ylabel)
    plt.title(f"{labels[0]} vs {labels[1]} - Seasonal Distribution")
    plt.legend(); plt.grid()
    plt.savefig(filename); plt.show()

# Storage DOY
plot_doy_percentiles(
    doy_list, column_1, column_2,
    ["Actual Storage", "Predicted Storage"],
    "Storage (af)", "percentiles_storage_doy.png"
)

# Release DOY
plot_doy_percentiles(
    doy_list, column_3, column_4,
    ["Actual Release", "Predicted Release"],
    "Release (af)", "percentiles_release_doy.png"
)

# Q1 vs Q2 + NIIP Demand DOY
def plot_q1_q2_with_demand(doy_vals, q1_vals, q2_vals, demand_vals, filename):
    df_temp = pd.DataFrame({
        "DOY": list(doy_vals) * 3,
        "Value": list(q1_vals) + list(q2_vals) + list(demand_vals),
        "Type": (["Q1 Release"] * len(q1_vals) +
                 ["Q2 Release"] * len(q2_vals) +
                 ["NIIP Demand"] * len(demand_vals))
    })
    plt.figure(figsize=(10, 6))
    for t, grp in df_temp.groupby("Type"):
        grouped = grp.groupby("DOY")["Value"]
        median = grouped.median()
        q25 = grouped.quantile(0.25)
        q75 = grouped.quantile(0.75)
        if t == "NIIP Demand":
            plt.plot(median.index, median.values, label=f"{t} Median", color="red", linestyle="--")
        else:
            plt.plot(median.index, median.values, label=f"{t} Median")
            plt.fill_between(median.index, q25, q75, alpha=0.3, label=f"{t} IQR")
    plt.xlabel("Day of Year")
    plt.ylabel("Release (af)")
    plt.title("Q1 vs Q2 vs NIIP Demand - Seasonal Distribution")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.show()

plot_q1_q2_with_demand(
    doy_list, column_5, column_6, niip_daily_demand_list,
    "percentiles_q1_q2_niip_doy.png"
)

# Hydropower combined percentile
def plot_single_percentile_by_doy(values, label, ylabel, filename):
    df_doy = pd.DataFrame({"DOY": doy_list, "Value": values})
    grouped = df_doy.groupby("DOY")["Value"]
    median = grouped.median()
    q25 = grouped.quantile(0.25)
    q75 = grouped.quantile(0.75)
    plt.figure(figsize=(10, 6))
    plt.plot(median.index, median.values, label=f"{label} Median", color="green")
    plt.fill_between(median.index, q25.values, q75.values, alpha=0.3, color="green", label="IQR")
    plt.xlabel("Day of Year")
    plt.ylabel(ylabel)
    plt.title(f"{label} - Seasonal Distribution")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.show()

plot_single_percentile_by_doy(hydropower_generation_test,"Hydropower Generation", "MWh", "percentiles_hydropower_doy.png")

# Hydropower combined percentile (ADD: 768 line + 1300 cfs curve)
def plot_hydropower_percentiles_with_1300(doy_vals, hydro_main, hydro_1300, filename):
    df_temp = pd.DataFrame({
        "DOY": list(doy_vals) * 2,
        "Value": list(hydro_main) + list(hydro_1300),
        "Type": (["Hydropower (MWh)"] * len(hydro_main) +
                 ["Hydropower @ 1300 cfs (MWh)"] * len(hydro_1300))
    })
    plt.figure(figsize=(10, 6))
    for t, grp in df_temp.groupby("Type"):
        grouped = grp.groupby("DOY")["Value"]
        median = grouped.median()
        q25 = grouped.quantile(0.25)
        q75 = grouped.quantile(0.75)
        plt.plot(median.index, median.values, label=f"{t} Median")
        plt.fill_between(median.index, q25, q75, alpha=0.3, label=f"{t} IQR")
    # Max possible line
    plt.axhline(y=768, color="red", linestyle="--", label="Max possible 768 MWh")
    plt.xlabel("Day of Year")
    plt.ylabel("MWh")
    plt.title("Hydropower (Daily) - Seasonal Distribution")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.show()

plot_hydropower_percentiles_with_1300(
    doy_list, hydropower_generation_test, hydropower_generation_1300, "percentiles_hydropower_doy_with_max_hydropower.png"
)

# Combined Seasonal Reward Percentiles
def plot_all_rewards_percentiles(doy_vals, rewards_dict, filename):
    plt.figure(figsize=(12, 7))

    for label, (rewards, color) in rewards_dict.items():
        df_temp = pd.DataFrame({"DOY": doy_vals, "Reward": rewards})
        grouped = df_temp.groupby("DOY")["Reward"]
        median = grouped.median()
        q25 = grouped.quantile(0.25)
        q75 = grouped.quantile(0.75)

        plt.plot(median.index, median.values, label=f"{label} Median", color=color)
        plt.fill_between(median.index, q25.values, q75.values, alpha=0.2, color=color, label=f"{label} IQR")

    plt.xlabel("Day of Year")
    plt.ylabel("Reward")
    plt.title("Reward Components - Seasonal Distribution (Median + IQR)")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.show()

# Call with all rewards
rewards_dict = {
    "Total": (total_rewards_test, "black"),
    "Storage": (storage_rewards_test, "blue"),
    "Fisheries": (fisheries_rewards_test, "orange"),
    "Hydropower": (hydropower_rewards_test, "green"),
    "NIIP": (niip_rewards_test, "red")
}

plot_all_rewards_percentiles(doy_list, rewards_dict, "percentiles_all_rewards_doy.png")

print("Testing complete. All results and plots saved.")