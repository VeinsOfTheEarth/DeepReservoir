# -*- coding: utf-8 -*-
"""
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
niip_model_path = "/content/drive/MyDrive/niip_demand.py"
niip_splinecurve_path = "/content/drive/MyDrive/niip_demand_spline.pkl"

# Copy required files
import shutil
shutil.copy(hydropower_model_path, "./hydropower_model.py")
shutil.copy(parameters_path, "./parameters.pkl")
shutil.copy(niip_model_path, "./niip_demand.py")
shutil.copy(niip_splinecurve_path, "./niip_demand_spline.pkl")

# Imports
import os
import pickle
import numpy as np
import pandas as pd
import torch as th
import tensorflow as tf
import gymnasium
from gymnasium import Env
from gymnasium.spaces import Box
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from hydropower_model import navajo_power_generation_model
from niip_demand import niip_daily_demand

# Load main data
data = pd.read_csv(main_data_path)
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
        self.cumulative_niip_release = 0
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
        hydropower_generation = navajo_power_generation_model([pd.to_datetime(date_str, format="%d-%b-%y")],[release_hydropower_cfs], [predicted_elevation_rescaled])[0]


        # Compute Storage Reward
        if self.min_storage_af <= self.current_storage <= self.max_storage_af:
          target_storage = (self.max_storage_af + self.min_storage_af) / 2
          r_storage = 1 - abs(self.current_storage - target_storage) / (self.max_storage_af - self.min_storage_af)
        else:
          r_storage = -1
        self.storage_rewards.append(r_storage)

        # Penalize for fisheries releasing below 350 cfs
        fisheries_threshold_af = 350 * 1.98211
        fisheries_threshold_normalized = (fisheries_threshold_af - means[3]) / stds[3]
        # Apply reward/penalty only to fisheries release
        if q2_release_af >= fisheries_threshold_normalized:
            r_fisheries = 0.5
        else:
            r_fisheries = -0.2

        self.fisheries_rewards.append(r_fisheries)

        # Compute Hydropower Reward
        min_mwh = 288
        max_mwh = 768

        if hydropower_generation < min_mwh:
            r_hydro = -0.5  # not enough generation

        # elif hydropower_generation > max_mwh:
        #     r_hydro = -0.5  # unrealistic, turbine capacity exceeded
        else:
            r_hydro = (hydropower_generation - min_mwh) / (max_mwh - min_mwh)  # normalized bonus

        self.hydropower_rewards.append(r_hydro)

        r_niip = 0
        if 50 <= doy <= 300:
            daily_demand = niip_daily_demand(doy) * 1.98211

            # Compute total area only over actual demand period (DOY 50 to 300)
            total_niip = np.sum(niip_daily_demand(np.arange(50, 301))) * 1.98211

            # Update cumulative NIIP release
            self.cumulative_niip_release += q1_release_af * stds[3] + means[3]

            # Cumulative demand from start of year to today
            cumulative_demand_upto_today = np.sum(niip_daily_demand(np.arange(50, doy+1))) * 1.98211 if doy >= 50 else 0.0
            delta = cumulative_demand_upto_today - self.cumulative_niip_release

            r_niip = 1 - abs(delta) / total_niip

        else:
            self.cumulative_niip_release = 0

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

print("Training complete. Model, Storage, Release, and Reward plots saved.")

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
niip_cumulative_release = []
niip_deltas = []
doy_list = []
year_list = []

# Initialize storage and NIIP cumulative tracker
current_storage = test_data[0, 0]
current_year = pd.to_datetime(test_date[0], format="%d-%b-%y").year
cumulative_niip_release = 0

for i in range(len(test_data)):
    dt = pd.to_datetime(test_date[i], format="%d-%b-%y")
    doy = dt.timetuple().tm_yday
    doy_list.append(doy)
    year = dt.year
    leap = is_leap(year)
    year_list.append(year)

    if i > 0 and year != current_year:
        cumulative_niip_release = 0
        current_year = year

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
                      ((inflow_af * stds[2] + means[2]) -
                       (evaporation_af * stds[1] + means[1]) -
                       (total_release_af * stds[3] + means[3]))
    current_storage = (current_storage - means[0]) / stds[0]

    predicted_elevation_rescaled = storage_to_elevation(current_storage)
    release_hydropower_cfs = (q2_release_af * stds[3] + means[3]) / 1.98211
    hydropower_generation = navajo_power_generation_model([dt],
                                                          [release_hydropower_cfs],
                                                          [predicted_elevation_rescaled])[0]
    hydropower_generation_test.append(hydropower_generation)

    # NIIP cumulative tracking (DOY 50–250)
    reg_release_af = q1_release_af * stds[3] + means[3]
    if 50 <= doy <= 250:
        cumulative_niip_release += reg_release_af
        demand_to_doy = np.sum(niip_daily_demand(np.arange(50, doy + 1))) * 1.98211
        niip_release_delta = demand_to_doy - cumulative_niip_release
    else:
        niip_release_delta = 0

    niip_cumulative_release.append(cumulative_niip_release)
    niip_deltas.append(niip_release_delta)

    predicted_storage.append(current_storage)
    q1_releases.append(q1_release_af)
    q2_releases.append(q2_release_af)
    predicted_release.append(total_release_af)
    actual_storage.append(test_data[i, 0])
    actual_release.append(test_data[i, 3])

# Save results
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
    "Year": year_list,
    "DOY": doy_list,
    "NIIP_Cumulative_Release (af)": niip_cumulative_release,
    "NIIP_Delta (af)": niip_deltas
}).to_csv("test_niip_release_tracking.csv", index=False)

# ---------------- Fulfillment Metrics ----------------
df_niip = pd.DataFrame({
    "Year": year_list,
    "DOY": doy_list,
    "Demand_Cumulative": [np.sum(niip_daily_demand(np.arange(50, doy + 1))) * 1.98211 if 50 <= doy <= 250 else 0 for doy in doy_list],
    "Release_Cumulative": niip_cumulative_release
})

metrics = []
for year, group in df_niip.groupby("Year"):
    demand_days = group[group["Demand_Cumulative"] > 0]
    if len(demand_days) == 0:
        continue
    fulfilled_days = (demand_days["Release_Cumulative"] >= demand_days["Demand_Cumulative"]).sum()
    percentage_fulfilled = (fulfilled_days / len(demand_days)) * 100
    total_release_vol = demand_days["Release_Cumulative"].iloc[-1]
    total_demand_vol = demand_days["Demand_Cumulative"].iloc[-1]
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

# ---------------- Rescaled Plots ----------------
df = pd.read_csv('test_predicted_vs_actual_storage_complete_data.csv')
column_1 = df['Actual Storage (af)'] * stds[0] + means[0]
column_2 = df['Predicted Storage (af)'] * stds[0] + means[0]

plt.figure(figsize=(10, 6))
plt.plot(df.index, column_1, label='Actual Storage(af)', color='blue')
plt.plot(df.index, column_2, label='Predicted Storage (af)', color='orange')
plt.axhline(y=500000, color="r", linestyle="--")
plt.axhline(y=1731750, color="g", linestyle="--")
plt.xlabel("Timestep"); plt.ylabel("Storage (af)")
plt.title("Predicted vs Actual Storage")
plt.legend(); plt.grid()
plt.savefig("test_predicted_vs_actual_storage_rescaled.png")
plt.show()

df_1 = pd.read_csv('test_predicted_vs_actual_release_complete_data.csv')
column_3 = df_1['Actual Release (af)'] * stds[3] + means[3]
column_4 = df_1['Predicted Release (af)'] * stds[3] + means[3]

plt.figure(figsize=(10, 6))
plt.plot(df_1.index, column_3, label='Actual Release (af)', color='blue')
plt.plot(df_1.index, column_4, label='Predicted Release (af)', color='orange')
plt.xlabel("Timestep"); plt.ylabel("Release (af)")
plt.title("Predicted vs Actual Release")
plt.legend(); plt.grid()
plt.savefig("test_predicted_vs_actual_release_rescaled.png")
plt.show()

df_2 = pd.read_csv('test_q1_vs_q2_release_complete_data.csv')
column_5 = df_2['Q1 Release (af)'] * stds[3] + means[3]
column_6 = df_2['Q2 Release (af)'] * stds[3] + means[3]

plt.figure(figsize=(10, 6))
plt.plot(df_2.index, column_5, label='Q1 Release (af)', color='blue')
plt.plot(df_2.index, column_6, label='Q2 Release (af)', color='orange')
plt.xlabel("Timestep"); plt.ylabel("Release (af)")
plt.title("Q1 vs Q2 Release")
plt.legend(); plt.grid()
plt.savefig("test_q1_vs_q2_release_rescaled.png")
plt.show()

pd.DataFrame(hydropower_generation_test, columns=["Hydropower (MWh)"]).to_csv("test_hydropower_generation.csv", index=False)
plt.figure(figsize=(10, 6))
plt.plot(hydropower_generation_test, label="Hydropower Generation (MWh)", color="green")
plt.xlabel("Timestep"); plt.ylabel("Hydropower (MWh)")
plt.title("Hydropower Generation During Testing")
plt.legend(); plt.grid()
plt.savefig("test_hydropower_generation.png")
plt.show()

# ---------------- Percentile DOY Plots ----------------
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

# Hydropower percentile
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

plot_single_percentile_by_doy(hydropower_generation_test, "Hydropower Generation", "MWh", "percentiles_hydropower_doy.png")

print("Testing complete. All results and plots saved.")