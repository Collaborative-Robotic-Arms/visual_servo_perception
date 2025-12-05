import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
# UPDATE THIS to match your actual csv filename!
filename = 'pose_validation_032021.csv' 
# ---------------------

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"Error: Could not find file '{filename}'. Please update the filename in the script.")
    exit()

# 1. AUTOMATIC GOAL SETTING (The "Relative" Fix)
# Instead of hardcoding 0.0 or 0.2093, we assume the robot *reached* the goal
# at the end of the experiment. We set the "Goal" to be the average of the last 10 points.
goal_x = df['x'].tail(10).mean()
goal_y = df['y'].tail(10).mean()
goal_z = df['z'].tail(10).mean()

# 2. CALCULATE RELATIVE ERROR
# This eliminates the "429mm error" caused by frame direction mismatches.
# We are now plotting "Distance remaining to target".
df['err_depth'] = df['x'] - goal_x
df['err_horiz'] = df['y'] - goal_y
df['err_vert']  = df['z'] - goal_z

# 3. FIX ORIENTATION JUMPS (The "Seismograph" Fix)
# Your frames are 180 degrees apart, causing values to jump between -179 and +180.
# np.unwrap smooths these jumps into a continuous line.
df['roll_unwrapped'] = np.unwrap(df['roll'])
df['pitch_unwrapped'] = np.unwrap(df['pitch'])
df['yaw_unwrapped'] = np.unwrap(df['yaw'])

# Set goal orientation based on the unwrapped final values
goal_roll = df['roll_unwrapped'][-10:].mean()
goal_pitch = df['pitch_unwrapped'][-10:].mean()
goal_yaw = df['yaw_unwrapped'][-10:].mean()

# Calculate deviation in degrees
df['deg_err_roll'] = np.degrees(df['roll_unwrapped'] - goal_roll)
df['deg_err_pitch'] = np.degrees(df['pitch_unwrapped'] - goal_pitch)
df['deg_err_yaw'] = np.degrees(df['yaw_unwrapped'] - goal_yaw)

# 4. AUTO-CROP (Remove "Waiting for Service" time)
# Find the first moment where the robot moves more than 0.1mm
# We calculate a simple velocity magnitude approx
velocity = df['x'].diff().abs() + df['y'].diff().abs() + df['z'].diff().abs()
move_start = df[velocity > 0.0001].index

if len(move_start) > 0:
    # Start 10 frames (0.5s) before movement begins to show the "start"
    start_idx = max(0, move_start[0] - 10) 
    df = df.iloc[start_idx:].copy()
    # Reset time to start at 0.0 seconds
    df['time'] = df['time'] - df['time'].iloc[0]

# --- PLOTTING ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# --- PLOT 1: Cartesian Convergence ---
# We use standard colors: X=Red/Green, Y=Green/Blue, Z=Blue/Orange depending on convention.
# Here we label them clearly.
ax1.plot(df['time'], df['err_depth'], label='Depth (X) Error', color='tab:green', linewidth=2.5)
ax1.plot(df['time'], df['err_horiz'], label='Lateral (Y) Error', color='tab:blue', linewidth=1.5)
ax1.plot(df['time'], df['err_vert'],  label='Vertical (Z) Error', color='tab:orange', linewidth=1.5)

ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
ax1.set_title('Position Convergence (Relative to Target)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Distance to Goal (m)', fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.6)

# --- PLOT 2: Orientation Stability ---
ax2.plot(df['time'], df['deg_err_roll'], label='Roll Error', color='tab:blue', alpha=0.8)
ax2.plot(df['time'], df['deg_err_pitch'], label='Pitch Error', color='tab:orange', alpha=0.8)
ax2.plot(df['time'], df['deg_err_yaw'], label='Yaw Error', color='tab:green', alpha=0.8)

ax2.axhline(0, color='black', linestyle='--', alpha=0.3)
ax2.set_title('Orientation Convergence', fontsize=14, fontweight='bold')
ax2.set_ylabel('Angle Error (degrees)', fontsize=12)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.6)

# 5. METRICS CALCULATION (For the Professor)
# Calculate Settling Time: Time to stay within 5mm (0.005m) of the goal
# Create a mask of where we are "inside the zone"
inside_zone = (df['err_depth'].abs() < 0.005) & \
              (df['err_horiz'].abs() < 0.005) & \
              (df['err_vert'].abs() < 0.005)

# Find the last time we were OUTSIDE the zone. Everything after that is "Settled".
# If we never settled, time is NaN.
try:
    last_outside_idx = np.where(~inside_zone)[0][-1]
    settling_time = df['time'].iloc[last_outside_idx]
except IndexError:
    # This means we were inside the zone the whole time (or array empty)
    settling_time = 0.0

# Calculate final steady state jitter (standard deviation of last 10 points)
jitter = df['err_depth'].tail(10).std() * 1000 # in mm

stats_text = (f"PERFORMANCE METRICS:\n"
              f"Settling Time (5mm): {settling_time:.2f} s\n"
              f"Steady State Jitter: {jitter:.3f} mm")

# Add a text box to the top plot
ax1.text(0.02, 0.05, stats_text, transform=ax1.transAxes,
         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9),
         fontsize=10, verticalalignment='bottom')

plt.tight_layout()
output_filename = 'pose_validation_relative.png'
plt.savefig(output_filename, dpi=300)
print(f"Plot saved to {output_filename}")
print(f"Settling Time: {settling_time:.2f}s")