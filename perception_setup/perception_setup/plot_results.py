import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
filename = 'pose_validation_YOUR_FILE.csv' 

# 1. SETUP GOALS BASED ON YOUR CONTROLLER CONFIGURATION
# Your Controller uses X as Forward/Depth
GOAL_DEPTH_X = 0.2093  # The target depth (0.2093m)
GOAL_LATERAL_Y = 0.0   # Center
GOAL_VERTICAL_Z = 0.0  # Center
# ---------------------

df = pd.read_csv(filename)

# 2. CALCULATE ERRORS (Mapping Axes Correctly)
# ERROR = Current_Pos - Goal
# We want to show the error decaying to zero.

df['err_depth'] = df['x'] - GOAL_DEPTH_X  # X is Depth in your setup
df['err_horiz'] = df['y'] - GOAL_LATERAL_Y
df['err_vert']  = df['z'] - GOAL_VERTICAL_Z

# Convert Radians to Degrees
df['deg_roll'] = np.degrees(df['roll'])
df['deg_pitch'] = np.degrees(df['pitch'])
df['deg_yaw'] = np.degrees(df['yaw'])

# 3. AUTO-CROP (Find when the robot starts moving)
# Detect movement in X (Depth axis)
move_start = df[ (df['x'].diff().abs() > 0.0001) ].index
if len(move_start) > 0:
    start_idx = max(0, move_start[0] - 10) # 0.5s buffer
    df = df.iloc[start_idx:].copy()
    df['time'] = df['time'] - df['time'].iloc[0]

# --- PLOT 1: Position Accuracy (Corrected Axes) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Note the labels!
ax1.plot(df['time'], df['err_depth'], label='X Error (Depth)', color='green', linewidth=2.5)
ax1.plot(df['time'], df['err_horiz'], label='Y Error (Lateral)', color='blue', linewidth=1.5, alpha=0.7)
ax1.plot(df['time'], df['err_vert'],  label='Z Error (Vertical)', color='orange', linewidth=1.5, alpha=0.7)

ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
ax1.set_title('Cartesian Position Error (Robot Frame)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Error (meters)', fontsize=12)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

# Add "Zoom In" Text Box for Final Accuracy
final_depth_err = df['err_depth'].iloc[-1] * 1000 # mm
final_horiz_err = df['err_horiz'].iloc[-1] * 1000
final_vert_err  = df['err_vert'].iloc[-1] * 1000

stats = (f"Final Accuracy (mm):\n"
         f"Depth (X): {final_depth_err:.2f}\n"
         f"Horiz (Y): {final_horiz_err:.2f}\n"
         f"Vert  (Z): {final_vert_err:.2f}")

ax1.text(0.02, 0.05, stats, transform=ax1.transAxes, 
         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))


# --- PLOT 2: Orientation Error ---
ax2.plot(df['time'], df['deg_roll'], label='Roll', linewidth=2)
ax2.plot(df['time'], df['deg_pitch'], label='Pitch', linewidth=2)
ax2.plot(df['time'], df['deg_yaw'], label='Yaw', linewidth=2)

ax2.axhline(0, color='black', linestyle='--', alpha=0.3)
ax2.set_title('Orientation Alignment Error', fontsize=14)
ax2.set_ylabel('Error (degrees)', fontsize=12)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('pose_validation_FINAL.png', dpi=300)
print(f"Plot saved. Final Depth Error: {final_depth_err:.3f} mm")