import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
# UPDATE THIS with your filename
filename = 'ibvs_hardware_test_173846.csv'

# Safety Limit (rad/s)
# AR4 Steppers usually slip above 2.0 rad/s. 
# We set a conservative limit of 1.5 rad/s.
SAFETY_LIMIT = 1.5 
# ---------------------

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"Error: Could not find file '{filename}'. Check the filename.")
    exit()

# Auto-crop waiting time (Start when Error changes or Vel > 0)
# We use the cartesian velocity to find the start of movement
move_start = df[df['cart_vel_x'].abs() > 0.001].index
if len(move_start) > 0:
    start_idx = max(0, move_start[0] - 10) 
    df = df.iloc[start_idx:].copy()
    df['time'] = df['time'] - df['time'].iloc[0]

# --- DIAGNOSTICS (Debug Mislabeled Data) ---
# This block helps identify if joint indices are swapped
all_joints = ['joint_1_vel', 'joint_2_vel', 'joint_3_vel', 
              'joint_4_vel', 'joint_5_vel', 'joint_6_vel']

print("-" * 40)
print("JOINT VELOCITY DIAGNOSTICS (Peak Values):")
print("-" * 40)
possible_mapping_issue = False
for joint in all_joints:
    # Handle case where column might be missing if robot has < 6 joints
    if joint in df.columns:
        peak = df[joint].abs().max()
        print(f"{joint}: {peak:.4f} rad/s")
        if peak < 0.0001:
            print(f"  >>> WARNING: {joint} is completely static (0.0).")
            possible_mapping_issue = True
    else:
        print(f"{joint}: NOT FOUND in CSV")

if possible_mapping_issue:
    print("\n[!] POTENTIAL MAPPING ISSUE DETECTED")
    print("If a joint SHOULD be moving but shows 0.0, your robot driver might")
    print("be publishing joint_states in a different order (e.g. alphabetical).")
print("-" * 40)
print("\nGenerating Plots...")

# Setup Plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# --- PLOT 1: Major Joints (Base, Shoulder, Elbow) ---
ax1.plot(df['time'], df['joint_1_vel'], label='J1 (Base)', linewidth=2)
ax1.plot(df['time'], df['joint_2_vel'], label='J2 (Shoulder)', linewidth=2)
ax1.plot(df['time'], df['joint_3_vel'], label='J3 (Elbow)', linewidth=2)

# Draw Safety Lines
ax1.axhline(SAFETY_LIMIT, color='red', linestyle='--', alpha=0.5, label='Safety Limit')
ax1.axhline(-SAFETY_LIMIT, color='red', linestyle='--', alpha=0.5)

ax1.set_title('Major Arm Joints Velocity (J1 - J3)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Velocity (rad/s)', fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.6)

# --- PLOT 2: Wrist Joints (Pitch, Roll, Yaw) ---
ax2.plot(df['time'], df['joint_4_vel'], label='J4 (Wrist Roll)', linewidth=2)
ax2.plot(df['time'], df['joint_5_vel'], label='J5 (Wrist Pitch)', linewidth=2)
ax2.plot(df['time'], df['joint_6_vel'], label='J6 (Wrist Yaw)', linewidth=2)

# Draw Safety Lines
ax2.axhline(SAFETY_LIMIT, color='red', linestyle='--', alpha=0.5, label='Safety Limit')
ax2.axhline(-SAFETY_LIMIT, color='red', linestyle='--', alpha=0.5)

ax2.set_title('Wrist Joints Velocity (J4 - J6)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Velocity (rad/s)', fontsize=12)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.6)

# --- SAFETY REPORT ON PLOT ---
max_vel = df[all_joints].abs().max().max()
violating_joints = []

for joint in all_joints:
    if joint in df.columns and df[joint].abs().max() > SAFETY_LIMIT:
        violating_joints.append(joint)

# Create Status Box
if max_vel < SAFETY_LIMIT:
    status_color = 'green'
    status_text = f"STATUS: SAFE\nMax Vel: {max_vel:.2f} rad/s\nLimit: {SAFETY_LIMIT} rad/s"
else:
    status_color = 'red'
    status_text = (f"STATUS: UNSAFE\n"
                   f"Max Vel: {max_vel:.2f} rad/s\n"
                   f"Violations: {', '.join(violating_joints)}")

ax1.text(0.02, 0.95, status_text, transform=ax1.transAxes, 
         verticalalignment='top', fontweight='bold', color=status_color,
         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=status_color, alpha=0.9))

plt.tight_layout()
output_filename = 'joint_safety_check.png'
plt.savefig(output_filename, dpi=300)

print(f"Plot saved to {output_filename}")
if max_vel > SAFETY_LIMIT:
    print(f"WARNING: Velocity limit exceeded! Max found: {max_vel:.2f} rad/s")
else:
    print(f"System Check: PASSED. Max velocity {max_vel:.2f} rad/s is within safe limits.")