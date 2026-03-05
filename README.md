# Dual-Arm Visual Servoing & Autonomous Manipulation

![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy-22314E?logo=ros)
![C++](https://img.shields.io/badge/C++-17-00599C?logo=c%2B%2B)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)
![Gazebo](https://img.shields.io/badge/Simulation-Gazebo-FFB500?logo=gazebo)

> **Note:** A video demonstration of the dual-arm simulation and visual servoing pipeline can be found here: [Insert Link to YouTube/LinkedIn Video]

https://github.com/user-attachments/assets/ed57de71-15a6-4d2e-9d5c-0abc3e6d334b

## 📌 Project Overview

This repository contains the software architecture and control pipeline for a collaborative, dual-arm robotic manipulation system. Built natively in **ROS 2 Jazzy**, the system bridges the gap between complex 3D perception and physical actuation. 

The core focus of this project is the implementation of highly responsive **Image-Based (IBVS)** and **Position-Based Visual Servoing (PBVS)** algorithms. By processing camera feeds in real-time and calculating the necessary homography and feature transformations, the system coordinates two independent manipulators to execute precise, collision-aware tasks in a shared workspace.

## 🚀 Key Features

* **Visual Servoing Pipeline:** Custom C++ and Python nodes executing both IBVS and PBVS logic to dynamically calculate joint velocities based on 2D image features and 3D spatial estimations.
* **Fisheye & Homography Integration:** Robust handling of fisheye camera distortion models and homography-based 3D reconstruction for accurate perception.
* **Collaborative Dual-Arm Control:** Coordinated motion planning utilizing **MoveIt 2** for collision avoidance and trajectory generation within a shared operational space.
* **Ground-Truth Validation Node:** A custom architectural component designed to rigorously test and validate the perception logic against the simulation's ground-truth state before physical deployment.
* **Complex TF2 Management:** Comprehensive handling of robotic kinematics and dynamic coordinate frame transformations (TF2) between the world, cameras, and multiple end-effectors.

## 🛠️ Tech Stack

* **Middleware:** ROS 2 (Jazzy Jalisco)
* **Languages:** C++, Python 3
* **Computer Vision:** OpenCV
* **Motion Planning:** MoveIt 2
* **Simulation & Visualization:** Gazebo, RViz

## 🏗️ System Architecture
![WhatsApp Image 2026-03-05 at 5 24 56 PM](https://github.com/user-attachments/assets/d09a9392-03a7-4177-8201-47a710e254d7)


