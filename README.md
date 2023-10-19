# Shared_control_architecture
This is my work of thesis:  "A shared control framework for mobile robot navigation based on Deep Reinforcement Learning"
Thesis paper is available at https://drive.google.com/file/d/1EZi89MBowX0mgvdnZjqGqDaWyGCuPqtj/view?usp=sharing


The objective is to develop a human-robot shared control system for mobile robots, that ensures a safe navigation in unknown and complex environments, using DRL techniques to adapt the autonomy level of the system.

The simulation architecture has been implemented using ROS-noetic framework for TIAgo robot in Gazebo.



## Installation
First, make sure everything is working properly make sure you have Ubuntu 20.04 with
ROS Noetic. To make sure all necessary pakages and dependences are installed, follow the TIAGo's tutorial in "http://wiki.ros.org/Robots/TIAGo/Tutorials/Installation/InstallUbuntuAndROS", to instal Ubuntu with ROS + TIAGO.

Consequently, create a catkin workspace and clone this
repository in the `src` folder. Then, build your code by running the following command:
```bash
catkin build
```


## Usage
In order to be able to run the Gazebo simulation, use the following command
```bash
roslaunch crowd_navigation_gazebo tiago_gazebo.launch public_sim:=true end_effector:=pal-gripper world:=WORLD
```
where `WORLD` is one of the worlds in the folder `gazebo_sim/worlds`.

The navigation model has two MODEs: "test" (using a DDQN agent for the arbitration) and "classic" (using a rule based arbitration). To run the navigation module with the human-like module, run:
```bash
roslaunch SC_navigation start_testing.launch mode:=MODE model:=MODEL
```
where MODEL is set in mode test and can be choosen in the folder weights.

To run the navigation module with the teleoperate module, run:
```bash
roslaunch SC_navigation start_teleop_base.launch mode:=MODE model:=MODEL
```
