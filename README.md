# StopGuardian
Deep Reinforcement Learning–Based Adaptive Traffic Signal Optimization  
Using SUMO and a Rainbow DQN agent.

## Overview
StopGuardian implements a full RL pipeline for traffic signal control at a single urban intersection.  
The project includes:

- A custom SUMO-based environment for the agent to understand traffic scenarios
- A fixed-time baseline controller (b1.py)
- A Rainbow DQN agent for adaptive signal control 
- An Exploratory Data Analysis (EDA) pipeline
- Training, evaluation, and visualization tools 

All simulations use the SUMO traffic simulator.

---
*** Important commands

** Build and run the Docker container (not recommended for testing purposes). 
Make sure container is running via Docker Desktop. This will build the Docker image with the tag "stopguardian." 
<...\StopGuardian> docker build -t stopguardian .
<...\StopGuardian> docker run --rm -it stopguardian


** to run the default sumo simulation manually on gui
<...\StopGuardian> sumo-gui -c simulation/sumo/test.sumocfg

** To get and compare results of the baseline and trained rl agent:

run train.py (train the agent and get its performance. This will take a while)
<...\StopGuardian> python rl/train.py

run b1.py (same but for fixed traffic signal baseline)
<...\StopGuardian> python python baselines/b1.py

run compare.py (generated txt file with comparison details). Make sure baseline and rl training run for same amount of steps for accurate comparison.
<...\StopGuardian> python compare.py


** Exploratory Data Analysis of the results of the baseline and trained rl agent
<...\StopGuardian> python eda.py

tensorboard command: tensorboard --logdir results/rl/tb
it will give localhost link to view the stats