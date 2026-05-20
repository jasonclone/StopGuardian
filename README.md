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

** run a file using docker; e.g. train.py
docker run --rm -it stopguardian python3 rl/train.py

** to run the default sumo simulation (uses fixed time traffic signals) manually on gui
<...\StopGuardian> sumo-gui -c simulation/sumo/test.sumocfg


** StopGuardian Pipeline Guide:

* run b1.py (run fixed time traffic signal baseline and collect metrics)
<...\StopGuardian> python baselines/b1.py

* run rl/train.py (train the rainbow and standard dqn models and collect metrics over the same seeds. This will take a while)
e.g.
<...\StopGuardian> python rl/train.py --model rainbow --episodes 250 --steps-per-episode 1000 --seeds 42 43 44
<...\StopGuardian> python rl/train.py --model standard --episodes 250 --steps-per-episode 1000 --seeds 42 43 44

* View results using:

* Tensorboard metrics
e.g.
tensorboard --logdir results/rl/rainbow/seed_42 --port 6006
tensorboard --logdir results/rl/standard/seed_43 --port 6006
tensorboard --logdir results/b1 --port 6006

* Performance report comparison between Rainbow and Standard DQN models
<...\StopGuardian> python rstats.py

* Exploratory Data Analysis
<...\StopGuardian> python rl/eda.py

* Demo visualization that runs parallel Rainbow DQN and the fixed time signal baseline on the SUMO GUI over seeds (THIS FILE CANNOT BE RUN ON WINDOWS DOCKER CONTAINER; SUMO GUI IS NOT COMPATIBLE WITH IT).
<...\StopGuardian> python demo.py





