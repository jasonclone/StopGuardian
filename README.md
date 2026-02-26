# StopGuardian

Deep Reinforcement Learning-Based Adaptive Traffic Signal Optimization

## Project Overview
This project explores the use of deep reinforcement learning (DQN) to optimize traffic signal control at urban intersections using the SUMO traffic simulator.

## Baseline
A fixed-time traffic signal controller is implemented as a baseline for comparison.

## Tools
- Python
- PyTorch
- SUMO


Important Info:

*** run sumo gui
sumo-gui -c simulation.sumocfg

***generate sumo network (go to simulation/sumo directory)
** NOTE: List build time input files in command, not runtime (like routes.rou.xml)
netconvert -n intersection.nod.xml -e intersection.edg.xml -o intersection.net.xml

*** Build and run the Docker container. Make sure container is running via Docker Desktop. This will build the Docker image with the tag "stopguardian." 
docker build -t stopguardian .
docker run --rm -it stopguardian

*** To run without docker e.g.:
python baselines/fixed_time_metrics.py
