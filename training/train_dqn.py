# training/train_dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from env.traffic_env import TrafficEnv
from models.dqn import DQN
from models.replay_buffer import ReplayBuffer

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MIN_REPLAY_SIZE = 2000
TARGET_UPDATE_FREQ = 1000
EPISODES = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# EPSILON-GREEDY ACTION SELECTION

def select_action(q_net, state, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)

    state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    q_values = q_net(state_t)
    return int(torch.argmax(q_values).item())



# MAIN TRAINING LOOP

def main():
    env = TrafficEnv()
    replay = ReplayBuffer()

    # Reset returns (state, info)
    state, _ = env.reset()
    state_dim = len(state)
    action_dim = env.action_space.n

    q_net = DQN(state_dim, action_dim).to(DEVICE)
    target_net = DQN(state_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LR)

    
    # FILL REPLAY BUFFER WITH RANDOM ACTIONS
    
    print("Initializing replay buffer...")

    while len(replay) < MIN_REPLAY_SIZE:
        action = random.randint(0, action_dim - 1)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay.push(state, action, reward, next_state, done)
        state = next_state

        if done:
            env.close()
            state, _ = env.reset()

    print("Replay buffer initialized.")

    
    # TRAINING LOOP
    
    global_step = 0

    for ep in range(EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            epsilon = max(0.05, 1.0 - global_step / 30000)
            action = select_action(q_net, state, epsilon, action_dim)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            global_step += 1

            
            # SAMPLE MINI-BATCH
            
            states, actions, rewards, next_states, dones = replay.sample(BATCH_SIZE)

            states = torch.tensor(states, dtype=torch.float32, device=DEVICE)
            actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
            next_states = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
            dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)

            # Q(s,a)
            q_values = q_net(states).gather(1, actions)

            # Target: r + γ max_a' Q_target(s', a')
            with torch.no_grad():
                next_q = target_net(next_states).max(dim=1, keepdim=True)[0]
                target = rewards + GAMMA * next_q * (1 - dones)

            loss = nn.MSELoss()(q_values, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update target network
            if global_step % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(q_net.state_dict())

        env.close()
        print(f"Episode {ep} | Reward: {total_reward:.2f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
