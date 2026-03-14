# train_dqn.py
import os
import csv
import traceback

import numpy as np

from traffic_env import TrafficEnv
from dqn_agent import DQNAgent


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_LOG_CSV = os.path.join(RESULTS_DIR, "dqn_training_log.csv")


def train_dqn(
    num_episodes: int = 200,
    max_steps_per_episode: int | None = None,
    save_every: int = 50,
):
    env = TrafficEnv(render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        tau=0.005,
    )

    if max_steps_per_episode is None:
        max_steps_per_episode = env.max_steps

    with open(TRAIN_LOG_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode", "episode_reward", "steps", "avg_loss", "epsilon"],
        )
        writer.writeheader()

        try:
            for ep in range(1, num_episodes + 1):
                state, _ = env.reset()
                done = False
                truncated = False
                ep_reward = 0.0
                ep_losses = []
                steps = 0

                while not (done or truncated) and steps < max_steps_per_episode:
                    action = agent.act(state)
                    next_state, reward, done, truncated, info = env.step(action)

                    agent.remember(state, action, reward, next_state, done or truncated)
                    loss = agent.train_step()
                    if loss is not None:
                        ep_losses.append(loss)

                    state = next_state
                    ep_reward += reward
                    steps += 1

                avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
                eps = agent.epsilon()

                writer.writerow(
                    {
                        "episode": ep,
                        "episode_reward": ep_reward,
                        "steps": steps,
                        "avg_loss": avg_loss,
                        "epsilon": eps,
                    }
                )
                f.flush()

                print(
                    f"[Episode {ep}/{num_episodes}] "
                    f"Reward={ep_reward:.1f}, Steps={steps}, "
                    f"AvgLoss={avg_loss:.4f}, Eps={eps:.3f}"
                )

                if ep % save_every == 0:
                    model_path = os.path.join(MODEL_DIR, f"dqn_ep{ep}.pt")
                    agent_state = {
                        "q_net": agent.q_net.state_dict(),
                        "target_net": agent.target_net.state_dict(),
                        "total_steps": agent.total_steps,
                    }
                    import torch

                    torch.save(agent_state, model_path)
                    print(f"Saved model to {model_path}")

        except Exception:
            print("ERROR during DQN training")
            traceback.print_exc()
        finally:
            env.close()


if __name__ == "__main__":
    train_dqn()
