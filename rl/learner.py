# rl/learner.py

import numpy as np
import torch


# ================================================================
# Distributional Projection (C51)
# ================================================================

def projection_distribution(
    next_dist,
    rewards,
    dones,
    gamma_n,
    support,
    v_min,
    v_max,
    delta_z,
):
    batch_size = rewards.size(0)
    num_atoms = support.size(0)

    rewards = rewards.unsqueeze(1)
    dones = dones.unsqueeze(1)

    tz = rewards + (1.0 - dones) * gamma_n * support.unsqueeze(0)
    tz = tz.clamp(v_min, v_max)

    b = (tz - v_min) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    l = l.clamp(0, num_atoms - 1)
    u = u.clamp(0, num_atoms - 1)

    proj_dist = torch.zeros(batch_size, num_atoms, device=next_dist.device)

    offset = torch.linspace(
        0,
        (batch_size - 1) * num_atoms,
        batch_size,
        device=next_dist.device,
    ).long().unsqueeze(1)

    proj_dist.view(-1).index_add_(
        0,
        (l + offset).view(-1),
        (next_dist * (u.float() - b)).view(-1),
    )
    proj_dist.view(-1).index_add_(
        0,
        (u + offset).view(-1),
        (next_dist * (b - l.float())).view(-1),
    )

    return proj_dist


# ================================================================
# Training Step
# ================================================================

def train_step(
    beta,
    online_model,
    target_model,
    replay_buffer,
    optimizer,
    device,
    batch_size,
    min_replay_size,
    gamma,
    n_steps,
    num_atoms,
    v_min,
    v_max,
    delta_z,
    normalize_state_torch, # passed in normalized function from env to be used for normalizing states in the sampled batch
):
    if len(replay_buffer) < min_replay_size:
        return None

    states, actions, rewards, next_states, dones, idxs, weights = replay_buffer.sample(
        batch_size, beta
    )

    # get sampled batch from replay buffer, normalize states and next states using env's normalize function, then convert to tensors
    states_tensor = torch.from_numpy(states).float().to(device)
    states_tensor = normalize_state_torch(states_tensor)

    next_states_tensor = torch.from_numpy(next_states).float().to(device)
    next_states_tensor = normalize_state_torch(next_states_tensor)
    
    
    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device)
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)


    gamma_n = gamma ** n_steps

    # =========================
    # Target distribution
    # =========================
    online_model.eval()
    target_model.eval()

    with torch.no_grad():
        next_q_values = online_model.q_values(next_states_tensor)
        next_actions = next_q_values.argmax(dim=1)

        target_logits = target_model(next_states_tensor)
        target_logits = target_logits.gather(
            1, next_actions.view(-1, 1, 1).expand(-1, 1, num_atoms)
        ).squeeze(1)

        target_probs = torch.softmax(target_logits, dim=-1)

        support = target_model.support

        proj_dist = projection_distribution(
            target_probs,
            rewards_tensor,
            dones_tensor,
            gamma_n,
            support,
            v_min,
            v_max,
            delta_z,
        )

    # =========================
    # Predicted distribution
    # =========================
    online_model.train()

    logits = online_model(states_tensor)
    logits = logits.gather(
        1, actions_tensor.view(-1, 1, 1).expand(-1, 1, num_atoms)
    ).squeeze(1)

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)

    loss_per_sample = -(proj_dist * log_probs).sum(dim=1)
    loss = (weights_tensor * loss_per_sample).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_model.parameters(), 10.0)
    optimizer.step()

    # =========================
    # PER update
    # =========================
    with torch.no_grad():
        q_pred = torch.sum(probs * support, dim=1)
        q_target = torch.sum(proj_dist * support, dim=1)
        td_errors = (q_target - q_pred).cpu().numpy()

    replay_buffer.update_priorities(idxs, td_errors)

    return float(loss.item())