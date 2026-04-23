# learner.py
import os
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
# Soft update helper (Polyak averaging)
# ================================================================
def soft_update(target_model, online_model, tau: float = 0.01):
    """
    Soft-update target network parameters:
    theta_target = tau * theta_online + (1 - tau) * theta_target
    """
    if target_model is None:
        return
    for t_param, o_param in zip(target_model.parameters(), online_model.parameters()):
        t_param.data.copy_(tau * o_param.data + (1.0 - tau) * t_param.data)


# ================================================================
# Training Step for Rainbow DQN (NoisyNet support, sigmas are learnable)
# ================================================================
def train_step_rainbow(
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
    current_step_global,
    priority_beta_start: float,
    priority_beta_end: float,
    total_updates: int,
    normalize_state_torch,
    tau: float = 0.01,
):
    """
    Learner for Rainbow DQN where NoisyNet sigma parameters are learned by the optimizer.
    This function:
      - computes PER beta internally (annealed from start to end over total_updates)
      - resets factorized noise each update (reset_noise)
      - computes C51 projection targets using the target network (deterministic)
      - performs a gradient step on the online network
      - performs a soft update (Polyak averaging) of the target network
      - updates PER priorities
    Returns:
      - loss (float) or None if not enough data / numerical issues
    """

    if len(replay_buffer) < min_replay_size:
        return None

    # Compute beta (anneal from start to end over planned updates)
    if total_updates <= 0:
        beta = float(priority_beta_end)
    else:
        frac = min(1.0, float(current_step_global) / float(total_updates))
        beta = float(priority_beta_start + frac * (priority_beta_end - priority_beta_start))
        beta = min(1.0, max(0.0, beta))

    # Reset NoisyNet noise for this update (online and target)
    try:
        online_model.reset_noise()
    except Exception:
        pass

    try:
        if target_model is not None:
            target_model.reset_noise()
    except Exception:
        pass

    # Sample batch (PER uses beta)
    states, actions, rewards, next_states, dones, idxs, weights = replay_buffer.sample(
        batch_size, beta
    )

    states_tensor = torch.from_numpy(states).float().to(device)
    states_tensor = normalize_state_torch(states_tensor)

    next_states_tensor = torch.from_numpy(next_states).float().to(device)
    next_states_tensor = normalize_state_torch(next_states_tensor)

    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device)
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    # input sanity checks
    if (
        torch.isnan(states_tensor).any() or torch.isinf(states_tensor).any() or
        torch.isnan(next_states_tensor).any() or torch.isinf(next_states_tensor).any() or
        torch.isnan(rewards_tensor).any() or torch.isinf(rewards_tensor).any()
    ):
        return None

    # Data augmentation: Gaussian noise injection (optional)
    if online_model.training:
        states_tensor += 0.01 * torch.randn_like(states_tensor)
        next_states_tensor += 0.01 * torch.randn_like(next_states_tensor)

    gamma_n = gamma ** n_steps

    # =========================
    # Target distribution (C51 projection)
    # =========================
    online_model.eval()
    if target_model is not None:
        target_model.eval()

    with torch.no_grad():
        next_q_values = online_model.q_values(next_states_tensor)
        if torch.isnan(next_q_values).any() or torch.isinf(next_q_values).any():
            return None

        next_actions = next_q_values.argmax(dim=1)

        if target_model is None:
            return None  # Rainbow requires a target model for distributional targets

        target_logits = target_model(next_states_tensor)
        if torch.isnan(target_logits).any() or torch.isinf(target_logits).any():
            return None

        target_logits = target_logits.gather(
            1, next_actions.view(-1, 1, 1).expand(-1, 1, num_atoms)
        ).squeeze(1)

        target_probs = torch.softmax(target_logits, dim=-1)
        if torch.isnan(target_probs).any() or torch.isinf(target_probs).any():
            return None

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

        if torch.isnan(proj_dist).any() or torch.isinf(proj_dist).any():
            return None

        proj_dist = torch.clamp(proj_dist, min=0.0)
        proj_dist_sum = proj_dist.sum(dim=1, keepdim=True)
        proj_dist = proj_dist / torch.clamp(proj_dist_sum, min=1e-8)

    # Predicted distribution and loss
    online_model.train()

    logits = online_model(states_tensor)
    logits = logits.gather(
        1, actions_tensor.view(-1, 1, 1).expand(-1, 1, num_atoms)
    ).squeeze(1)

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)

    if (
        torch.isnan(log_probs).any() or torch.isinf(log_probs).any() or
        torch.isnan(probs).any() or torch.isinf(probs).any()
    ):
        return None

    loss_per_sample = -(proj_dist * log_probs).sum(dim=1)
    loss = (weights_tensor * loss_per_sample).mean()

    if torch.isnan(loss) or torch.isinf(loss):
        return None

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_model.parameters(), 10.0)
    optimizer.step()

    # Soft-update target network (Polyak averaging)
    if target_model is not None:
        soft_update(target_model, online_model, tau=tau)

    # PER update: compute TD errors for priority update
    with torch.no_grad():
        q_pred = torch.sum(probs * support, dim=1)
        q_target = torch.sum(proj_dist * support, dim=1)
        td_errors = (q_target - q_pred).cpu().numpy()

    try:
        replay_buffer.update_priorities(idxs, td_errors)
    except Exception:
        pass

    return float(loss.item())


# ================================================================
# Training Step for standard DQN baseline model
# ================================================================
def train_step_standard(
    online_model,  # no target network (pure single dqn)
    replay_buffer,
    optimizer,
    device,
    batch_size,
    min_replay_size,
    gamma,
    normalize_state_torch,
):
    if len(replay_buffer) < min_replay_size:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states = torch.from_numpy(states).float().to(device)
    next_states = torch.from_numpy(next_states).float().to(device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    states = normalize_state_torch(states)
    next_states = normalize_state_torch(next_states)

    # Q(s,a)
    q_values = online_model.q_values(states)
    q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # PURE DQN TARGET: r + γ max_a' Q_online(s')
    with torch.no_grad():
        next_q = online_model.q_values(next_states).max(1)[0]
        target = rewards + gamma * (1 - dones) * next_q

    loss = torch.nn.functional.mse_loss(q_taken, target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_model.parameters(), 10.0)
    optimizer.step()

    return float(loss.item())
