# rl/model.py
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from config import (
    DEVICE,
    GAMMA,
    N_STEPS,
    BUFFER_SIZE,
    BATCH_SIZE,
    NOISY_SIGMA,
    MIN_REPLAY_SIZE,
    WARMUP_STEPS,
    USE_EPSILON,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY_STEPS,
    ACTIONS,
    NUM_ACTIONS,
    PRIORITY_ALPHA,
    PRIORITY_BETA_START,
    PRIORITY_BETA_END,
    TAU,
    LEARNING_RATE,
    CHECKPOINT_EVERY_EPISODES,
    NUM_ATOMS,
    V_MIN,
    V_MAX,
    DELTA_Z,
)


class NoisyLinear(nn.Module):
    """Factorized Noisy Linear layer with trainable sigmas."""
    def __init__(self, in_features, out_features, sigma_init=NOISY_SIGMA):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        mu_range = 1.0 / np.sqrt(in_features)
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-mu_range, mu_range)
        )
        self.bias_mu = nn.Parameter(torch.empty(out_features).uniform_(-mu_range, mu_range))

        base_sigma = float(sigma_init) / np.sqrt(in_features)
        self.register_buffer("weight_sigma_base", torch.full((out_features, in_features), base_sigma))
        self.register_buffer("bias_sigma_base", torch.full((out_features,), base_sigma))

        self.weight_sigma = nn.Parameter(self.weight_sigma_base.clone())
        self.bias_sigma = nn.Parameter(self.bias_sigma_base.clone())

        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))
        
        self.noise_enabled = True

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.sqrt(torch.abs(x))

    def disable_noise(self):
        self.noise_enabled = False

    def enable_noise(self):
        self.noise_enabled = True

    def reset_noise(self, device=DEVICE):
        """
        Reset factorized noise. If device is provided, create noise on that device.
        Otherwise infer device from parameters.
        """
        try:
            if device is None:
                device = self.weight_mu.device
            eps_in = torch.randn(self.in_features, device=device)
            eps_out = torch.randn(self.out_features, device=device)
            f_in = self._f(eps_in)
            f_out = self._f(eps_out)
            # use outer product (ger) to create factorized noise
            self.weight_epsilon.copy_(f_out.ger(f_in))
            self.bias_epsilon.copy_(f_out)
        except Exception as e:
            print("NoisyLinear.reset_noise failed:", e)

    def forward(self, x):
        try:
            if self.training and self.noise_enabled:
                # explicit scalar checks to avoid ambiguous tensor->bool conversion
                if self.weight_epsilon.abs().sum().item() == 0 and self.bias_epsilon.abs().sum().item() == 0:
                    # ensure noise is created on the same device as the input
                    self.reset_noise(device=x.device)
                weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
                bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                weight = self.weight_mu
                bias = self.bias_mu
            return x @ weight.t() + bias
        except Exception as e:
            print("NoisyLinear.forward failed:", e)
            # fallback: deterministic forward to avoid crash
            return x @ self.weight_mu.t() + self.bias_mu


class RainbowDQN(nn.Module):
    """Rainbow DQN with C51 distributional output and NoisyNet layers."""
    def __init__(self, state_size, num_actions, num_atoms, v_min, v_max):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms

        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(p=0.1)

        self.v_noisy1 = NoisyLinear(256, 256)
        self.v_noisy2 = NoisyLinear(256, num_atoms)

        self.a_noisy1 = NoisyLinear(256, 256)
        self.a_noisy2 = NoisyLinear(256, num_actions * num_atoms)

        self.relu = nn.ReLU()
        # support will be moved with the model when .to(device) is called
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))

    def forward(self, x):
        try:
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.dropout(x)

            v = self.relu(self.v_noisy1(x))
            v = self.v_noisy2(v)

            a = self.relu(self.a_noisy1(x))
            a = self.a_noisy2(a)
            a = a.view(-1, self.num_actions, self.num_atoms)

            v = v.view(-1, 1, self.num_atoms)
            a_mean = a.mean(dim=1, keepdim=True)
            return v + (a - a_mean)
        except Exception as e:
            print("RainbowDQN.forward failed:", e)
            # fallback: return zeros to avoid crashing caller
            batch = x.shape[0] if x is not None else 1
            return torch.zeros(batch, self.num_actions, self.num_atoms, device=next(self.parameters()).device)

    def q_values(self, x):
        try:
            probs = torch.softmax(self.forward(x), dim=-1)
            return torch.sum(probs * self.support, dim=-1)
        except Exception as e:
            print("RainbowDQN.q_values failed:", e)
            # fallback: zeros
            batch = x.shape[0] if x is not None else 1
            return torch.zeros(batch, self.num_actions, device=next(self.parameters()).device)
        
    
    def disable_noise(self):
        self.v_noisy1.disable_noise()
        self.v_noisy2.disable_noise()
        self.a_noisy1.disable_noise()
        self.a_noisy2.disable_noise()

    def enable_noise(self):
        self.v_noisy1.enable_noise()
        self.v_noisy2.enable_noise()
        self.a_noisy1.enable_noise()
        self.a_noisy2.enable_noise()


    def reset_noise(self, device=DEVICE):
        """
        Reset noise for all NoisyLinear layers.
        Accepts an optional device argument and forwards it to sublayers.
        If device is None, infer from model parameters.
        """
        try:
            if device is None:
                try:
                    device = next(self.parameters()).device
                except StopIteration:
                    device = None

            for layer_name in ("v_noisy1", "v_noisy2", "a_noisy1", "a_noisy2"):
                layer = getattr(self, layer_name, None)
                if layer is None:
                    continue
                try:
                    layer.reset_noise(device=device)
                except Exception as e:
                    print(f"Failed resetting noise for layer {layer_name}:", e)
        except Exception as e:
            print("RainbowDQN.reset_noise failed:", e)


class DQN(nn.Module):
    """Baseline DQN for comparison."""
    def __init__(self, state_size, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        try:
            return self.net(x)
        except Exception as e:
            print("DQN.forward failed:", e)
            batch = x.shape[0] if x is not None else 1
            return torch.zeros(batch, self.net[-1].out_features, device=next(self.parameters()).device)

    def q_values(self, x):
        return self.forward(x)


def _make_linear_decay_scheduler(optimizer: optim.Optimizer, lr_decay_steps: int):
    """Return a LambdaLR that linearly decays base_lr -> 0 over lr_decay_steps."""
    if lr_decay_steps is None or lr_decay_steps <= 0:
        raise ValueError("lr_decay_steps must be a positive integer")
    def lr_lambda(step):
        progress = min(float(step) / float(lr_decay_steps), 1.0)
        return 1.0 - progress
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _wrap_optimizer_with_scheduler(optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler):
    """
    Wrap optimizer.step so the scheduler is stepped exactly once per optimizer.step().
    Preserves metadata and exposes scheduler as optimizer.scheduler.
    """
    original_step = optimizer.step

    @functools.wraps(original_step)
    def step_with_scheduler(*args, **kwargs):
        result = original_step(*args, **kwargs)
        try:
            scheduler.step()
        except Exception as e:
            print("Scheduler.step failed inside wrapped optimizer.step:", e)
        return result

    optimizer.step = step_with_scheduler
    optimizer.scheduler = scheduler
    return optimizer


def build_rainbow_model(state_size,  lr_decay_steps, num_actions = NUM_ACTIONS, num_atoms = NUM_ATOMS, v_min = V_MIN, v_max = V_MAX, lr = LEARNING_RATE, device = DEVICE, weight_decay: float = 1e-5):
    """
    Build Rainbow model and optimizer.
    - Returns (model, optimizer) for backward compatibility.
    - Internally creates a linear-decay scheduler and auto-steps it after each optimizer.step().
    - Scheduler is exposed as optimizer.scheduler for logging/checkpointing.
    """
    model = RainbowDQN(state_size, num_actions, num_atoms, v_min, v_max).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1.5e-4, weight_decay=weight_decay)

    scheduler = _make_linear_decay_scheduler(optimizer, lr_decay_steps)
    optimizer = _wrap_optimizer_with_scheduler(optimizer, scheduler)
    return model, optimizer


def build_standard_model(state_size, lr_decay_steps, num_actions = NUM_ACTIONS, lr = LEARNING_RATE, device = DEVICE):
    """
    Build standard DQN model and optimizer.
    - Returns (model, optimizer) for backward compatibility.
    - Internally creates a linear-decay scheduler and auto-steps it after each optimizer.step().
    - Scheduler is exposed as optimizer.scheduler for logging/checkpointing.
    """
    model = DQN(state_size, num_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1.5e-4)
    
    scheduler = _make_linear_decay_scheduler(optimizer, lr_decay_steps)
    optimizer = _wrap_optimizer_with_scheduler(optimizer, scheduler)
    return model, optimizer


# --- Helpers for checkpointing scheduler state ---
def save_checkpoint(path, model, optimizer):
    """
    Save model, optimizer, and scheduler state (if present).
    """
    try:
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if hasattr(optimizer, "scheduler") and optimizer.scheduler is not None:
            payload["scheduler"] = optimizer.scheduler.state_dict()
        torch.save(payload, path)
    except Exception as e:
        print("save_checkpoint failed:", e)


def load_checkpoint(path, model, optimizer, device=DEVICE):
    """
    Load model, optimizer, and scheduler state.
    After loading optimizer state, rewrap optimizer.step to restore scheduler wrapper.
    """
    try:
        ckpt = torch.load(path, map_location=device or "cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and hasattr(optimizer, "scheduler"):
            try:
                optimizer.scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                print("Failed loading scheduler state:", e)
        # Rewrap optimizer.step in case load_state_dict replaced the wrapper
        if hasattr(optimizer, "scheduler") and optimizer.scheduler is not None:
            _wrap_optimizer_with_scheduler(optimizer, optimizer.scheduler)
    except Exception as e:
        print("load_checkpoint failed:", e)
    return model, optimizer


# --- Unit test snippet (run as a quick smoke test) ---
if __name__ == "__main__":
    # test to verify LR decays linearly
    dummy_state = 10
    dummy_actions = 2
    dummy_atoms = 51
    model, opt = build_rainbow_model(
        dummy_state, dummy_actions, dummy_atoms, -10, 10, lr=1e-3, device=DEVICE, lr_decay_steps=10
    )
    print("initial lr:", opt.param_groups[0]["lr"])
    for i in range(12):
        opt.zero_grad()
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.zeros_like(p)
        opt.step()
        print(f"step {i+1:02d} lr {opt.param_groups[0]['lr']:.6f}")
