"""Training callbacks for the coin-heaven DQN agent."""

from __future__ import annotations

import events as e
import torch

from .callbacks import (
    MODEL_FILE,
    TRAINING_SEED,
)
from .dqn import optimize_dqn, update_target_network
from .features import state_to_features
from .model import ACTIONS, CoinCollectorDQN
from .replay_buffer import ReplayBuffer

try:
    from tools.trainlog import TrainLogger
except ImportError:
    TrainLogger = None


# DQN hyperparameters
GAMMA = 0.99
LEARNING_RATE = 1e-4

REPLAY_CAPACITY = 20_000
BATCH_SIZE = 64
MIN_REPLAY_SIZE = 1_000
TARGET_UPDATE_INTERVAL = 1_000

# Linear epsilon schedule, measured in environment transitions
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 100_000

# Reward design for task 1
STEP_REWARD = -0.05

EVENT_REWARDS = {
    e.COIN_COLLECTED: 5.0,
    e.INVALID_ACTION: -1.0,
}

RUN_LABEL = "dqn_v7_mask_coin5_300ep_seed20260805"
CHECKPOINT_INTERVAL = 100


def setup_training(self) -> None:
    """Initialize all state required only during training."""
    if torch.backends.mps.is_available():
        self.device = torch.device("mps")
    else:
        self.device = torch.device("cpu")

    self.logger.info("Training DQN on device %s.", self.device)

    self.online_network.to(self.device)
    self.online_network.train()

    self.target_network = CoinCollectorDQN().to(self.device)
    update_target_network(
        self.online_network,
        self.target_network,
    )
    self.target_network.eval()

    for parameter in self.target_network.parameters():
        parameter.requires_grad_(False)

    self.optimizer = torch.optim.Adam(
        self.online_network.parameters(),
        lr=LEARNING_RATE,
    )

    self.replay_buffer = ReplayBuffer(
        capacity=REPLAY_CAPACITY,
    )

    self.environment_steps = 0
    self.optimization_steps = 0
    self.epsilon = EPSILON_START

    self.episode_reward = 0.0
    self.episode_events = []
    self.episode_losses = []

    self.trainlog = None

    if TrainLogger is not None:
        self.trainlog = TrainLogger(
            agent="ben_coin_collector_dqn",
            run=RUN_LABEL,
            hyperparams={
                "gamma": GAMMA,
                "learning_rate": LEARNING_RATE,
                "replay_capacity": REPLAY_CAPACITY,
                "batch_size": BATCH_SIZE,
                "min_replay_size": MIN_REPLAY_SIZE,
                "target_update_interval": TARGET_UPDATE_INTERVAL,
                "epsilon_start": EPSILON_START,
                "epsilon_end": EPSILON_END,
                "epsilon_decay_steps": EPSILON_DECAY_STEPS,
                "step_reward": STEP_REWARD,
                "coin_collected_reward": EVENT_REWARDS[
                    e.COIN_COLLECTED
                ],
                "invalid_action_reward": EVENT_REWARDS[
                    e.INVALID_ACTION
                ],
                "training_seed": TRAINING_SEED,
                "device": str(self.device),
            },
            extra_columns=["mean_loss", "buffer_size"],
            flush_every=1,
        )


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: list[str],
) -> None:
    """Store and learn from one non-terminal transition."""
    if old_game_state is None:
        return

    reward = reward_from_events(events)

    state = state_to_features(old_game_state)
    next_state = state_to_features(new_game_state)
    action_index = ACTIONS.index(self_action)

    self.replay_buffer.append(
        state=state,
        action=action_index,
        reward=reward,
        next_state=next_state,
        done=False,
    )

    self.episode_reward += reward
    self.episode_events.extend(events)

    _after_transition(self)


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: list[str],
) -> None:
    """Mark the final transition, save the model and log the episode."""
    self.replay_buffer.mark_last_terminal()

    # The final action events were already recorded by
    # game_events_occurred(). Only SURVIVED_ROUND is new here.
    if e.SURVIVED_ROUND in events:
        self.episode_events.append(e.SURVIVED_ROUND)

    cpu_state_dict = {
        name: parameter.detach().cpu()
        for name, parameter in self.online_network.state_dict().items()
    }

    torch.save(cpu_state_dict, MODEL_FILE)

    episode = last_game_state["round"]

    if episode % CHECKPOINT_INTERVAL == 0:
        checkpoint_file = (
            f"{RUN_LABEL}__episode_{episode}.pt"
        )

        torch.save(
            cpu_state_dict,
            checkpoint_file,
        )

        self.logger.info(
            "Saved checkpoint %s.",
            checkpoint_file,
        )

    if self.trainlog is not None:
        mean_loss = (
            sum(self.episode_losses) / len(self.episode_losses)
            if self.episode_losses
            else float("nan")
        )

        self.trainlog.log_episode(
            episode=last_game_state["round"],
            score=self.episode_events.count(e.COIN_COLLECTED),
            steps=last_game_state["step"],
            events=self.episode_events,
            reward=self.episode_reward,
            epsilon=self.epsilon,
            extra={
                "mean_loss": mean_loss,
                "buffer_size": len(self.replay_buffer),
            },
        )

    self.logger.info(
        "Episode %d finished: reward %.3f, epsilon %.4f, "
        "buffer %d, mean loss %s.",
        last_game_state["round"],
        self.episode_reward,
        self.epsilon,
        len(self.replay_buffer),
        (
            f"{sum(self.episode_losses) / len(self.episode_losses):.6f}"
            if self.episode_losses
            else "n/a"
        ),
    )

    self.episode_reward = 0.0
    self.episode_events = []
    self.episode_losses = []


def reward_from_events(events: list[str]) -> float:
    """Return the task-1 reward for one transition."""
    reward = STEP_REWARD

    for event in events:
        reward += EVENT_REWARDS.get(event, 0.0)

    return reward


def _after_transition(self) -> None:
    """Advance schedules and perform at most one optimization step."""
    self.environment_steps += 1
    self.epsilon = _epsilon_for_step(self.environment_steps)

    if len(self.replay_buffer) < MIN_REPLAY_SIZE:
        return

    transitions = self.replay_buffer.sample(BATCH_SIZE)

    loss = optimize_dqn(
        online_network=self.online_network,
        target_network=self.target_network,
        optimizer=self.optimizer,
        transitions=transitions,
        gamma=GAMMA,
        device=self.device,
    )

    self.episode_losses.append(loss)
    self.optimization_steps += 1

    if self.optimization_steps % TARGET_UPDATE_INTERVAL == 0:
        update_target_network(
            self.online_network,
            self.target_network,
        )

        self.logger.debug(
            "Updated target network after %d optimization steps.",
            self.optimization_steps,
        )


def _epsilon_for_step(environment_step: int) -> float:
    """Linearly decay epsilon from its start to its final value."""
    progress = min(
        environment_step / EPSILON_DECAY_STEPS,
        1.0,
    )

    return (
        EPSILON_START
        + progress * (EPSILON_END - EPSILON_START)
    )


