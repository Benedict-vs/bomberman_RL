"""Training callbacks for the task-2 DQN agent."""

from __future__ import annotations

from collections import deque

import events as e
import torch

from .augmentation import randomly_transform_transition
from .callbacks import MODEL_FILE, TRAINING_SEED
from .dqn import (
    optimize_dqn,
    soft_update_target_network,
    update_target_network,
)
from .features import DANGER_CHANNEL, state_to_features
from .model import ACTIONS, CoinCollectorDQN
from .replay_buffer import ReplayBuffer

try:
    from tools.trainlog import TrainLogger
except ImportError:
    TrainLogger = None


# DQN hyperparameters
GAMMA = 0.99
LEARNING_RATE = 1e-4

REPLAY_CAPACITY = 200_000
BATCH_SIZE = 64
MIN_REPLAY_SIZE = 5_000

# Soft target-network update
SOFT_TARGET_TAU = 1e-4

# Linear epsilon schedule, measured in environment transitions
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 100_000

# Reward design for task 2
STEP_REWARD = -0.05

EVENT_REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.CRATE_DESTROYED: 0.2,
    e.INVALID_ACTION: -1.0,
    e.GOT_KILLED: -5.0,
    # A suicide emits GOT_KILLED and KILLED_SELF. Keep the latter at
    # zero so that every death receives the penalty exactly once.
    e.KILLED_SELF: 0.0,
}

# Potential-based reward shaping
POTENTIAL_REWARD_SCALE = 0.0
POTENTIAL_DISTANCE_NORMALIZER = 32.0
SAFETY_POTENTIAL_SCALE = 1.0
MAX_EPISODE_STEPS = 400

RUN_LABEL = "task2_safety_potential_v1_15000ep_seed11"
TRAINLOG_OUT_DIR = "results/train/ben_task2"
CHECKPOINT_DIR = "../../results/train/ben_task2"

CHECKPOINT_INTERVAL = 100


def setup_training(self) -> None:
    """Initialize all state required only during training."""
    if torch.backends.mps.is_available():
        self.device = torch.device("mps")
    else:
        self.device = torch.device("cpu")

    self.logger.info(
        "Training DQN on device %s.",
        self.device,
    )

    self.online_network.to(self.device)
    self.online_network.train()

    self.target_network = CoinCollectorDQN().to(
        self.device
    )

    # Both networks start with identical parameters.
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
            agent="ben_task2",
            run=RUN_LABEL,
            out_dir=TRAINLOG_OUT_DIR,
            hyperparams={
                "gamma": GAMMA,
                "learning_rate": LEARNING_RATE,
                "replay_capacity": REPLAY_CAPACITY,
                "batch_size": BATCH_SIZE,
                "min_replay_size": MIN_REPLAY_SIZE,
                "soft_target_tau": SOFT_TARGET_TAU,
                "epsilon_start": EPSILON_START,
                "epsilon_end": EPSILON_END,
                "epsilon_decay_steps": (
                    EPSILON_DECAY_STEPS
                ),
                "step_reward": STEP_REWARD,
                "coin_collected_reward": EVENT_REWARDS[
                    e.COIN_COLLECTED
                ],
                "crate_destroyed_reward": EVENT_REWARDS[
                    e.CRATE_DESTROYED
                ],
                "invalid_action_reward": EVENT_REWARDS[
                    e.INVALID_ACTION
                ],
                "got_killed_reward": EVENT_REWARDS[
                    e.GOT_KILLED
                ],
                "killed_self_reward": EVENT_REWARDS[
                    e.KILLED_SELF
                ],
                "potential_reward_scale": (
                    POTENTIAL_REWARD_SCALE
                ),
                "potential_distance_normalizer": (
                    POTENTIAL_DISTANCE_NORMALIZER
                ),
                "safety_potential_scale": SAFETY_POTENTIAL_SCALE,
                "symmetry_augmentation": True,
                "legal_action_mask": True,
                "training_seed": TRAINING_SEED,
                "device": str(self.device),
            },
            extra_columns=[
                "mean_loss",
                "buffer_size",
            ],
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

    reward += potential_shaping_reward(
        old_game_state,
        new_game_state,
    )
    reward += safety_potential_shaping_reward(
        old_game_state,
        new_game_state,
    )

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
    if e.GOT_KILLED in events:
        # The framework skips game_events_occurred() once an agent is
        # dead. Store the otherwise missing transition of the lethal
        # final action here, including its terminal events and reward.
        reward = reward_from_events(events)
        reward += potential_shaping_reward(
            last_game_state,
            None,
        )
        reward += safety_potential_shaping_reward(
            last_game_state,
            None,
        )

        self.replay_buffer.append(
            state=state_to_features(last_game_state),
            action=ACTIONS.index(last_action),
            reward=reward,
            next_state=None,
            done=True,
        )

        self.episode_reward += reward
        self.episode_events.extend(events)
        _after_transition(self)
    else:
        # For a surviving agent, game_events_occurred() already stored
        # the final action. Only its terminal marker is still missing.
        self.replay_buffer.mark_last_terminal()

    if e.SURVIVED_ROUND in events:
        self.episode_events.append(
            e.SURVIVED_ROUND
        )

    cpu_state_dict = {
        name: parameter.detach().cpu()
        for name, parameter
        in self.online_network.state_dict().items()
    }

    torch.save(
        cpu_state_dict,
        MODEL_FILE,
    )

    episode = last_game_state["round"]

    if episode % CHECKPOINT_INTERVAL == 0:
        checkpoint_file = (
            f"{CHECKPOINT_DIR}/{RUN_LABEL}"
            f"__episode_{episode}.pt"
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
            sum(self.episode_losses)
            / len(self.episode_losses)
            if self.episode_losses
            else float("nan")
        )

        self.trainlog.log_episode(
            episode=episode,
            score=self.episode_events.count(
                e.COIN_COLLECTED
            ),
            steps=last_game_state["step"],
            events=self.episode_events,
            reward=self.episode_reward,
            epsilon=self.epsilon,
            extra={
                "mean_loss": mean_loss,
                "buffer_size": len(
                    self.replay_buffer
                ),
            },
        )

    self.logger.info(
        "Episode %d finished: reward %.3f, epsilon %.4f, "
        "buffer %d, mean loss %s.",
        episode,
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


def reward_from_events(
    events: list[str],
) -> float:
    """Return the task-2 event reward for one transition."""
    reward = STEP_REWARD

    for event in events:
        reward += EVENT_REWARDS.get(
            event,
            0.0,
        )

    return reward


def potential_shaping_reward(
    old_game_state: dict | None,
    new_game_state: dict | None,
) -> float:
    """Return potential-based shaping for one transition."""
    old_potential = _coin_potential(
        old_game_state
    )
    new_potential = _coin_potential(
        new_game_state
    )

    return POTENTIAL_REWARD_SCALE * (
        GAMMA * new_potential
        - old_potential
    )


def safety_potential_shaping_reward(
    old_game_state: dict | None,
    new_game_state: dict | None,
) -> float:
    """Reward transitions toward safety without prescribing an action."""
    old_potential = _safety_potential(old_game_state)
    new_potential = _safety_potential(new_game_state)

    return SAFETY_POTENTIAL_SCALE * (
        GAMMA * new_potential
        - old_potential
    )


def _safety_potential(game_state: dict | None) -> float:
    """Return one minus the bomb danger at the agent's current tile."""
    if game_state is None:
        return 0.0

    features = state_to_features(game_state)
    self_x, self_y = game_state["self"][3]
    danger = float(features[DANGER_CHANNEL, self_y, self_x])
    return 1.0 - danger


def _coin_potential(
    game_state: dict | None,
) -> float:
    """Return normalized negative distance to the nearest coin."""
    if game_state is None:
        return 0.0

    if (
        game_state.get("step", 0)
        >= MAX_EPISODE_STEPS
    ):
        return 0.0

    if not game_state.get("coins"):
        return 0.0

    distance = _shortest_coin_distance(
        game_state
    )

    if distance is None:
        return 0.0

    return (
        -float(distance)
        / POTENTIAL_DISTANCE_NORMALIZER
    )


def _shortest_coin_distance(
    game_state: dict,
) -> int | None:
    """Find the shortest walkable distance to any visible coin."""
    field = game_state["field"]
    start = tuple(game_state["self"][3])

    coins = {
        tuple(position)
        for position in game_state["coins"]
    }

    if start in coins:
        return 0

    width, height = field.shape

    queue = deque(
        [(start[0], start[1], 0)]
    )
    visited = {start}

    directions = (
        (0, -1),
        (1, 0),
        (0, 1),
        (-1, 0),
    )

    while queue:
        x, y, distance = queue.popleft()

        for delta_x, delta_y in directions:
            next_x = x + delta_x
            next_y = y + delta_y

            next_position = (
                next_x,
                next_y,
            )

            if not (
                0 <= next_x < width
                and 0 <= next_y < height
            ):
                continue

            if next_position in visited:
                continue

            if field[next_x, next_y] != 0:
                continue

            if next_position in coins:
                return distance + 1

            visited.add(next_position)

            queue.append(
                (
                    next_x,
                    next_y,
                    distance + 1,
                )
            )

    return None


def _after_transition(self) -> None:
    """Advance schedules and perform at most one optimization step."""
    self.environment_steps += 1

    self.epsilon = _epsilon_for_step(
        self.environment_steps
    )

    if len(self.replay_buffer) < MIN_REPLAY_SIZE:
        return

    sampled_transitions = self.replay_buffer.sample(
        BATCH_SIZE
    )

    transitions = [
        randomly_transform_transition(transition)
        for transition in sampled_transitions
    ]

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

    # Move the target network a very small amount toward
    # the online network after every optimization step.
    soft_update_target_network(
        online_network=self.online_network,
        target_network=self.target_network,
        tau=SOFT_TARGET_TAU,
    )


def _epsilon_for_step(
    environment_step: int,
) -> float:
    """Linearly decay epsilon from its start to its final value."""
    progress = min(
        environment_step
        / EPSILON_DECAY_STEPS,
        1.0,
    )

    return (
        EPSILON_START
        + progress
        * (EPSILON_END - EPSILON_START)
    )
