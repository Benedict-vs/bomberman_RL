"""Training callbacks for Ben's isolated Task-4 DQN development agent."""

from __future__ import annotations

from collections import deque
import os
import random
import tempfile

import events as e
import numpy as np
import torch

from .augmentation import all_symmetry_transforms, randomly_transform_transition
from .callbacks import (
    CONTINUE_COIN_REWARD15,
    CRATE_WAIT_PENALTY,
    ESCAPE_FEATURE_MODE,
    EXPERIMENT_STEM,
    FINE_TUNE_COIN_POTENTIAL,
    FINE_TUNE_COIN_REWARD,
    FINE_TUNE_CRATE_WAIT,
    LOAD_MODEL_FILE,
    MODEL_FILE,
    MULTISEED_ARM,
    INPUT_CHANNELS,
    OPPONENT_ALIGNMENT_MODE,
    OPPONENT_ESCAPE_PRESSURE_MODE,
    OPPONENT_BOMB_READY_MODE,
    OPPONENT_BOMB_TRADEOFF_MODE,
    COIN_NAVIGATION_MODE,
    SPACETIME_SAFETY_MODE,
    TEMPORAL_SAFETY_MODE,
    TOTAL_EPISODES,
    TRAINING_ARM,
    TRAINING_SEED,
    DUELING_DQN,
    NETWORK_CLASS,
    OPPONENT_PREDICTION,
    BOMB_OUTCOME_AUXILIARY,
    FULL_SYMMETRY_AUGMENTATION,
)
from .dqn import (
    optimize_dqn,
    soft_update_target_network,
    update_target_network,
)
from .features import (
    DANGER_CHANNEL,
    ESCAPE_TILES_CHANNEL,
    VISIT_COUNT_NORMALIZER,
    VISIT_COUNT_MAX,
    _blast_coordinates,
    state_to_features,
)
from .model import ACTIONS
from .replay_buffer import BombOutcomeBuffer, ReplayBuffer, Transition

try:
    from tools.trainlog import TrainLogger
except ImportError:
    TrainLogger = None


# DQN hyperparameters
GAMMA = 0.99
LEARNING_RATE = 5e-5 if TRAINING_ARM == "learning_rate5e5_v1" else 1e-4

REPLAY_CAPACITY = 200_000
BATCH_SIZE = 64
MIN_REPLAY_SIZE = 5_000

# Soft target-network update
SOFT_TARGET_TAU = 1e-4
N_STEP_RETURN = 3 if TRAINING_ARM == "nstep3_v1" else 1
DOUBLE_DQN = TRAINING_ARM == "double_dqn_v1"
PRIORITIZED_REPLAY = TRAINING_ARM == "per_v1"
PER_BETA_START = 0.4
PER_BETA_ANNEAL_STEPS = 100_000

# Linear epsilon schedule, measured in environment transitions
EPSILON_START = (
    0.15
    if TRAINING_ARM
    in {"exploration_eps15_v1", "exploration_eps15_long3000_v1"}
    else 0.05
    if FINE_TUNE_CRATE_WAIT
    else 1.0
)
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 100_000

# Task-4 reward design. GOT_KILLED applies to every death; KILLED_SELF is an
# additional event only for an own-bomb death, so this arm prices suicides at
# -7 and opponent-caused deaths at -5.
STEP_REWARD = -0.05
EVENT_REPLAY_MODE = (
    "balanced"
    if TRAINING_ARM == "event_replay_balanced_v1"
    else "uniform"
)

EVENT_REWARDS = {
    e.COIN_COLLECTED: 1.5 if FINE_TUNE_COIN_REWARD else 1.0,
    e.CRATE_DESTROYED: 0.3,
    e.KILLED_OPPONENT: (
        7.5
        if TRAINING_ARM.startswith("kill_reward75_")
        else 5.0
    ),
    e.INVALID_ACTION: -1.0,
    e.GOT_KILLED: (
        -10.0
        if TRAINING_ARM == "survival_penalty10_v1"
        else -5.0
    ),
    e.KILLED_SELF: (
        -2.0 if TRAINING_ARM == "rule_based_suicide7_v1" else 0.0
    ),
}

# Potential-based reward shaping
POTENTIAL_REWARD_SCALE = 1.0 if FINE_TUNE_COIN_POTENTIAL else 0.0
POTENTIAL_DISTANCE_NORMALIZER = 32.0
OPPONENT_POTENTIAL_SCALE = 0.0
SAFETY_POTENTIAL_SCALE = 1.0
TERMINAL_SAFETY_POTENTIAL_ZERO = TRAINING_ARM == "terminal_safety_v1"
MAX_EPISODE_STEPS = 400
SAFE_COIN_WAIT_PENALTY = 0.0
SAFE_CRATE_WAIT_PENALTY = (
    CRATE_WAIT_PENALTY if FINE_TUNE_CRATE_WAIT else 0.0
)
SAFE_OFFENSE_BOMB_REWARD = (
    0.2
    if TRAINING_ARM == "safe_offense_reward02_v1"
    else 0.1
    if TRAINING_ARM == "safe_offense_reward01_v1"
    else 0.0
)

VISIT_COUNT_ENABLED = True
VISIT_COUNT_ENCODING = "linear_10"
RUN_LABEL = (
    f"task4_{EXPERIMENT_STEM}_{TOTAL_EPISODES}ep_seed{TRAINING_SEED}"
)
TRAINLOG_OUT_DIR = "results/train/ben_task4"
CHECKPOINT_DIR = "../../results/train/ben_task4"

CHECKPOINT_INTERVAL = 100
RESUME_STATE_FILE = os.environ.get("BM_TASK4_RESUME_STATE_FILE")
if RESUME_STATE_FILE is not None and os.path.isabs(RESUME_STATE_FILE):
    raise ValueError("BM_TASK4_RESUME_STATE_FILE must be relative.")
SAVE_TRAINING_STATE = os.environ.get("BM_TASK4_SAVE_TRAINING_STATE", "0") == "1"
TRAINING_STATE_FILE = (
    f"{CHECKPOINT_DIR}/{RUN_LABEL}__training_state.pt"
)
TRAINING_OPPONENTS = (
    "rule_based_agent,rule_based_agent,rule_based_agent"
    if TRAINING_ARM
    in {
        "rule_based_suicide7_v1",
        "rule_based_continue_v1",
        "rule_based_continue_control1000_v1",
        "mixed_kill_consolidate1000_v1",
        "terminal_safety_control_v1",
        "terminal_safety_v1",
        "opponent_alignment_zero_v1",
        "opponent_alignment_v1",
        "safe_offense_zero_v1",
        "safe_offense_reward01_v1",
        "safe_offense_reward02_v1",
        "opponent_escape_pressure_zero_v1",
        "opponent_escape_pressure_v1",
        "external_trio_control_v1",
        "opponent_bomb_ready_zero_v1",
        "opponent_bomb_ready_v1",
        "exploration_eps05_v1",
        "exploration_eps15_v1",
        "exploration_eps05_long3000_v1",
        "exploration_eps15_long3000_v1",
    }
    else (
        "peaceful_agent,rule_based_agent,rule_based_agent"
        if TRAINING_ARM
        in {
            "mixed_kill_v1",
            "mixed_kill_seed12_v1",
            "mixed_kill_seed13_v1",
            "mixed_kill_continue7000_v1",
            "opponent_alignment_mixed_zero_v1",
            "opponent_alignment_mixed_v1",
            "temporal_safety_zero_v1",
                "temporal_safety_v1",
                "spacetime_safety_zero_v1",
                "spacetime_safety_v1",
                "coin_navigation_zero_v1",
                "coin_navigation_v1",
                "nstep1_control_v1",
                "nstep3_v1",
                "opponent_bomb_tradeoff_zero_v1",
        "opponent_bomb_tradeoff_v1",
                "mixed_external_control_v1",
                "survival_penalty_control_v1",
                "survival_penalty10_v1",
                "curriculum_control_p1_v1",
                "curriculum_control_p2_v1",
                "curriculum_control_p3_v1",
                "curriculum_control_p4_v1",
                "curriculum_candidate_p1_v1",
                "curriculum_candidate_p2_v1",
                "curriculum_candidate_p3_v1",
                "curriculum_candidate_p4_v1",
                "kill_reward_control_p1_v1",
                "kill_reward_control_p2_v1",
                "kill_reward_control_p3_v1",
                "kill_reward_control_p4_v1",
                "kill_reward75_p1_v1",
                "kill_reward75_p2_v1",
                "kill_reward75_p3_v1",
                "kill_reward75_p4_v1",
                "event_replay_control_v1",
                "event_replay_balanced_v1",
                "learning_rate_control_v1",
                "learning_rate5e5_v1",
                "double_dqn_control_v1",
                "double_dqn_v1",
                "per_control_v1",
                "per_v1",
                "dueling_control_v1",
                "dueling_v1",
                "opponent_prediction_control_v1",
                "opponent_prediction_v1",
        }
        else (
            "ext_lijesse_featureeverything,ext_xiaoxiae_bindist_v2,"
            "ext_xiaoxiae_binary_v6"
            if TRAINING_ARM in {"external_trio_v1", "mixed_external_v1"}
            else "configured_by_main_cli"
        )
    )
)
TRAINING_OPPONENTS = os.environ.get(
    "BM_TASK4_TRAINING_OPPONENTS",
    TRAINING_OPPONENTS,
)


def setup_training(self) -> None:
    """Initialize all state required only during training."""
    self.visit_count_encoding = VISIT_COUNT_ENCODING
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

    self.target_network = NETWORK_CLASS(input_channels=INPUT_CHANNELS).to(
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
        event_balance=EVENT_REPLAY_MODE == "balanced",
        prioritized=PRIORITIZED_REPLAY,
    )
    self.n_step_queue = deque()
    self.bomb_outcome_buffer = BombOutcomeBuffer()
    self.pending_bomb_outcome = None

    self.environment_steps = 0
    self.optimization_steps = 0
    self.epsilon = EPSILON_START
    self.episode_offset = 0

    self.episode_reward = 0.0
    self.episode_events = []
    self.episode_losses = []
    self.safe_offense_bombs = 0

    self.trainlog = None

    if RESUME_STATE_FILE is not None:
        _restore_training_state(self, RESUME_STATE_FILE)

    if TrainLogger is not None:
        self.trainlog = TrainLogger(
            agent="ben_task4",
            run=RUN_LABEL,
            out_dir=TRAINLOG_OUT_DIR,
            hyperparams={
                "gamma": GAMMA,
                "learning_rate": LEARNING_RATE,
                "dueling_dqn": DUELING_DQN,
                "auxiliary_opponent_prediction": OPPONENT_PREDICTION,
                "replay_capacity": REPLAY_CAPACITY,
                "batch_size": BATCH_SIZE,
                "min_replay_size": MIN_REPLAY_SIZE,
                "soft_target_tau": SOFT_TARGET_TAU,
                "n_step_return": N_STEP_RETURN,
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
                "killed_opponent_reward": EVENT_REWARDS[
                    e.KILLED_OPPONENT
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
                "potential_target": (
                    "nearest_reachable_visible_coin"
                    if FINE_TUNE_COIN_POTENTIAL
                    else "disabled"
                ),
                "safe_coin_wait_penalty": SAFE_COIN_WAIT_PENALTY,
                "safe_crate_wait_penalty": SAFE_CRATE_WAIT_PENALTY,
                "safe_offense_bomb_reward": SAFE_OFFENSE_BOMB_REWARD,
                "potential_distance_normalizer": (
                    POTENTIAL_DISTANCE_NORMALIZER
                ),
                "opponent_potential_scale": OPPONENT_POTENTIAL_SCALE,
                "opponent_alignment_mode": OPPONENT_ALIGNMENT_MODE,
                "opponent_escape_pressure_mode": (
                    OPPONENT_ESCAPE_PRESSURE_MODE
                ),
                "opponent_bomb_ready_mode": OPPONENT_BOMB_READY_MODE,
                "opponent_bomb_tradeoff_mode": OPPONENT_BOMB_TRADEOFF_MODE,
                "spacetime_safety_mode": SPACETIME_SAFETY_MODE,
                "coin_navigation_mode": COIN_NAVIGATION_MODE,
                "temporal_safety_mode": TEMPORAL_SAFETY_MODE,
                "safety_potential_scale": SAFETY_POTENTIAL_SCALE,
                "terminal_safety_potential_zero": (
                    TERMINAL_SAFETY_POTENTIAL_ZERO
                ),
                "symmetry_augmentation": (
                    "all_d4"
                    if FULL_SYMMETRY_AUGMENTATION
                    else "random_one"
                ),
                "legal_action_mask": True,
                "visit_count_channel": VISIT_COUNT_ENABLED,
                "visit_count_encoding": VISIT_COUNT_ENCODING,
                "visit_count_max": VISIT_COUNT_MAX,
                "visit_count_normalizer": VISIT_COUNT_NORMALIZER,
                "training_seed": TRAINING_SEED,
                "training_arm": TRAINING_ARM,
                "training_opponents": TRAINING_OPPONENTS,
                "planned_total_episodes": TOTAL_EPISODES,
                "model_file": MODEL_FILE,
                "load_model_file": LOAD_MODEL_FILE,
                "resume_state_file": RESUME_STATE_FILE,
                "save_training_state": SAVE_TRAINING_STATE,
                "training_state_file": TRAINING_STATE_FILE,
                "episode_offset": self.episode_offset,
                "fine_tune_crate_wait": FINE_TUNE_CRATE_WAIT,
                "fine_tune_coin_potential": (
                    FINE_TUNE_COIN_POTENTIAL
                ),
                "fine_tune_coin_reward": FINE_TUNE_COIN_REWARD,
                "continue_coin_reward15": CONTINUE_COIN_REWARD15,
                "input_channels": INPUT_CHANNELS,
                "escape_feature_mode": ESCAPE_FEATURE_MODE,
                "multiseed_arm": MULTISEED_ARM,
                "device": str(self.device),
            },
            extra_columns=[
                "mean_loss",
                "buffer_size",
                "safe_offense_bombs",
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

    _observe_pending_bomb(self, new_game_state, events)

    reward = reward_from_events(events)

    reward += potential_shaping_reward(
        old_game_state,
        new_game_state,
    )
    reward += safety_potential_shaping_reward(
        old_game_state,
        new_game_state,
    )
    reward += opponent_potential_shaping_reward(
        old_game_state,
        new_game_state,
    )
    reward += safe_coin_wait_penalty(
        old_game_state,
        self_action,
    )
    reward += safe_crate_wait_penalty(
        old_game_state,
        self_action,
    )
    safe_offense_reward = safe_offense_bomb_reward(
        old_game_state,
        self_action,
    )
    reward += safe_offense_reward
    if safe_offense_reward > 0.0:
        self.safe_offense_bombs += 1

    state = _features_for_stored_action(self, old_game_state)
    next_state = _next_features_with_visit_count(self, new_game_state)
    action_index = ACTIONS.index(self_action)

    _append_n_step_transition(
        self,
        state=state,
        action=action_index,
        reward=reward,
        next_state=next_state,
        done=False,
        event_tags=_replay_tags(events),
    )

    if e.BOMB_DROPPED in events:
        self.pending_bomb_outcome = {
            "position": tuple(old_game_state["self"][3]),
            "state": state.copy(),
            "kill": False,
            "self_death": False,
        }

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
    _observe_pending_bomb(self, None, events, final_round=True)
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
        reward += opponent_potential_shaping_reward(
            last_game_state,
            None,
        )
        reward += safe_coin_wait_penalty(
            last_game_state,
            last_action,
        )
        reward += safe_crate_wait_penalty(
            last_game_state,
            last_action,
        )
        safe_offense_reward = safe_offense_bomb_reward(
            last_game_state,
            last_action,
        )
        reward += safe_offense_reward
        if safe_offense_reward > 0.0:
            self.safe_offense_bombs += 1

        _append_n_step_transition(
            self,
            state=_features_for_stored_action(self, last_game_state),
            action=ACTIONS.index(last_action),
            reward=reward,
            next_state=None,
            done=True,
            event_tags=_replay_tags(events),
        )

        self.episode_reward += reward
        self.episode_events.extend(events)
        _after_transition(self)
    else:
        # For a surviving agent, game_events_occurred() already stored
        # the final action. Only its terminal marker is still missing.
        terminal_safety_adjustment = _terminal_safety_shaping_adjustment(
            last_game_state
        )
        if N_STEP_RETURN == 1:
            self.replay_buffer.mark_last_terminal(
                reward_adjustment=terminal_safety_adjustment
            )
        else:
            _finish_n_step_episode(
                self,
                terminal_reward_adjustment=terminal_safety_adjustment,
            )

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

    episode = self.episode_offset + last_game_state["round"]

    if episode % CHECKPOINT_INTERVAL == 0:
        checkpoint_file = (
            f"{CHECKPOINT_DIR}/{RUN_LABEL}"
            f"__episode_{episode}.pt"
        )

        torch.save(
            cpu_state_dict,
            checkpoint_file,
        )
        if SAVE_TRAINING_STATE:
            _save_training_state(self, episode)

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
            score=_official_score(self.episode_events),
            steps=last_game_state["step"],
            events=self.episode_events,
            reward=self.episode_reward,
            epsilon=self.epsilon,
            extra={
                "mean_loss": mean_loss,
                "buffer_size": len(
                    self.replay_buffer
                ),
                "safe_offense_bombs": self.safe_offense_bombs,
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
    self.safe_offense_bombs = 0


def _observe_pending_bomb(
    self,
    new_game_state: dict | None,
    events: list[str],
    final_round: bool = False,
) -> None:
    """Resolve one own bomb only once its position has disappeared."""
    pending = getattr(self, "pending_bomb_outcome", None)
    if pending is None:
        return
    pending["kill"] = pending["kill"] or e.KILLED_OPPONENT in events
    pending["self_death"] = pending["self_death"] or e.KILLED_SELF in events
    resolved = pending["self_death"]
    if new_game_state is not None:
        bomb_positions = {
            tuple(position) for position, _timer in new_game_state.get("bombs", [])
        }
        resolved = resolved or pending["position"] not in bomb_positions
    elif final_round:
        # A kill/self-death event can only be emitted after an explosion. A
        # timeout with an active bomb remains intentionally unlabeled.
        resolved = resolved or pending["kill"]
    if not resolved:
        return
    outcome = 2 if pending["self_death"] else 1 if pending["kill"] else 0
    self.bomb_outcome_buffer.append(pending["state"], outcome)
    self.pending_bomb_outcome = None


def _save_training_state(self, episode: int) -> None:
    """Atomically save all local state needed to resume after a checkpoint.

    The resulting file is for local training only and may contain Python and
    NumPy objects. It is intentionally separate from the CPU-only inference
    model saved as ``MODEL_FILE``.
    """
    state = {
        "format_version": 1,
        "training_arm": TRAINING_ARM,
        "training_seed": TRAINING_SEED,
        "input_channels": INPUT_CHANNELS,
        "network_class": NETWORK_CLASS.__name__,
        "episode": int(episode),
        "online_network": _cpu_state_dict(self.online_network),
        "target_network": _cpu_state_dict(self.target_network),
        "optimizer": self.optimizer.state_dict(),
        "replay_buffer": self.replay_buffer.state_dict(),
        "n_step_queue": list(self.n_step_queue),
        "environment_steps": int(self.environment_steps),
        "optimization_steps": int(self.optimization_steps),
        "epsilon": float(self.epsilon),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    destination = os.path.join(os.path.dirname(__file__), TRAINING_STATE_FILE)
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as temporary:
        temporary_name = temporary.name
    try:
        torch.save(state, temporary_name)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _restore_training_state(self, relative_path: str) -> None:
    """Restore a trusted local training state after checking its identity."""
    path = os.path.join(os.path.dirname(__file__), relative_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Task-4 training state not found: {relative_path}")
    # The state includes Torch's process-global RNG as a CPU ByteTensor.  Do
    # not map the whole checkpoint to MPS/CUDA: torch.set_rng_state accepts
    # only a CPU tensor. Network/optimizer state is moved as needed by their
    # respective load_state_dict calls below.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("format_version") != 1:
        raise ValueError("Unsupported Task-4 training-state format.")
    expected = {
        "training_arm": TRAINING_ARM,
        "training_seed": TRAINING_SEED,
        "input_channels": INPUT_CHANNELS,
        "network_class": NETWORK_CLASS.__name__,
    }
    for name, value in expected.items():
        if state.get(name) != value:
            raise ValueError(
                f"Training-state {name}={state.get(name)!r} does not match {value!r}."
            )
    self.online_network.load_state_dict(state["online_network"])
    self.target_network.load_state_dict(state["target_network"])
    self.optimizer.load_state_dict(state["optimizer"])
    replay_buffer = ReplayBuffer.from_state_dict(state["replay_buffer"])
    if (
        replay_buffer.capacity != REPLAY_CAPACITY
        or replay_buffer.event_balance != (EVENT_REPLAY_MODE == "balanced")
        or replay_buffer.prioritized != PRIORITIZED_REPLAY
    ):
        raise ValueError("Training-state replay-buffer configuration does not match.")
    self.replay_buffer = replay_buffer
    self.n_step_queue = deque(state["n_step_queue"])
    self.episode_offset = int(state["episode"])
    self.environment_steps = int(state["environment_steps"])
    self.optimization_steps = int(state["optimization_steps"])
    self.epsilon = float(state["epsilon"])
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"])
    self.logger.info(
        "Resumed Task-4 training state from %s after episode %d.",
        relative_path,
        self.episode_offset,
    )


def _cpu_state_dict(network: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy network parameters to portable CPU tensors for local checkpoints."""
    return {
        name: parameter.detach().cpu()
        for name, parameter in network.state_dict().items()
    }


def reward_from_events(
    events: list[str],
) -> float:
    """Return the inherited baseline reward for one transition."""
    reward = STEP_REWARD

    for event in events:
        reward += EVENT_REWARDS.get(
            event,
            0.0,
        )

    return reward


def _replay_tags(events: list[str]) -> frozenset[str]:
    """Label rare outcome transitions for the optional balanced replay arm."""
    tags: set[str] = set()
    if e.KILLED_OPPONENT in events:
        tags.add("kill")
    if e.GOT_KILLED in events:
        tags.add("death")
    return frozenset(tags)


def _official_score(events: list[str]) -> int:
    """Return framework score: one per coin and five per kill."""
    return (
        events.count(e.COIN_COLLECTED)
        + 5 * events.count(e.KILLED_OPPONENT)
    )


def _features_for_stored_action(self, game_state: dict) -> np.ndarray:
    """Reuse exactly the features on which the stored action was selected."""
    expected = (game_state.get("round"), game_state.get("step"))
    if (
        getattr(self, "last_action_features", None) is not None
        and getattr(self, "last_feature_round_step", None) == expected
    ):
        return self.last_action_features
    return state_to_features(_state_with_escape_mode(game_state))


def _next_features_with_visit_count(self, game_state: dict) -> np.ndarray:
    """Preview the visit count that act() will apply to the next state."""
    if not VISIT_COUNT_ENABLED:
        return state_to_features(_state_with_escape_mode(game_state))

    visit_counts = getattr(self, "visit_counts", None)
    if visit_counts is None:
        return state_to_features(_state_with_escape_mode(game_state))

    next_counts = visit_counts.copy()
    self_x, self_y = game_state["self"][3]
    next_counts[self_x, self_y] += 1.0
    augmented_state = dict(game_state)
    augmented_state["visit_counts"] = next_counts
    augmented_state["visit_count_encoding"] = VISIT_COUNT_ENCODING
    augmented_state["escape_feature_mode"] = ESCAPE_FEATURE_MODE
    augmented_state["opponent_alignment_mode"] = OPPONENT_ALIGNMENT_MODE
    augmented_state["opponent_escape_pressure_mode"] = (
        OPPONENT_ESCAPE_PRESSURE_MODE
    )
    augmented_state["opponent_bomb_ready_mode"] = OPPONENT_BOMB_READY_MODE
    augmented_state["temporal_safety_mode"] = TEMPORAL_SAFETY_MODE
    augmented_state["opponent_bomb_tradeoff_mode"] = (
        OPPONENT_BOMB_TRADEOFF_MODE
    )
    augmented_state["spacetime_safety_mode"] = SPACETIME_SAFETY_MODE
    augmented_state["coin_navigation_mode"] = COIN_NAVIGATION_MODE
    return state_to_features(augmented_state)


def _state_with_escape_mode(game_state: dict) -> dict:
    """Return a shallow state copy configured for the current feature arm."""
    augmented_state = dict(game_state)
    augmented_state["escape_feature_mode"] = ESCAPE_FEATURE_MODE
    augmented_state["opponent_alignment_mode"] = OPPONENT_ALIGNMENT_MODE
    augmented_state["opponent_escape_pressure_mode"] = (
        OPPONENT_ESCAPE_PRESSURE_MODE
    )
    augmented_state["opponent_bomb_ready_mode"] = OPPONENT_BOMB_READY_MODE
    augmented_state["temporal_safety_mode"] = TEMPORAL_SAFETY_MODE
    augmented_state["opponent_bomb_tradeoff_mode"] = (
        OPPONENT_BOMB_TRADEOFF_MODE
    )
    augmented_state["coin_navigation_mode"] = COIN_NAVIGATION_MODE
    return augmented_state


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


def _terminal_safety_shaping_adjustment(
    last_game_state: dict | None,
) -> float:
    """Remove the non-terminal safety endpoint after a survived round.

    The framework reports survival only after storing the final action, so
    the candidate arm removes the ``gamma * Phi(s')`` term retroactively.
    The control preserves the historical reward unchanged.
    """
    if not TERMINAL_SAFETY_POTENTIAL_ZERO:
        return 0.0
    return -SAFETY_POTENTIAL_SCALE * GAMMA * _safety_potential(
        last_game_state
    )


def opponent_potential_shaping_reward(
    old_game_state: dict | None,
    new_game_state: dict | None,
) -> float:
    """Shape transitions by reachable distance to the nearest opponent."""
    old_potential = _opponent_potential(old_game_state)
    new_potential = _opponent_potential(new_game_state)

    return OPPONENT_POTENTIAL_SCALE * (
        GAMMA * new_potential
        - old_potential
    )


def safe_coin_wait_penalty(
    game_state: dict | None,
    action: str,
) -> float:
    """Penalize WAIT only beside a reachable goal and outside danger."""
    if game_state is None or action != "WAIT":
        return 0.0

    coin_distance = _shortest_coin_distance(game_state)
    if coin_distance is None or coin_distance <= 0:
        return 0.0

    features = state_to_features(game_state)
    self_x, self_y = game_state["self"][3]
    if features[DANGER_CHANNEL, self_y, self_x] > 0.0:
        return 0.0

    return SAFE_COIN_WAIT_PENALTY


def safe_crate_wait_penalty(
    game_state: dict | None,
    action: str,
) -> float:
    """Penalize safe WAIT beside a crate when a bomb has an escape."""
    if game_state is None or action != "WAIT":
        return 0.0

    self_x, self_y = game_state["self"][3]
    if not game_state["self"][2]:
        return 0.0

    field = game_state["field"]
    next_to_crate = any(
        0 <= self_x + delta_x < field.shape[0]
        and 0 <= self_y + delta_y < field.shape[1]
        and field[self_x + delta_x, self_y + delta_y] == 1
        for delta_x, delta_y in ((0, -1), (1, 0), (0, 1), (-1, 0))
    )
    if not next_to_crate:
        return 0.0

    augmented_state = dict(game_state)
    augmented_state["escape_feature_mode"] = "reachable_safe_tiles"
    features = state_to_features(augmented_state)
    if features[DANGER_CHANNEL, self_y, self_x] > 0.0:
        return 0.0
    if features[ESCAPE_TILES_CHANNEL].max() <= 0.0:
        return 0.0

    return SAFE_CRATE_WAIT_PENALTY


def safe_offense_bomb_reward(
    game_state: dict | None,
    action: str,
) -> float:
    """Reward a real bomb that threatens an opponent and has an escape."""
    if (
        game_state is None
        or action != "BOMB"
        or not game_state["self"][2]
        or SAFE_OFFENSE_BOMB_REWARD == 0.0
    ):
        return 0.0

    self_position = tuple(game_state["self"][3])
    opponents = {tuple(other[3]) for other in game_state.get("others", [])}
    if not opponents.intersection(
        _blast_coordinates(game_state["field"], self_position)
    ):
        return 0.0

    augmented_state = dict(game_state)
    augmented_state["escape_feature_mode"] = "reachable_safe_tiles"
    augmented_state["opponent_alignment_mode"] = "disabled"
    features = state_to_features(augmented_state)
    if features[ESCAPE_TILES_CHANNEL].max() <= 0.0:
        return 0.0

    return SAFE_OFFENSE_BOMB_REWARD


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


def _opponent_potential(game_state: dict | None) -> float:
    """Return normalized negative distance to a reachable opponent."""
    if game_state is None:
        return 0.0
    if game_state.get("step", 0) >= MAX_EPISODE_STEPS:
        return 0.0

    distance = _shortest_opponent_distance(game_state)
    if distance is None:
        return 0.0

    return -float(distance) / POTENTIAL_DISTANCE_NORMALIZER


def _navigation_potential(game_state: dict | None) -> float:
    """Prefer visible coins, otherwise approach a reachable crate."""
    if game_state is None:
        return 0.0

    if game_state.get("step", 0) >= MAX_EPISODE_STEPS:
        return 0.0

    coin_distance = _shortest_coin_distance(game_state)
    if coin_distance is not None:
        return -float(coin_distance) / POTENTIAL_DISTANCE_NORMALIZER

    crate_distance = _shortest_crate_approach_distance(game_state)
    if crate_distance is None:
        return 0.0

    return -float(crate_distance) / POTENTIAL_DISTANCE_NORMALIZER


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


def _shortest_opponent_distance(game_state: dict) -> int | None:
    """Find the shortest crate-free distance to any opponent tile."""
    field = game_state["field"]
    start = tuple(game_state["self"][3])
    opponents = {tuple(other[3]) for other in game_state.get("others", [])}
    if not opponents:
        return None
    if start in opponents:
        return 0

    width, height = field.shape
    queue = deque([(start[0], start[1], 0)])
    visited = {start}
    directions = ((0, -1), (1, 0), (0, 1), (-1, 0))

    while queue:
        x, y, distance = queue.popleft()
        for delta_x, delta_y in directions:
            next_x = x + delta_x
            next_y = y + delta_y
            next_position = (next_x, next_y)
            if not (0 <= next_x < width and 0 <= next_y < height):
                continue
            if next_position in visited:
                continue
            if next_position in opponents:
                return distance + 1
            if field[next_x, next_y] != 0:
                continue
            visited.add(next_position)
            queue.append((next_x, next_y, distance + 1))

    return None


def _shortest_crate_approach_distance(game_state: dict) -> int | None:
    """Find the nearest walkable tile adjacent to at least one crate."""
    field = game_state["field"]
    start = tuple(game_state["self"][3])
    width, height = field.shape
    directions = ((0, -1), (1, 0), (0, 1), (-1, 0))

    def next_to_crate(x: int, y: int) -> bool:
        return any(
            0 <= x + dx < width
            and 0 <= y + dy < height
            and field[x + dx, y + dy] == 1
            for dx, dy in directions
        )

    queue = deque([(start[0], start[1], 0)])
    visited = {start}

    while queue:
        x, y, distance = queue.popleft()
        if next_to_crate(x, y):
            return distance

        for delta_x, delta_y in directions:
            next_x = x + delta_x
            next_y = y + delta_y
            next_position = (next_x, next_y)

            if not (0 <= next_x < width and 0 <= next_y < height):
                continue
            if next_position in visited or field[next_x, next_y] != 0:
                continue

            visited.add(next_position)
            queue.append((next_x, next_y, distance + 1))

    return None


def _append_n_step_transition(
    self,
    state: np.ndarray,
    action: int,
    reward: float,
    next_state: np.ndarray | None,
    done: bool,
    event_tags: frozenset[str] = frozenset(),
) -> None:
    """Queue a raw step and emit every complete n-step replay item."""
    if next_state is None:
        next_state = np.zeros_like(state)
    self.n_step_queue.append(
        Transition(
            state=state.copy(),
            action=int(action),
            reward=float(reward),
            next_state=next_state.copy(),
            done=bool(done),
            event_tags=frozenset(event_tags),
        )
    )

    if done:
        while self.n_step_queue:
            _emit_n_step_transition(self)
    elif N_STEP_RETURN == 1 or len(self.n_step_queue) > N_STEP_RETURN:
        # Retain one complete window until a following step proves that its
        # endpoint is non-terminal. A surviving round is only declared done
        # after game_events_occurred() has processed its final action.
        _emit_n_step_transition(self)


def _emit_n_step_transition(self) -> None:
    """Aggregate the oldest pending transition over up to N steps."""
    window = []
    for transition in self.n_step_queue:
        window.append(transition)
        if len(window) >= N_STEP_RETURN or transition.done:
            break

    reward = sum(
        (GAMMA ** index) * transition.reward
        for index, transition in enumerate(window)
    )
    first = window[0]
    last = window[-1]
    self.replay_buffer.append(
        state=first.state,
        action=first.action,
        reward=reward,
        next_state=last.next_state,
        done=last.done,
        n_steps=len(window),
        event_tags=frozenset(
            tag for transition in window for tag in transition.event_tags
        ),
    )
    self.n_step_queue.popleft()


def _finish_n_step_episode(
    self,
    terminal_reward_adjustment: float = 0.0,
) -> None:
    """Mark the pending final raw step terminal and flush short returns."""
    if not self.n_step_queue:
        raise ValueError("Cannot finish an empty n-step queue.")
    last = self.n_step_queue[-1]
    self.n_step_queue[-1] = Transition(
        state=last.state,
        action=last.action,
        reward=last.reward + float(terminal_reward_adjustment),
        next_state=np.zeros_like(last.next_state),
        done=True,
        event_tags=last.event_tags,
    )
    while self.n_step_queue:
        _emit_n_step_transition(self)


def _after_transition(self) -> None:
    """Advance schedules and perform at most one optimization step."""
    self.environment_steps += 1

    self.epsilon = _epsilon_for_step(
        self.environment_steps
    )

    if len(self.replay_buffer) < MIN_REPLAY_SIZE:
        return

    if PRIORITIZED_REPLAY:
        beta = min(
            1.0,
            PER_BETA_START
            + (1.0 - PER_BETA_START)
            * self.optimization_steps
            / PER_BETA_ANNEAL_STEPS,
        )
        sampled_transitions, priority_indices, importance_weights = (
            self.replay_buffer.sample_prioritized(BATCH_SIZE, beta)
        )
    else:
        sampled_transitions = self.replay_buffer.sample(BATCH_SIZE)

    if FULL_SYMMETRY_AUGMENTATION:
        transitions = [
            transformed
            for transition in sampled_transitions
            for transformed in all_symmetry_transforms(transition)
        ]
    else:
        transitions = [
            randomly_transform_transition(transition)
            for transition in sampled_transitions
        ]

    optimization_result = optimize_dqn(
        online_network=self.online_network,
        target_network=self.target_network,
        optimizer=self.optimizer,
        transitions=transitions,
        gamma=GAMMA,
        device=self.device,
        double_dqn=DOUBLE_DQN,
        importance_weights=(importance_weights if PRIORITIZED_REPLAY else None),
        return_td_errors=PRIORITIZED_REPLAY,
        auxiliary_opponent_prediction=OPPONENT_PREDICTION,
        auxiliary_bomb_outcome=BOMB_OUTCOME_AUXILIARY,
        bomb_outcome_examples=(
            self.bomb_outcome_buffer.sample(min(BATCH_SIZE, len(self.bomb_outcome_buffer)))
            if BOMB_OUTCOME_AUXILIARY and self.bomb_outcome_buffer else None
        ),
    )
    if PRIORITIZED_REPLAY:
        loss, td_errors = optimization_result
        self.replay_buffer.update_priorities(priority_indices, td_errors)
    else:
        loss = optimization_result

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
