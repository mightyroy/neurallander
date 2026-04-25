#!/usr/bin/env python3
"""
Dual-GPU / vectorized DQN trainer for LunarLander using legacy OpenAI Gym.

This version is tuned for a workstation with multiple NVIDIA GPUs, e.g. two
RTX 3090s. It still works with old `gym` and defaults to `LunarLander-v2`.

Important reality check:
- LunarLander DQN is a tiny neural-network workload. The Gym/Box2D environment
  runs on the CPU, so two 3090s will not be fully saturated the way CNN training
  would be.
- To make the GPUs more useful, this version collects experience from multiple
  environments, uses larger replay batches, and can use TensorFlow
  MirroredStrategy across all visible GPUs.

Recommended run on two 3090s:
    python3 keraslander2_oldgym_dual3090.py --fresh

More aggressive:
    python3 keraslander2_oldgym_dual3090.py --fresh --num-envs 32 --batch-size 1024 --gradient-steps 2

Test best model:
    python3 keraslander2_oldgym_dual3090.py --test --episodes 20 --savefile keraslander2_dual3090_best.keras
"""

from __future__ import annotations

import argparse
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple

# Keep TensorFlow logs less noisy. Must be set before importing tensorflow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import gym


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        self.buffer.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                terminated=bool(terminated),
            )
        )

    def add_many(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        terminated: np.ndarray,
    ) -> None:
        for s, a, r, ns, t in zip(states, actions, rewards, next_states, terminated):
            self.add(s, int(a), float(r), ns, bool(t))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch: List[Transition] = random.sample(self.buffer, batch_size)
        states = np.asarray([t.state for t in batch], dtype=np.float32)
        actions = np.asarray([t.action for t in batch], dtype=np.int32)
        rewards = np.asarray([t.reward for t in batch], dtype=np.float32)
        next_states = np.asarray([t.next_state for t in batch], dtype=np.float32)
        terminated = np.asarray([t.terminated for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, terminated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/test a DQN agent on LunarLander with old Gym.")
    parser.add_argument("--env", default="LunarLander-v2", help="Legacy Gym env id. For old gym, use LunarLander-v2.")
    parser.add_argument("--episodes", type=int, default=2500, help="Number of completed train/test episodes.")
    parser.add_argument("--render", action="store_true", help="Show the environment while running. Forces --num-envs 1.")
    parser.add_argument("--record", action="store_true", help="Record videos into --video-dir. Forces --num-envs 1.")
    parser.add_argument("--video-dir", default="./videos/keraslander2_dual3090", help="Directory for recorded videos.")
    parser.add_argument("--test", action="store_true", help="Run greedy evaluation only; no training.")
    parser.add_argument("--savefile", default="keraslander2_dual3090.keras", help="Model save/load path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fresh", action="store_true", help="Ignore an existing savefile and start a new model.")

    # GPU / distribution controls
    parser.add_argument("--force-cpu", action="store_true", help="Hide GPUs from TensorFlow and train on CPU only.")
    parser.add_argument(
        "--strategy",
        choices=["auto", "single", "mirrored"],
        default="auto",
        help="auto = use MirroredStrategy when 2+ GPUs are visible; single = one GPU; mirrored = all GPUs.",
    )

    # Experience collection. Multiple envs improve wall-clock speed because the
    # GPU gets larger batches and the CPU collects more transitions per loop.
    parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel Gym environments for training.")

    # DQN hyperparameters. Defaults are more GPU-friendly than the single-GPU file.
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512, help="Global replay batch size across all GPUs.")
    parser.add_argument("--replay-size", type=int, default=300_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--train-frequency", type=int, default=16, help="Train every N collected environment transitions.")
    parser.add_argument("--gradient-steps", type=int, default=2, help="Replay gradient updates each time training is triggered.")
    parser.add_argument("--target-update-steps", type=int, default=2_000)
    parser.add_argument("--save-every", type=int, default=25, help="Save every N completed episodes.")
    parser.add_argument("--avg-window", type=int, default=100, help="Moving average reward window.")
    parser.add_argument("--hidden-units", type=int, default=512, help="Dense layer width for new models.")
    parser.add_argument("--hidden-layers", type=int, default=3, help="Number of hidden dense layers for new models.")

    # Exploration schedule. Decay is per collected environment step.
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-final", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-steps", type=int, default=150_000)
    return parser.parse_args()


def configure_tensorflow(force_cpu: bool, strategy_name: str) -> tf.distribute.Strategy:
    """Configure TensorFlow and return a distribution strategy."""
    if force_cpu:
        try:
            tf.config.set_visible_devices([], "GPU")
            print("TensorFlow GPU disabled by --force-cpu.")
        except RuntimeError as exc:
            print(f"Could not disable GPU because TensorFlow was already initialized: {exc}")
        return tf.distribute.get_strategy()

    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        print("TensorFlow sees no GPU. Training will run on CPU.")
        return tf.distribute.get_strategy()

    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Already initialized. Not fatal.
            pass

    print("TensorFlow physical GPUs detected:")
    for gpu in physical_gpus:
        print(f"  - {gpu.name}")

    if strategy_name == "single":
        print("Using single-device strategy. TensorFlow will usually use GPU:0.")
        return tf.distribute.get_strategy()

    if strategy_name == "mirrored" or (strategy_name == "auto" and len(physical_gpus) >= 2):
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas.")
        print("Note: NVLink is not selected manually in Python; TensorFlow/NCCL uses the available GPU interconnect.")
        return strategy

    print("Only one GPU visible; using normal single-GPU training.")
    return tf.distribute.get_strategy()


def make_env(env_id: str, render: bool = False, record: bool = False, video_dir: str = "./videos"):
    """Create a single environment compatible with old Gym and Gym 0.26."""
    manual_render = False

    render_mode = None
    if render:
        render_mode = "human"
    elif record:
        render_mode = "rgb_array"

    if render_mode is not None:
        try:
            env = gym.make(env_id, render_mode=render_mode)
        except TypeError:
            env = gym.make(env_id)
            manual_render = bool(render)
    else:
        env = gym.make(env_id)

    if record:
        if hasattr(gym.wrappers, "RecordVideo"):
            env = gym.wrappers.RecordVideo(env, video_folder=video_dir, episode_trigger=lambda ep: True)
        elif hasattr(gym.wrappers, "Monitor"):
            env = gym.wrappers.Monitor(env, video_dir, force=True)
        else:
            print("Warning: this Gym version has no RecordVideo/Monitor wrapper; recording disabled.")

    setattr(env, "_manual_render_each_step", manual_render)
    return env


def make_env_fn(env_id: str, seed: int):
    def _make():
        env = make_env(env_id, render=False, record=False)
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
        try:
            env.seed(seed)
        except Exception:
            pass
        return env

    return _make


def reset_env(env, seed: Optional[int] = None) -> np.ndarray:
    """Handle both Gymnasium reset() -> (obs, info) and old Gym reset() -> obs."""
    try:
        result = env.reset(seed=seed)
    except TypeError:
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)
        result = env.reset()

    if isinstance(result, tuple):
        obs, _info = result
    else:
        obs = result
    return np.asarray(obs, dtype=np.float32)


def reset_vector_env(env, seed: int, num_envs: int) -> np.ndarray:
    """Reset vector envs across older/newer Gym APIs."""
    try:
        result = env.reset(seed=[seed + i for i in range(num_envs)])
    except TypeError:
        try:
            env.seed([seed + i for i in range(num_envs)])
        except Exception:
            pass
        result = env.reset()

    if isinstance(result, tuple):
        obs, _info = result
    else:
        obs = result
    return np.asarray(obs, dtype=np.float32)


def step_env(env, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
    """Return obs, reward, terminated, truncated, info for both new and old step APIs."""
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
    else:
        obs, reward, done, info = result
        truncated = bool(info.get("TimeLimit.truncated", False)) if isinstance(info, dict) else False
        terminated = bool(done and not truncated)
    return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), info


def step_vector_env(env, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    """Return obs, rewards, terminated, truncated, infos from vector env across Gym APIs."""
    result = env.step(actions)
    if len(result) == 5:
        obs, rewards, terminated, truncated, infos = result
    else:
        obs, rewards, done, infos = result
        done = np.asarray(done, dtype=bool)
        truncated = np.zeros_like(done, dtype=bool)
        if isinstance(infos, dict) and "TimeLimit.truncated" in infos:
            truncated = np.asarray(infos["TimeLimit.truncated"], dtype=bool)
        terminated = np.logical_and(done, np.logical_not(truncated))
    return (
        np.asarray(obs, dtype=np.float32),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(terminated, dtype=bool),
        np.asarray(truncated, dtype=bool),
        infos,
    )


def build_q_network(
    obs_dim: int,
    action_dim: int,
    learning_rate: float,
    hidden_units: int = 512,
    hidden_layers: int = 3,
) -> keras.Model:
    inputs = keras.Input(shape=(obs_dim,), name="state")
    x = inputs
    for _ in range(hidden_layers):
        x = layers.Dense(hidden_units, activation="relu")(x)
    outputs = layers.Dense(action_dim, activation="linear", name="q_values")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="lunar_lander_dqn")
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=10.0)
    model.compile(optimizer=optimizer, loss=keras.losses.Huber())
    return model


def load_or_create_model(
    savefile: str,
    obs_dim: int,
    action_dim: int,
    learning_rate: float,
    hidden_units: int,
    hidden_layers: int,
    fresh: bool = False,
) -> keras.Model:
    if os.path.isfile(savefile) and not fresh:
        try:
            model = keras.models.load_model(savefile, compile=False)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=10.0),
                loss=keras.losses.Huber(),
            )
            print(f"Loaded existing model: {savefile}")
            return model
        except Exception as exc:
            print(f"Could not load {savefile}: {exc}")
            print("Creating a new model instead.")

    print("Created new model.")
    return build_q_network(obs_dim, action_dim, learning_rate, hidden_units, hidden_layers)


def epsilon_by_step(step: int, eps_start: float, eps_final: float, decay_steps: int) -> float:
    fraction = min(float(step) / float(decay_steps), 1.0)
    return eps_start + fraction * (eps_final - eps_start)


def sample_random_actions(action_space, n: int) -> np.ndarray:
    return np.asarray([int(action_space.sample()) for _ in range(n)], dtype=np.int64)


def choose_actions(model: keras.Model, states: np.ndarray, action_space, epsilon: float) -> np.ndarray:
    """Vectorized epsilon-greedy action selection."""
    n = int(states.shape[0])
    random_mask = np.random.random(n) < epsilon

    # Only run the model if at least one env needs greedy action. Early in
    # training epsilon is ~1, so this avoids unnecessary GPU calls.
    if np.all(random_mask):
        return sample_random_actions(action_space, n)

    q_values = model(states, training=False).numpy()
    greedy_actions = np.argmax(q_values, axis=1).astype(np.int64)

    if np.any(random_mask):
        random_actions = sample_random_actions(action_space, n)
        greedy_actions[random_mask] = random_actions[random_mask]
    return greedy_actions


def train_one_batch(
    model: keras.Model,
    target_model: keras.Model,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
) -> float:
    states, actions, rewards, next_states, terminated = replay.sample(batch_size)

    # Current predictions for all actions. We only replace the taken action's target.
    targets = model(states, training=False).numpy()

    # Double-DQN target:
    # online model chooses best next action; target model evaluates that action.
    next_q_online = model(next_states, training=False).numpy()
    next_actions = np.argmax(next_q_online, axis=1)
    next_q_target = target_model(next_states, training=False).numpy()
    next_values = next_q_target[np.arange(batch_size), next_actions]

    # Bootstrap only when the episode truly terminated. Time-limit truncation can still bootstrap.
    new_q_values = rewards + gamma * next_values * (1.0 - terminated)
    targets[np.arange(batch_size), actions] = new_q_values

    loss = model.train_on_batch(states, targets)
    if isinstance(loss, (list, tuple)):
        loss = loss[0]
    return float(loss)


def create_training_envs(args: argparse.Namespace):
    """Create either one normal env or an AsyncVectorEnv."""
    if args.render or args.record:
        if args.num_envs != 1:
            print("Rendering/recording forces --num-envs 1.")
        env = make_env(args.env, render=args.render, record=args.record, video_dir=args.video_dir)
        return env, 1, False

    if args.num_envs <= 1:
        env = make_env(args.env, render=False, record=False)
        return env, 1, False

    if not hasattr(gym, "vector") or not hasattr(gym.vector, "AsyncVectorEnv"):
        print("This Gym version has no AsyncVectorEnv; falling back to --num-envs 1.")
        env = make_env(args.env, render=False, record=False)
        return env, 1, False

    env_fns = [make_env_fn(args.env, args.seed + 10_000 + i) for i in range(args.num_envs)]
    env = gym.vector.AsyncVectorEnv(env_fns)
    return env, args.num_envs, True


def train(args: argparse.Namespace, strategy: tf.distribute.Strategy) -> None:
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)

    env, num_envs, is_vector = create_training_envs(args)

    if is_vector:
        action_space = env.single_action_space
        try:
            action_space.seed(args.seed)
        except Exception:
            pass
        obs_shape = env.single_observation_space.shape
        action_dim = int(env.single_action_space.n)
        states = reset_vector_env(env, args.seed, num_envs)
        episode_rewards_running = np.zeros(num_envs, dtype=np.float32)
    else:
        try:
            env.action_space.seed(args.seed)
        except Exception:
            pass
        action_space = env.action_space
        try:
            action_space.seed(args.seed)
        except Exception:
            pass
        obs_shape = env.observation_space.shape
        action_dim = int(env.action_space.n)
        state = reset_env(env, seed=args.seed)
        states = state[None, :]
        episode_rewards_running = np.zeros(1, dtype=np.float32)

    obs_dim = int(obs_shape[0])

    with strategy.scope():
        model = load_or_create_model(
            args.savefile,
            obs_dim,
            action_dim,
            args.learning_rate,
            args.hidden_units,
            args.hidden_layers,
            args.fresh,
        )
        target_model = keras.models.clone_model(model)
        target_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=10.0),
            loss=keras.losses.Huber(),
        )
        target_model.set_weights(model.get_weights())

    replay = ReplayBuffer(args.replay_size)
    recent_rewards: Deque[float] = deque(maxlen=args.avg_window)

    total_steps = 0
    completed_episodes = 0
    best_avg_reward = -float("inf")
    last_loss: Optional[float] = None
    next_train_step = args.learning_starts
    next_target_update = args.target_update_steps

    while completed_episodes < args.episodes:
        epsilon = epsilon_by_step(total_steps, args.epsilon_start, args.epsilon_final, args.epsilon_decay_steps)
        actions = choose_actions(model, states, action_space, epsilon)

        if is_vector:
            next_states, rewards, terminated, truncated, _infos = step_vector_env(env, actions)
            done = np.logical_or(terminated, truncated)
        else:
            ns, reward, term, trunc, _info = step_env(env, int(actions[0]))
            if getattr(env, "_manual_render_each_step", False):
                env.render()
            next_states = ns[None, :]
            rewards = np.asarray([reward], dtype=np.float32)
            terminated = np.asarray([term], dtype=bool)
            truncated = np.asarray([trunc], dtype=bool)
            done = np.asarray([term or trunc], dtype=bool)

        # Use done as the no-bootstrap mask here. In vector envs, completed environments
        # may auto-reset, so bootstrapping from the returned next_state after a truncation
        # can be unsafe. Treating truncation as terminal is slightly conservative.
        replay.add_many(states, actions, rewards, next_states, done)
        episode_rewards_running += rewards
        total_steps += num_envs

        # Train after adding experience. The loop catches up when num_envs > train_frequency.
        if total_steps >= args.learning_starts and len(replay) >= args.batch_size:
            while total_steps >= next_train_step:
                for _ in range(args.gradient_steps):
                    last_loss = train_one_batch(model, target_model, replay, args.batch_size, args.gamma)
                next_train_step += args.train_frequency

        if total_steps >= next_target_update:
            target_model.set_weights(model.get_weights())
            while total_steps >= next_target_update:
                next_target_update += args.target_update_steps

        # Episode accounting and resets.
        if np.any(done):
            finished_indices = np.flatnonzero(done)
            for idx in finished_indices:
                completed_episodes += 1
                ep_reward = float(episode_rewards_running[idx])
                episode_rewards_running[idx] = 0.0
                recent_rewards.append(ep_reward)
                avg_reward = float(np.mean(recent_rewards))
                loss_text = "n/a" if last_loss is None else f"{last_loss:.5f}"
                epsilon_now = epsilon_by_step(total_steps, args.epsilon_start, args.epsilon_final, args.epsilon_decay_steps)

                print(
                    f"episode={completed_episodes:5d} "
                    f"reward={ep_reward:8.2f} "
                    f"avg{args.avg_window}={avg_reward:8.2f} "
                    f"steps={total_steps:8d} "
                    f"epsilon={epsilon_now:.4f} "
                    f"loss={loss_text} "
                    f"envs={num_envs}"
                )

                if avg_reward > best_avg_reward and len(recent_rewards) == args.avg_window:
                    best_avg_reward = avg_reward
                    best_path = os.path.splitext(args.savefile)[0] + "_best.keras"
                    model.save(best_path)
                    print(f"Saved best model to {best_path} with avg reward {best_avg_reward:.2f}")

                if completed_episodes % args.save_every == 0:
                    model.save(args.savefile)
                    print(f"Saved model to {args.savefile}")

                if completed_episodes >= args.episodes:
                    break

            if not is_vector:
                # For a single env we must manually reset after terminal/truncated steps.
                if completed_episodes < args.episodes:
                    next_states[0] = reset_env(env, seed=args.seed + completed_episodes)

        states = next_states

    model.save(args.savefile)
    env.close()
    print(f"Training finished. Final model saved to {args.savefile}")


def choose_action_single(model: keras.Model, state: np.ndarray, env, epsilon: float = 0.0) -> int:
    return int(choose_actions(model, state[None, :], env.action_space, epsilon)[0])


def test(args: argparse.Namespace, strategy: tf.distribute.Strategy) -> None:
    env = make_env(args.env, render=args.render, record=args.record, video_dir=args.video_dir)
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    with strategy.scope():
        model = load_or_create_model(
            args.savefile,
            obs_dim,
            action_dim,
            args.learning_rate,
            args.hidden_units,
            args.hidden_layers,
            fresh=False,
        )

    rewards = []
    for episode in range(1, args.episodes + 1):
        state = reset_env(env, seed=args.seed + episode)
        done = False
        episode_reward = 0.0

        while not done:
            action = choose_action_single(model, state, env, epsilon=0.0)
            state, reward, terminated, truncated, _info = step_env(env, action)
            if getattr(env, "_manual_render_each_step", False):
                env.render()
            done = terminated or truncated
            episode_reward += reward

        rewards.append(episode_reward)
        print(f"test_episode={episode:4d} reward={episode_reward:8.2f}")

    env.close()
    print(f"Average test reward over {len(rewards)} episodes: {np.mean(rewards):.2f}")


def main() -> None:
    args = parse_args()

    if args.render or args.record:
        args.num_envs = 1

    strategy = configure_tensorflow(args.force_cpu, args.strategy)

    print(
        "Settings: "
        f"env={args.env}, episodes={args.episodes}, strategy={args.strategy}, "
        f"replicas={strategy.num_replicas_in_sync}, num_envs={args.num_envs}, "
        f"batch_size={args.batch_size}, gradient_steps={args.gradient_steps}, "
        f"learning_starts={args.learning_starts}, train_frequency={args.train_frequency}, "
        f"epsilon_decay_steps={args.epsilon_decay_steps}, "
        f"hidden_layers={args.hidden_layers}, hidden_units={args.hidden_units}"
    )

    if args.test:
        test(args, strategy)
    else:
        train(args, strategy)


if __name__ == "__main__":
    main()
