#!/usr/bin/env python3
"""
Watch a trained LunarLander DQN model live.

Works with:
- legacy OpenAI Gym: LunarLander-v2
- newer Gym/Gymnasium-style reset/step returns

Examples:
    python3 watch_lander_live.py --model keraslander2_dual3090_best.keras
    python3 watch_lander_live.py --model keraslander2_fast_best.keras --episodes 5
    python3 watch_lander_live.py --model keraslander2_dual3090_best.keras --fps 30
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras


def import_gym():
    """Prefer legacy gym because Roy's LunarLander setup uses LunarLander-v2."""
    try:
        import gym  # type: ignore
        return gym, "gym"
    except Exception:
        pass

    try:
        import gymnasium as gym  # type: ignore
        return gym, "gymnasium"
    except Exception as exc:
        raise RuntimeError(
            "Could not import gym or gymnasium. Install one of these:\n"
            "  pip install 'gym[box2d]'\n"
            "or:\n"
            "  pip install 'gymnasium[box2d]'"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a trained LunarLander model live.")
    parser.add_argument("--model", default="keraslander2_dual3090.keras", help="Path to .keras model file.")
    parser.add_argument("--env", default="LunarLander-v2", help="Use LunarLander-v2 for old gym, LunarLander-v3 for Gymnasium.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of live episodes to watch.")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed.")
    parser.add_argument("--fps", type=float, default=60.0, help="Approximate render speed cap. Use 0 for no delay.")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Random action probability during viewing. Usually keep 0.")
    parser.add_argument("--deterministic", action="store_true", help="Alias for --epsilon 0.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for model inference.")
    return parser.parse_args()


def configure_tensorflow(force_cpu: bool = False) -> None:
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("TensorFlow GPUs detected:", gpus)
        except Exception as exc:
            print("Could not set GPU memory growth:", exc)
    else:
        print("No TensorFlow GPU detected. Running inference on CPU.")


def make_live_env(gym_module, env_id: str):
    """Create a live-rendering env across old Gym and newer Gym/Gymnasium."""
    try:
        env = gym_module.make(env_id, render_mode="human")
        manual_render = False
    except TypeError:
        # Very old Gym does not accept render_mode.
        env = gym_module.make(env_id)
        manual_render = True
    except Exception as exc:
        # Common mismatch: old gym has LunarLander-v2, Gymnasium has LunarLander-v3.
        raise RuntimeError(
            f"Could not create env '{env_id}'.\n"
            "If you installed old gym, try: --env LunarLander-v2\n"
            "If you installed Gymnasium, try: --env LunarLander-v3\n"
            f"Original error: {exc}"
        ) from exc

    setattr(env, "_manual_render_each_step", manual_render)
    return env


def reset_env(env, seed: Optional[int] = None) -> np.ndarray:
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


def step_env(env, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
    else:
        obs, reward, done, info = result
        truncated = bool(info.get("TimeLimit.truncated", False)) if isinstance(info, dict) else False
        terminated = bool(done and not truncated)
    return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), info


def choose_action(model: keras.Model, state: np.ndarray, action_n: int, epsilon: float) -> Tuple[int, np.ndarray]:
    if random.random() < epsilon:
        q_values = model(state[None, :], training=False).numpy()[0]
        return random.randrange(action_n), q_values

    q_values = model(state[None, :], training=False).numpy()[0]
    return int(np.argmax(q_values)), q_values


def main() -> None:
    args = parse_args()
    if args.deterministic:
        args.epsilon = 0.0

    configure_tensorflow(args.cpu)

    gym_module, gym_name = import_gym()
    print(f"Using {gym_name}")
    print(f"Loading model: {args.model}")
    model = keras.models.load_model(args.model, compile=False)

    env = make_live_env(gym_module, args.env)
    action_n = int(env.action_space.n)

    if hasattr(env.action_space, "seed"):
        env.action_space.seed(args.seed)

    delay = 0.0 if args.fps <= 0 else 1.0 / args.fps

    try:
        for episode in range(1, args.episodes + 1):
            state = reset_env(env, seed=args.seed + episode)
            done = False
            total_reward = 0.0
            steps = 0

            print(f"\nEpisode {episode}/{args.episodes}")

            while not done:
                action, q_values = choose_action(model, state, action_n, args.epsilon)
                state, reward, terminated, truncated, _info = step_env(env, action)

                if getattr(env, "_manual_render_each_step", False):
                    env.render()

                total_reward += reward
                steps += 1
                done = terminated or truncated

                # Live console status. Action names for LunarLander:
                # 0 no-op, 1 left engine, 2 main engine, 3 right engine
                print(
                    f"\rstep={steps:4d} reward={total_reward:8.2f} "
                    f"action={action} q={np.round(q_values, 2)}",
                    end="",
                    flush=True,
                )

                if delay > 0:
                    time.sleep(delay)

            reason = "landed/crashed" if terminated else "time-limit"
            print(f"\nEpisode {episode} finished: reward={total_reward:.2f}, steps={steps}, reason={reason}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
