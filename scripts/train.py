import os
import time
import torch
import csv
import cv2
import numpy as np
from tqdm import tqdm
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from src.env.mario_env import MarioEnv
from src.utils.logger import get_logger
from src.agent.config import get_study_paths, NUM_ENVS, TOTAL_TIMESTEPS
from src.utils.callbacks import MarioCallback
from src.agent.agent_factory import load_or_create_agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, choices=["v1", "v2"], default="v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--dashboard", action="store_true")
    args = parser.parse_args()

    paths = get_study_paths(args.version, args.seed)
    
    current_version = args.version

    is_headless = not args.dashboard

    STUDY_DIR = str(paths["models"].parent) # This points to data/study/v1/seed_0
    LOG_DIR = str(paths["logs"])             # This points to data/study/v1/seed_0/tensorboard
    MODEL_DIR = str(paths["models"])         # This points to data/study/v1/seed_0/models
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    latest_path = os.path.join(MODEL_DIR, "mario_ppo_latest.zip")
    best_path = os.path.join(MODEL_DIR, "mario_ppo_best.zip")

    logger = get_logger(f"T_{args.version}_S{args.seed}", f"{args.version}/seed_{args.seed}/train.log")
    
    def make_env():
        return MarioEnv(version=current_version)

    train_env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])
    train_env = VecFrameStack(train_env, n_stack=4)
    train_env = VecMonitor(train_env)

    model = load_or_create_agent(train_env, LOG_DIR, latest_path)

    callback = MarioCallback(
        total_timesteps=args.total_steps,
        logger=logger,
        headless=not args.dashboard,
        log_dir=STUDY_DIR,
        version=current_version,
        latest_path=latest_path,
        best_path=best_path
    )

    print(f"\nTraining for {args.total_steps:,} timesteps")
    print(f"Tensorboard logs: {LOG_DIR}")
    print("\nRun:")
    print(f"tensorboard --logdir {LOG_DIR}\n")

    try:

        model.learn(
            total_timesteps=args.total_steps,
            callback=callback,
            reset_num_timesteps=False,
            tb_log_name="run" # This creates: study/v1/seed_0/tensorboard/run_1/
        )
        model.save(latest_path)

    except KeyboardInterrupt:

        print("\nTraining interrupted! Saving latest model...")

        model.save(latest_path)

    finally:
        train_env.close()

if __name__ == "__main__":
    main()