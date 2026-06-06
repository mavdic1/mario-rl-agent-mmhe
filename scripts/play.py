import os
import sys
import time
import argparse
import numpy as np
import torch

# Fix pathing to allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from src.env.mario_env import MarioEnv
from src.agent.config import get_study_paths

def main():
    parser = argparse.ArgumentParser(description="Watch the Mario agent play.")
    parser.add_argument("--version", type=str, choices=["v1", "v2"], default="v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["best", "latest"], default="best")
    args = parser.parse_args()

    # 1. Get paths from config
    paths = get_study_paths(args.version, args.seed)
    model_path = paths["best"] if args.mode == "best" else paths["latest"]

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        # Fallback check
        alt_path = paths["latest"] if args.mode == "best" else paths["best"]
        if os.path.exists(alt_path):
            print(f"Loading alternative: {alt_path}")
            model_path = alt_path
        else:
            print("No models found for this version/seed. Did you finish training?")
            return

    # 2. Setup Environment
    # We use DummyVecEnv + VecFrameStack to match training conditions
    def make_env():
        return MarioEnv(version=args.version)

    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)

    # 3. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_path} on {device}")
    model = PPO.load(model_path, env=env, device=device)

    print(f"\nPlaying {args.version.upper()} (Seed {args.seed})")
    print("Press Ctrl+C to stop.")
    print("-" * 50)

    try:
        episode = 1
        while True:
            obs = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Predict action
                action, _ = model.predict(obs, deterministic=False)
                
                # Step
                obs, reward, done, info = env.step(action)
                
                total_reward += reward[0]
                
                # Render (Gym-Retro render)
                env.render()
                
                # Slow down slightly for human eyes
                time.sleep(0.01)

                # Get progress from info dict
                max_x = info[0].get("max_x", 0)
                print(f"Ep {episode} | Max X: {max_x:<5} | Reward: {total_reward:.1f}", end="\r")

            print(f"\nEpisode {episode} Finished. Final X: {max_x}")
            episode += 1
            time.sleep(1) # Gap between episodes

    except KeyboardInterrupt:
        print("\n\nPlayback stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    main()