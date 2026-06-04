import time
import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from mario_env import MarioEnv
from logger import get_logger

MODEL_DIR = "models/"
LATEST_PATH = os.path.join(MODEL_DIR, "latest/mario_ppo.zip")
BEST_PATH = os.path.join(MODEL_DIR, "best/mario_ppo_best.zip")

def load_selected_model(env, mode):
    target = BEST_PATH if mode == "best" else LATEST_PATH
    if os.path.exists(target):
        print(f"Loading {mode.upper()} model from: {target}")
        return PPO.load(target, env=env)
    
    alt = LATEST_PATH if mode == "best" else BEST_PATH
    if os.path.exists(alt):
        print(f"{mode.upper()} not found. Loading fallback: {alt}")
        return PPO.load(alt, env=env)
    
    raise FileNotFoundError("No models found. Run train.py first.")

def main():
    parser = argparse.ArgumentParser(description="Watch the Mario agent play.")
    parser.add_argument("mode", choices=["best", "latest"], nargs="?", default="best")
    args = parser.parse_args()

    logger = get_logger("PLAY", "play.log")
    
    # --- CRITICAL CHANGE: Wrap the environment ---
    # The model expects a stack of 4 frames (4, 84, 84)
    env = VecFrameStack(DummyVecEnv([lambda: MarioEnv()]), n_stack=4)
    
    try:
        model = load_selected_model(env, args.mode)
    except Exception as e:
        print(e)
        return

    print(f"\nPlaying {args.mode.upper()} model.")
    print("Press Ctrl+C to stop.")
    print("-" * 50)

    episode = 1
    try:
        while True:
            obs = env.reset()
            done = False
            total_reward = 0
            start_time = time.time()
            
            while not done:
                # Predict action
                action, _ = model.predict(obs, deterministic=True)
                
                # Step (VecEnv returns arrays, so we take index 0)
                obs, reward, done_array, info_array = env.step(action)
                
                reward = reward[0]
                done = done_array[0]
                info = info_array[0]
                
                total_reward += reward
                
                # Render the inner retro environment
                env.render()
                time.sleep(0.01)

                # Get progress
                max_x = info.get("max_x") or 0
                
                # Overwriting status line
                print(f"[{args.mode.upper()}] Ep {episode} | Max X: {max_x:<5} | Ep Reward: {total_reward:.1f}", end="\r")

            # --- EPISODE SUMMARY ---
            duration = time.time() - start_time
            final_x = info.get("max_x") or 0
            
            logger.info(f"MODE: {args.mode} | EP: {episode} | X: {final_x} | R: {total_reward:.2f}")
            
            print(f"\nEpisode {episode} Finished")
            print(f"Max Distance: {final_x}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Duration: {duration:.1f}s")
            print(f"\n")
            
            episode += 1

    except KeyboardInterrupt:
        print("\n\nPlayback stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    main()