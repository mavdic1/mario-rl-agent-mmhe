import os
import time
import torch
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

from mario_env import MarioEnv
from logger import get_logger

# --- CONFIGURATION ---
NUM_ENVS = 10           
TOTAL_TIMESTEPS = 5_000_000
EVAL_FREQ = 250_000      # How often to run the evaluation agent
MODEL_DIR = "models/"
LOG_DIR = "./logs/ppo_mario_tensorboard/"

LATEST_PATH = os.path.join(MODEL_DIR, "latest/mario_ppo.zip")
BEST_PATH = os.path.join(MODEL_DIR, "best/mario_ppo_best.zip")

class MarioCallback(BaseCallback):
    """
    Custom callback for reporting training progress with a tqdm progress bar.
    """
    def __init__(self, total_timesteps, eval_env, logger, verbose=0):
        super(MarioCallback, self).__init__(verbose)
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.eval_env = eval_env
        self.logger = logger
        self.best_mean_reward = -np.inf

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Mario", unit="steps", colour="green")
        self.pbar.update(self.model.num_timesteps)

    def _on_step(self) -> bool:
        # 1. Update the progress bar step count
        self.pbar.update(self.training_env.num_envs)
        
        # 2. Grab LIVE Max X from all 10 running environments
        infos = self.locals.get("infos", [])
        if infos:
            current_max_x = max([info.get("max_x", 0) for info in infos])
            
            # Update the bar with the live Max X
            self.pbar.set_postfix({
                "Live_X": int(current_max_x),
                "Best_R": f"{self.best_mean_reward:.1f}"
            })
        
        # 3. Periodic Evaluation
        if (self.n_calls * self.training_env.num_envs) % EVAL_FREQ == 0:
            self.evaluate()
            self.model.save(LATEST_PATH)
            
        return True

    def evaluate(self):
        all_rewards = []
        max_xs = []
        
        for _ in range(3):
            obs = self.eval_env.reset()
            done = False
            total_rew = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                total_rew += reward
            all_rewards.append(total_rew)
            max_xs.append(info.get("max_x", 0))
            
        mean_reward = np.mean(all_rewards)
        mean_x = np.mean(max_xs)
        
        # Update progress bar description with stats
        self.pbar.set_postfix({
            "Best_R": f"{self.best_mean_reward:.1f}",
            "Curr_R": f"{mean_reward:.1f}",
            "Max_X": int(mean_x)
        })

        self.logger.info(f"EVAL | Step: {self.model.num_timesteps} | Reward: {mean_reward:.2f} | Max X: {mean_x}")

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(BEST_PATH)
            self.logger.info(f"NEW BEST saved to {BEST_PATH}")

    def _on_training_end(self):
        self.pbar.close()

def make_env():
    def _init():
        return MarioEnv()
    return _init

def load_or_create(env):
    if os.path.exists(LATEST_PATH):
        print(f"--> Resuming from checkpoint: {LATEST_PATH}")
        return PPO.load(LATEST_PATH, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
    
    print("--> Creating new PPO model...")
    return PPO(
        "CnnPolicy",
        env,
        verbose=0, # Set to 0 because our callback handles the output
        learning_rate=7e-5,
        n_steps=2048,
        batch_size=1024,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.2,
        tensorboard_log=LOG_DIR,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

def main():
    os.makedirs("models/latest", exist_ok=True)
    os.makedirs("models/best", exist_ok=True)
    
    logger = get_logger("TRAIN", "train.log")

    # 1. Training Envs
    train_env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])
    train_env = VecMonitor(train_env)

    # 2. Evaluation Env
    eval_env = MarioEnv()

    # 3. Model
    model = load_or_create(train_env)

    # 4. Callback
    callback = MarioCallback(TOTAL_TIMESTEPS, eval_env, logger)

    print(f"\nTraining for {TOTAL_TIMESTEPS} steps. Logs: {LOG_DIR}")
    print("Check Tensorboard for detailed charts: tensorboard --logdir logs\n")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("\n\nTraining Interrupted! Saving...")
        model.save(LATEST_PATH)
    finally:
        eval_env.close()
        train_env.close()

if __name__ == "__main__":
    main()