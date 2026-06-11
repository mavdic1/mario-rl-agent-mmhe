# callbacks.py
import os
import time
import csv
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from tqdm import tqdm

from src.agent.config import CSV_COLUMNS
from src.env.mario_env import MarioEnv
from src.utils.dashboard import show_mario_dashboard
from src.agent.config import EVAL_FREQ, EVAL_EPISODES

LEVEL_FINISH_X = 3150


class MarioCallback(BaseCallback):
    def __init__(self, total_timesteps, logger, headless, log_dir, version, latest_path, best_path, verbose=0):
        super().__init__(verbose)
        self.headless = headless
        self.total_timesteps = total_timesteps
        self.logger = logger
        self.version = version
        self.latest_path = latest_path
        self.best_path = best_path
        
        self.pbar = None
        self.best_mean_x = -np.inf
        self.last_eval = 0

        self.start_time = time.time()

        # Initialize storage directory and history file
        os.makedirs(log_dir, exist_ok=True)
        self.results_csv = os.path.join(log_dir, "eval_history.csv")

        self.best_mean_x = self._recover_best_mean_x()

        # Separate environment for periodic performance testing
        self.eval_env = VecFrameStack(
            DummyVecEnv([lambda: MarioEnv(version=self.version)]), 
            n_stack=4
        )

        # Create CSV and write header if file is new
        if not os.path.exists(self.results_csv):
            with open(self.results_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_COLUMNS)

    def _recover_best_mean_x(self):
        # Scan historical logs to find the highest mean_x for safe resuming
        if os.path.exists(self.results_csv):
            try:
                df = pd.read_csv(self.results_csv)
                if not df.empty and "mean_x" in df.columns:
                    historical_best = df["mean_x"].max()
                    self.logger.info(f"Resuming: Historical Best Mean X is {historical_best:.1f}")
                    return historical_best
            except Exception as e:
                self.logger.error(f"Could not recover best X: {e}")
        return 0

    def _on_step(self) -> bool:
        # Update training progress bar
        self.pbar.update(self.training_env.num_envs)

        # Render dashboard visualizer if not in headless mode
        # Should not be used while training, only for debugging as it greatly reduces performance
        if not self.headless:
            show_mario_dashboard(
                obs=self.locals.get("new_obs"),
                infos=self.locals.get("infos", []),
                rewards=self.locals.get("rewards"),
                actions=self.locals.get("actions"),
                model=self.model,
                num_timesteps=self.num_timesteps,
                version=self.version
            )

        # Update progress bar metrics with current live data
        infos = self.locals.get("infos", [])

        if infos:
            current_max_x = max(
                [info.get("max_x", 0) for info in infos]
            )

            self.pbar.set_postfix({
                "Live_X": int(current_max_x),
                "Best_X": f"{self.best_mean_x:.1f}"
            })

        # Trigger evaluation and save model at defined intervals
        if self.model.num_timesteps - self.last_eval >= EVAL_FREQ:
            self.last_eval = self.model.num_timesteps
            self.evaluate()
            self.model.save(self.latest_path)   

        return True

    def evaluate(self):
        # Measure agent performance across multiple episodes
        eval_rewards = []
        eval_xs = []
        reached_levels = []
        wins = 0

        num_episodes = 100 if self.num_timesteps >= self.total_timesteps else EVAL_EPISODES

        for _ in range(num_episodes):
            obs = self.eval_env.reset()
            done = False
            ep_ret = 0
            ep_max_x = 0
            ep_best_lvl = "1-1"
            steps = 0
            ep_won = False

            while not done and steps < 10000:
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, done_array, info_array = self.eval_env.step(action)
                
                info = info_array[0]
                ep_ret += reward[0]
                
                if info.get("is_finished", False):
                    ep_won = True
                
                curr_x = info.get("max_x", 0)
                if curr_x > ep_max_x: 
                    ep_max_x = curr_x
                    ep_best_lvl = f"{info.get('world', 1)}-{info.get('level', 1)}"
                
                done = done or done_array[0]
                steps += 1

            if ep_won:
                wins += 1
            
            eval_rewards.append(ep_ret)
            eval_xs.append(ep_max_x)
            reached_levels.append(ep_best_lvl)

        # Calculate metrics and identify the most frequent level reached
        m_reward, s_reward = np.mean(eval_rewards), np.std(eval_rewards)
        m_x, s_x = np.mean(eval_xs), np.std(eval_xs)
        peak_x = np.max(eval_xs) # Absolute record distance in this session
        wr = (wins / num_episodes) * 100
        elapsed = int(time.time() - self.start_time)
        
        best_level_str = max(set(reached_levels), key=reached_levels.count)

        # Append performance statistics to the results CSV
        with open(self.results_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.num_timesteps,     # 1. step
                elapsed,                # 2. elapsed_time_sec
                round(m_reward, 2),     # 3. mean_reward
                round(s_reward, 2),     # 4. std_reward
                int(m_x),               # 5. mean_x
                round(s_x, 2),          # 6. std_x
                int(peak_x),            # 7. peak_x
                best_level_str,         # 8. max_level
                int(wr),                # 9. win_rate_pct
                num_episodes            # 10. eval_episodes
            ])

        # Log summary to console and save best model if mean distance improves
        self.logger.info(
            f"EVAL | Step: {self.num_timesteps} | "
            f"Mean X: {int(m_x)} | Peak: {int(peak_x)} | "
            f"Lvl: {best_level_str} | WR: {int(wr)}%"
        )

        if self.pbar is not None:
            self.pbar.set_postfix({
                "Best_X": int(self.best_mean_x),
                "Peak_X": int(peak_x),
                "Lvl": best_level_str
            })

        if m_x > self.best_mean_x:
            self.best_mean_x = m_x
            self.model.save(self.best_path)
            self.logger.info(f"--> NEW BEST DISTANCE MODEL SAVED: {int(m_x)} ({best_level_str})")

    def _on_training_start(self):
        # Initialize terminal progress bar
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training Mario",
            unit="steps",
            colour="green"
        )

        self.pbar.update(self.model.num_timesteps)
        self.last_eval = self.model.num_timesteps

    def _on_training_end(self):
        # Clean up progress bar and evaluation resources
        if self.pbar is not None:
            self.pbar.close()

        if self.eval_env is not None:
            print(f"\nClosing Evaluation Environment for {self.version}...")
            self.eval_env.close()