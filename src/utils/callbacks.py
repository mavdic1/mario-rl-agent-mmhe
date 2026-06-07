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

        os.makedirs(log_dir, exist_ok=True)
        self.results_csv = os.path.join(log_dir, "eval_history.csv")

        self.best_mean_x = self._recover_best_mean_x()

        self.eval_env = VecFrameStack(
            DummyVecEnv([lambda: MarioEnv(version=self.version)]), 
            n_stack=4
        )

        if not os.path.exists(self.results_csv):
            with open(self.results_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_COLUMNS)

    def _recover_best_mean_x(self):
        """Finds the historical best mean_x from CSV to ensure safe resuming."""
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

        self.pbar.update(self.training_env.num_envs)

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

        infos = self.locals.get("infos", [])

        if infos:
            current_max_x = max(
                [info.get("max_x", 0) for info in infos]
            )

            self.pbar.set_postfix({
                "Live_X": int(current_max_x),
                "Best_X": f"{self.best_mean_x:.1f}"
            })

        if self.model.num_timesteps - self.last_eval >= EVAL_FREQ:
            self.last_eval = self.model.num_timesteps
            self.evaluate()
            self.model.save(self.latest_path)   

        return True

    def evaluate(self):
        """
        Runs a set of evaluation episodes to measure agent progress.
        Logic: 
        - Calculates mean/std of Reward and Global X.
        - Identifies the furthest World-Level reached.
        - Records results to a 10-column CSV.
        """
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
            
            # Run one episode
            while not done and steps < 10000: # 10k step safety ceiling
                # Predict action (Deterministic=False allows for exploration/variation)
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, done_array, info_array = self.eval_env.step(action)
                
                info = info_array[0]
                ep_ret += reward[0]
                
                # Global X logic: 
                # This uses the cumulative distance calculated in mario_env.py
                curr_x = info.get("max_x", 0)
                if curr_x > ep_max_x: 
                    ep_max_x = curr_x
                    # Store which level we were in when we hit this new peak
                    ep_best_lvl = f"{info.get('world', 1)}-{info.get('level', 1)}"
                
                # Check for flag pole or level completion
                if info.get("is_finished", False):
                    wins += 1
                    done = True
                
                done = done or done_array[0]
                steps += 1

            eval_rewards.append(ep_ret)
            eval_xs.append(ep_max_x)
            reached_levels.append(ep_best_lvl)

        # --- 1. STATISTICS CALCULATION ---
        m_reward, s_reward = np.mean(eval_rewards), np.std(eval_rewards)
        m_x, s_x = np.mean(eval_xs), np.std(eval_xs)
        peak_x = np.max(eval_xs) # Absolute record distance in this session
        wr = (wins / num_episodes) * 100
        elapsed = int(time.time() - self.start_time)
        
        # Calculate the "Mode" level (the level reached most frequently)
        best_level_str = max(set(reached_levels), key=reached_levels.count)

        # --- 2. CSV WRITING (10 COLUMNS) ---
        # Header sequence in config.py MUST be:
        # step, elapsed_time_sec, mean_reward, std_reward, mean_x, std_x, peak_x, max_level, win_rate_pct, eval_episodes
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

        # --- 3. LOGGING & PROGRESS ---
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

        # --- 4. MODEL SAVING (Based on Mean Distance) ---
        if m_x > self.best_mean_x:
            self.best_mean_x = m_x
            self.model.save(self.best_path)
            self.logger.info(f"--> NEW BEST DISTANCE MODEL SAVED: {int(m_x)} ({best_level_str})")

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training Mario",
            unit="steps",
            colour="green"
        )

        self.pbar.update(self.model.num_timesteps)
        self.last_eval = self.model.num_timesteps

    def _on_training_end(self):
        if self.pbar is not None:
            self.pbar.close()

        if self.eval_env is not None:
            print(f"\nClosing Evaluation Environment for {self.version}...")
            self.eval_env.close()