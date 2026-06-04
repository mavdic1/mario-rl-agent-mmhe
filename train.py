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


from mario_env import MarioEnv
from logger import get_logger

NUM_ENVS = 12
TOTAL_TIMESTEPS = 20_000_000
EVAL_FREQ = 250_000

MODEL_DIR = "models/"
LOG_DIR = "./logs/ppo_mario_tensorboard/"

LATEST_PATH = os.path.join(MODEL_DIR, "latest/mario_ppo.zip")
BEST_PATH = os.path.join(MODEL_DIR, "best/mario_ppo_best.zip")

class MarioCallback(BaseCallback):
    """
    Custom callback with:
    - tqdm progress bar
    - live max X tracking
    - periodic evaluation
    - automatic checkpoint saving
    """

    def __init__(self, total_timesteps, logger, headless, log_dir, version, latest_path, best_path, verbose=0):
        super().__init__(verbose)
        self.headless = headless
        self.total_timesteps = total_timesteps
        self.logger = logger
        self.version = version
        self.latest_path = latest_path
        self.best_path = best_path
        
        self.pbar = None
        self.best_mean_reward = -np.inf
        self.last_eval = 0

        # Timing and CSV Tracking
        self.start_time = time.time()
        self.results_csv = os.path.join(log_dir, "eval_history.csv")

        os.makedirs(log_dir, exist_ok=True)
        with open(self.results_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "elapsed_time_sec", "mean_reward", "mean_x"])


    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training Mario",
            unit="steps",
            colour="green"
        )

        self.pbar.update(self.model.num_timesteps)

    def _on_step(self) -> bool:

        self.pbar.update(self.training_env.num_envs)

        if not self.headless:
            obs = self.locals.get("new_obs")
            infos = self.locals.get("infos", [])
            rewards = self.locals.get("rewards")
            actions = self.locals.get("actions")

            if obs is not None and len(infos) > 0:
                # --- DATA EXTRACTION (Env 0) ---
                info = infos[0]
                curr_x = info.get("max_x", 0)
                stuck = info.get("stuck_timer", 0)
                rew = rewards[0] if rewards is not None else 0
                obs_tensor = torch.as_tensor(obs).to(self.model.device).float()

                # --- BRAIN LOGIC (VALUE & NEURAL) ---
                with torch.no_grad():
                    # Value Estimate (Optimism)
                    value_est = self.model.policy.predict_values(obs_tensor)[0].item()
                    
                    # Neural Activations (Heatmap)
                    cnn_layer1 = self.model.policy.features_extractor.cnn[0]
                    activations = cnn_layer1(obs_tensor[0:1])
                    heatmap = torch.mean(activations[0], dim=0).cpu().numpy()
                    
                    # Neural Weights (Filters) - first 8
                    weights = cnn_layer1.weight[0:8, 0, :, :].cpu().numpy()

                # --- VISUAL PROCESSING ---
                # Main Game View (Motion Ghosting)
                ghost_frame = np.mean(obs[0], axis=0).astype(np.uint8)
                main_view = cv2.resize(ghost_frame, (350, 350), interpolation=cv2.INTER_NEAREST)
                main_view = cv2.cvtColor(main_view, cv2.COLOR_GRAY2BGR)

                # Heatmap Processing
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                heatmap_img = cv2.resize(heatmap, (180, 180))
                heatmap_img = cv2.applyColorMap((heatmap_img * 255).astype(np.uint8), cv2.COLORMAP_JET)

                # Filters Processing (Grid of 8)
                weight_grid = []
                for w in weights:
                    w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)
                    w_img = cv2.resize(w_norm, (35, 35), interpolation=cv2.INTER_NEAREST)
                    weight_grid.append((w_img * 255).astype(np.uint8))
                row1 = np.hstack(weight_grid[0:4])
                row2 = np.hstack(weight_grid[4:8])
                filters_img = cv2.cvtColor(np.vstack([row1, row2]), cv2.COLOR_GRAY2BGR)

                # --- BUILD CONSOLIDATED CANVAS ---
                # Total Width: 350 (Game) + 200 (Sidebar) = 550
                # Total Height: 60 (Top) + 350 (Mid) + 60 (Bottom) = 470
                canvas = np.zeros((470, 550, 3), dtype=np.uint8)

                # 1. Place Main View
                canvas[60:410, 0:350] = main_view

                # 2. Place Sidebar Visuals
                cv2.putText(canvas, "NEURAL ACTIVATIONS", (365, 80), 0, 0.4, (255,255,255), 1)
                canvas[90:270, 360:540] = heatmap_img
                cv2.putText(canvas, "LEARNED FILTERS", (365, 300), 0, 0.4, (255,255,255), 1)
                canvas[315:385, 380:520] = filters_img

                # --- HUD & TEXT OVERLAYS ---
                # Header Bar
                cv2.rectangle(canvas, (0, 0), (550, 60), (30, 30, 30), -1)
                cv2.putText(canvas, f"STEP: {self.num_timesteps//1000}K", (10, 35), 0, 0.5, (200,200,200), 1)
                cv2.putText(canvas, f"X: {int(curr_x)}", (140, 40), 0, 0.8, (0, 255, 0), 2)
                v_color = (0, 255, 0) if value_est > 0 else (0, 0, 255)
                cv2.putText(canvas, f"VAL: {value_est:.1f}", (380, 38), 0, 0.6, v_color, 2)

                # Bottom Dashboard Bar
                cv2.rectangle(canvas, (0, 410), (550, 470), (20, 20, 20), -1)
                
                # Action Mapping
                action_names = ["IDLE", "RIGHT", "R+JUMP", "R+RUN", "R+R+J", "JUMP", "LEFT"]
                act_str = action_names[actions[0]] if actions[0] < len(action_names) else str(actions[0])
                cv2.putText(canvas, f"INPUT: {act_str}", (10, 435), 0, 0.5, (0, 255, 255), 2)
                
                # Stuck Timer
                s_color = (0, 0, 255) if stuck > 150 else (255, 255, 255)
                cv2.putText(canvas, f"STUCK: {stuck}/250", (10, 458), 0, 0.4, s_color, 1)

                # Reward
                cv2.putText(canvas, f"REW: {rew:.1f}", (200, 435), 0, 0.5, (255, 255, 255), 1)

                # Progress Bar (Yellow)
                progress = min(curr_x / 3200, 1.0)
                cv2.rectangle(canvas, (200, 445), (530, 455), (50, 50, 50), -1)
                cv2.rectangle(canvas, (200, 445), (200 + int(330 * progress), 455), (0, 255, 255), -1)

                # --- DISPLAY ---
                cv2.imshow("Mario AI - Master Research Dashboard", canvas)
                cv2.waitKey(1)

        infos = self.locals.get("infos", [])

        if infos:
            current_max_x = max(
                [info.get("max_x", 0) for info in infos]
            )

            self.pbar.set_postfix({
                "Live_X": int(current_max_x),
                "Best_R": f"{self.best_mean_reward:.1f}"
            })

        if self.model.num_timesteps - self.last_eval >= EVAL_FREQ:
                    self.last_eval = self.model.num_timesteps
                    self.evaluate()
                    self.model.save(self.latest_path)

        return True

    def evaluate(self):
        eval_version = self.version
        eval_vec_env = VecFrameStack(
            DummyVecEnv([lambda: MarioEnv(version=eval_version)]), 
            n_stack=4
        )

        all_rewards = []
        max_xs = []

        for _ in range(3):
            obs = eval_vec_env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_vec_env.step(action)
                total_reward += reward
            all_rewards.append(total_reward)
            max_xs.append(info[0].get("max_x", 0))

        eval_vec_env.close()

        mean_reward = np.mean(all_rewards)
        mean_x = np.mean(max_xs)
        elapsed_time = time.time() - self.start_time
        
        with open(self.results_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.num_timesteps, int(elapsed_time), round(mean_reward, 2), int(mean_x)])

        self.logger.info(
            f"EVAL | "
            f"Step: {self.model.num_timesteps} | "
            f"Reward: {mean_reward:.2f} | "
            f"Max X: {mean_x}"
        )

        self.pbar.set_postfix({
            "Best_R": f"{self.best_mean_reward:.1f}",
            "Curr_R": f"{mean_reward:.1f}",
            "Max_X": int(mean_x)
        })

        if mean_reward > self.best_mean_reward:

            self.best_mean_reward = mean_reward

            self.model.save(self.best_path)

            self.logger.info(
                f"NEW BEST MODEL SAVED -> {self.best_path}"
            )

    def _on_training_end(self):

        if self.pbar is not None:
            self.pbar.close()

def make_env():
    return MarioEnv()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def load_or_create(env, log_dir, latest_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(latest_path):

        # Load the model but force it to use the NEW log_dir for this specific seed
        model = PPO.load(latest_path, env=env, device=device)
        model.tensorboard_log = log_dir 
        return model

    print("--> Creating new PPO model...")

    return PPO(
        policy="CnnPolicy",
        env=env,
        verbose=0,
        tensorboard_log=log_dir,
        device=device,
        learning_rate=linear_schedule(2.5e-4),
        n_steps=1024,
        batch_size=2048,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.015,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03, 
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--version", type=str, choices=["v1", "v2"], default="v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=5000000)
    args = parser.parse_args()

    current_version = args.version

    is_headless = not args.dashboard

    STUDY_DIR = f"study/{args.version}/seed_{args.seed}"
    LOG_DIR = os.path.join(STUDY_DIR, "tensorboard") # logs will go here
    MODEL_DIR = os.path.join(STUDY_DIR, "models")
    
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

    model = load_or_create(train_env, LOG_DIR, latest_path)

    callback = MarioCallback(
        total_timesteps=args.total_steps,
        logger=logger,
        headless=not args.dashboard,
        log_dir=STUDY_DIR,
        version=current_version,
        latest_path=latest_path,
        best_path=best_path
    )

    print(f"\nTraining for {TOTAL_TIMESTEPS:,} timesteps")
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