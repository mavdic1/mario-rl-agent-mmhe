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


# =========================================================
# CONFIG
# =========================================================

NUM_ENVS = 10
TOTAL_TIMESTEPS = 10_000_000
EVAL_FREQ = 250_000

MODEL_DIR = "models/"
LOG_DIR = "./logs/ppo_mario_tensorboard/"

LATEST_PATH = os.path.join(MODEL_DIR, "latest/mario_ppo.zip")
BEST_PATH = os.path.join(MODEL_DIR, "best/mario_ppo_best.zip")


# =========================================================
# CALLBACK
# =========================================================

class MarioCallback(BaseCallback):
    """
    Custom callback with:
    - tqdm progress bar
    - live max X tracking
    - periodic evaluation
    - automatic checkpoint saving
    """

    def __init__(self, total_timesteps, eval_env, logger, verbose=0):
        super().__init__(verbose)

        self.total_timesteps = total_timesteps
        self.eval_env = eval_env
        self.logger = logger

        self.pbar = None

        self.best_mean_reward = -np.inf
        self.last_eval = 0

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training Mario",
            unit="steps",
            colour="green"
        )

        self.pbar.update(self.model.num_timesteps)

    def _on_step(self) -> bool:

        # =====================================================
        # Progress bar update
        # =====================================================

        self.pbar.update(self.training_env.num_envs)

        infos = self.locals.get("infos", [])

        if infos:
            current_max_x = max(
                [info.get("max_x", 0) for info in infos]
            )

            self.pbar.set_postfix({
                "Live_X": int(current_max_x),
                "Best_R": f"{self.best_mean_reward:.1f}"
            })

        # =====================================================
        # Evaluation
        # =====================================================

        if self.model.num_timesteps - self.last_eval >= EVAL_FREQ:

            self.last_eval = self.model.num_timesteps

            self.evaluate()

            self.model.save(LATEST_PATH)

        return True

    def evaluate(self):

        all_rewards = []
        max_xs = []

        for _ in range(3):

            obs = self.eval_env.reset()

            done = False
            total_reward = 0

            while not done:

                action, _ = self.model.predict(
                    obs,
                    deterministic=True
                )

                obs, reward, done, info = self.eval_env.step(action)

                total_reward += reward

            all_rewards.append(total_reward)
            max_xs.append(info.get("max_x", 0))

        mean_reward = np.mean(all_rewards)
        mean_x = np.mean(max_xs)

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

        # =====================================================
        # Save best model
        # =====================================================

        if mean_reward > self.best_mean_reward:

            self.best_mean_reward = mean_reward

            self.model.save(BEST_PATH)

            self.logger.info(
                f"NEW BEST MODEL SAVED -> {BEST_PATH}"
            )

    def _on_training_end(self):

        if self.pbar is not None:
            self.pbar.close()


# =========================================================
# ENV FACTORY
# =========================================================

def make_env():

    def _init():
        return MarioEnv()

    return _init


# =========================================================
# LOAD / CREATE MODEL
# =========================================================

def load_or_create(env):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(LATEST_PATH):

        print(f"--> Resuming from checkpoint: {LATEST_PATH}")

        return PPO.load(
            LATEST_PATH,
            env=env,
            device=device
        )

    print("--> Creating new PPO model...")

    return PPO(
        policy="CnnPolicy",
        env=env,

        verbose=0,

        tensorboard_log=LOG_DIR,
        device=device,

        # =====================================================
        # PPO HYPERPARAMETERS
        # =====================================================

        learning_rate=2.5e-4,

        n_steps=2048,

        batch_size=1024,

        n_epochs=10,

        gamma=0.99,

        gae_lambda=0.95,

        clip_range=0.1,

        ent_coef=0.01,

        vf_coef=0.5,

        max_grad_norm=0.5,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    os.makedirs("models/latest", exist_ok=True)
    os.makedirs("models/best", exist_ok=True)

    logger = get_logger("TRAIN", "train.log")

    # =====================================================
    # TRAIN ENVS
    # =====================================================

    train_env = SubprocVecEnv(
        [make_env() for _ in range(NUM_ENVS)]
    )

    train_env = VecMonitor(train_env)

    # =====================================================
    # EVAL ENV
    # =====================================================

    eval_env = MarioEnv()

    # =====================================================
    # MODEL
    # =====================================================

    model = load_or_create(train_env)

    # =====================================================
    # CALLBACK
    # =====================================================

    callback = MarioCallback(
        TOTAL_TIMESTEPS,
        eval_env,
        logger
    )

    print(f"\nTraining for {TOTAL_TIMESTEPS:,} timesteps")
    print(f"Tensorboard logs: {LOG_DIR}")
    print("\nRun:")
    print(f"tensorboard --logdir {LOG_DIR}\n")

    # =====================================================
    # TRAIN
    # =====================================================

    try:

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            reset_num_timesteps=False
        )

    except KeyboardInterrupt:

        print("\nTraining interrupted! Saving latest model...")

        model.save(LATEST_PATH)

    finally:

        eval_env.close()
        train_env.close()


# =========================================================
# ENTRY
# =====================================================

if __name__ == "__main__":
    main()