from pathlib import Path

# System Settings
NUM_ENVS = 12
EVAL_FREQ = 250_000
TOTAL_TIMESTEPS = 20_000_000
EVAL_EPISODES = 10

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
STUDY_DIR = DATA_DIR / "study"

# PPO Hyperparameters
PPO_CONFIG = {
    "learning_rate": 2.5e-4, # Initial LR
    "n_steps": 1024,
    "batch_size": 4096,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.15,
    "ent_coef": 0.015,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.03,
}

def get_study_paths(version, seed):
    base = STUDY_DIR / version / f"seed_{seed}"
    return {
        "models": base / "models",
        "logs": base / "tensorboard",
        "csv": base / "eval_history.csv",
        "latest": base / "models" / "mario_ppo_latest.zip",
        "best": base / "models" / "mario_ppo_best.zip",
        "marker": base / ".completed"
    }

def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func