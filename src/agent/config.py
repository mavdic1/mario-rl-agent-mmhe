from pathlib import Path

# Execution settings
NUM_ENVS = 12
EVAL_FREQ = 250_000
TOTAL_TIMESTEPS = 5_000_000
EVAL_EPISODES = 10

# Directory structure
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
STUDY_DIR = DATA_DIR / "study"  

# PPO Hyperparameters
PPO_CONFIG = {
    "learning_rate": 2.5e-4,
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

# CSV logging schema
CSV_COLUMNS = [
    "step", "elapsed_time_sec", "mean_reward", "std_reward", 
    "mean_x", "std_x", "peak_x", "max_level", "win_rate_pct", "eval_episodes"
]

def get_study_paths(version, seed):
    # Generates structured paths for model saves and logs
    base = STUDY_DIR / version / f"seed_{seed}"
    return {
        "models": base / "models",
        "logs": base / "tensorboard",
        "csv": base / "eval_history.csv",
        "latest": base / "models" / "mario_ppo_latest.zip",
        "best": base / "models" / "mario_ppo_best.zip",
        "marker": base / ".completed"
    }

#Was not used in the final training representation,
#but could be used or a different similar function when trainig large models
def linear_schedule(initial_value: float):
    # Calculates a decaying learning rate based on training progress
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func