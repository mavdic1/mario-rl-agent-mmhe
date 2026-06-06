# src/agent/factory.py
import os
import torch
from stable_baselines3 import PPO
from src.agent.config import PPO_CONFIG

def linear_schedule(initial_value: float):
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def load_or_create_agent(env, log_dir: str, model_path: str):
    """
    Business logic for Agent creation. 
    Decoupled from CLI arguments.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logic: Resume if file exists, otherwise create new
    if os.path.exists(model_path):
        print(f"--> Resuming existing model: {model_path}")
        model = PPO.load(model_path, env=env, device=device)
        model.tensorboard_log = log_dir
        return model

    print(f"--> Creating new PPO model with log_dir: {log_dir}")
    
    # Prepare params from config
    params = PPO_CONFIG.copy()
    base_lr = params.pop("learning_rate", 2.5e-4)

    return PPO(
        policy="CnnPolicy",
        env=env,
        verbose=0,
        tensorboard_log=log_dir,
        device=device,
        learning_rate=2.5e-4,
        **params
    )