import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from mario_env import MarioEnv
import torch

NUM_ENVS = 4
TOTAL_TIMESTEPS = 2_000_000
SAVE_EVERY = 50_000

LATEST_PATH = "models/latest/mario_ppo.zip"
BEST_PATH = "models/best/mario_ppo_best.zip"


def make_env():
    def _init():
        return MarioEnv()
    return _init


def load_or_create(env):
    if os.path.exists(LATEST_PATH):
        print(f"Resuming from {LATEST_PATH}")
        return PPO.load(LATEST_PATH, env=env, device="cuda")
    else:
        print("Starting new model")
        return PPO(
            "CnnPolicy",
            env,
            verbose=1,
            learning_rate=2.5e-4,
            n_steps=1024,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )


def main():
    os.makedirs("models/latest", exist_ok=True)
    os.makedirs("models/best", exist_ok=True)

    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])
    model = load_or_create(env)

    best_reward = -1e9

    step = 0
    while step < TOTAL_TIMESTEPS:
        model.learn(total_timesteps=SAVE_EVERY, reset_num_timesteps=False)
        step += SAVE_EVERY

        # save latest (always overwrite)
        model.save(LATEST_PATH)

        # evaluate simple heuristic (approx)
        mean_reward = evaluate(model)

        if mean_reward > best_reward:
            best_reward = mean_reward
            model.save(BEST_PATH)
            print(f"🔥 New BEST model saved at step {step} reward={mean_reward}")

        print(f"Saved latest model at step {step}")


def evaluate(model, episodes=3):
    env = MarioEnv()
    total = 0

    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward

        total += ep_reward

    return total / episodes


if __name__ == "__main__":
    main()