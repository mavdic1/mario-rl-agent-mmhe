import time
from stable_baselines3 import PPO
from mario_env import MarioEnv

MODEL_PATH = "models/latest/mario_ppo.zip"


def main():
    try:
        model = PPO.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    env = MarioEnv()
    obs = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        env.render()
        time.sleep(0.02)

        if done:
            print("🔁 Reset")
            obs = env.reset()


if __name__ == "__main__":
    main()