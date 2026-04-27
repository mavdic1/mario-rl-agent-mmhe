# mario-rl-agent-mmhe
Reinforcement Learning–Based Autonomous Gameplay in Super Mario Bros Using Computer Vision for Preprocessing

This is a project for the course Practical Applications of AI at The Faculty of Electrical Engineering of University of Sarajevo.

The students that are working on this project are: Muhamed Avdić, Mak Mičijević, Enis Džinović and Hamza Marić

The goal of this project are evolving as the project is being worked on and as we discover what is practically achievable.

---

# Problem definiton and goal

Reinforcement learning agents often struggle to efficiently learn from raw game frames because the input is high-dimensional and contains irrelevant information.
This slows training and reduces performance.

The current aim is to develop a reinforcement learning (RL) agent capable of playing Super Mario Bros using a combination of modern RL techniques
and computer vision preprocessing as a means of improving the training process.

---

# Steps we need to take


The first step would be implementing a Deep Q-Network (DQN) agent using a Python library gym-super-mario-bros as the RL environment.

The gym library provides a standard interface to the game, giving access to raw frames, rewards, and actions in a structured format that is compatible with common RL libraries.

After that in order to improve the efficiency and effectiveness of training we want to try using OpenCV to preprocess the raw visual input.
The goal is to reduce the complexity of the input and help the agent focus on the most important information, such as the player character, enemies, platforms, and obstacles.

The hope is that OpenCV preprocessing can improve training efficiency by reducing input complexity and choosing transformations that highlight important features,
enabling the agent to learn faster, train more stably, and achieve higher final performance.

To test the OpenCV approach we will try to create two agents trained under identical conditions with the difference being
one using raw game frames, and one using OpenCV-processed frames.

Final step would be comparing these agents in terms of learning speed, the score the models achieve in game, and overall stability.

---

The steps in a more concise format would be:
1. Set up the gym-super-mario-bros environment
2. Implement a Deep Q-Network agent
3. Integrate OpenCV preprocessing pipeline
4. Train two agents: raw frames vs preprocessed frames
5. Compare results: learning speed, score, stability
6. Analyze and summarize findings

---

# Challenges

Some challenges we are expecting are the training of the model can be quite slow,
deciding and optimizing how OpenCV processes the data inorder not to lose important features or make training longer and more difficult.

Training two models also will be a challenge considering this is our first time tackling deep learning but we hope we can learn and add
value to RL models by investigating if OpenCV can be used in a creative way to speed up the learning process.

---

# Resources

https://github.com/Kautenja/gym-super-mario-bros - the main library for the RL enviorment  
https://medium.com/@simeetnayan81/training-an-agent-to-play-breakout-using-deep-reinforcement-learning-b5ca02c81182 - extracting information on how an agent works and how we should go about implementing them  
PyTorch - RL algorithms, DQN  
OpenCV – computer vision library for frame preprocessing  
NumPy / Pandas – for data manipulation  
Matplotlib / Seaborn – for plotting training progress and results  
 

Here is an updated **fully copy-paste safe `README.md`** with **detailed setup steps** added (clean formatting, no special UI blocks, no rendering issues):

---

# Mario RL Agent (PPO + Retro Gym)


# 1. Project Overview

The agent learns to play Mario using:

* PPO (Stable-Baselines3)
* Custom Gym environment (MarioEnv)
* Frame stacking (4 grayscale frames, 84x84)
* Reward shaping based on horizontal progress (x_pos)
* Parallel environments (SubprocVecEnv)

---

# 2. Full Setup Guide (From Zero)

## Step 1: Install system dependencies

On Fedora (or similar Linux):

sudo dnf install -y python3 python3-pip git cmake gcc gcc-c++

You also need OpenGL support:

sudo dnf install -y mesa-libGL mesa-libGLU

---

## Step 2: Project setup

git clone [https://github.com/mavdic1/mario-rl-agent-mmhe.git](https://github.com/mavdic1/mario-rl-agent-mmhe.git)
cd mario-rl-agent-mmhe

# Create Python environment (recommended)

conda create -n mario python=3.8
conda activate mario

pip install -r requirements.txt

---

## Step 3: Import Retro games

This step is REQUIRED.

python -m retro.import .

This scans and registers ROMs.

You must have:

SuperMarioBros-Nes
Level1-1

If missing, Retro will fail.

---

## Step 4: Verify environment works

Run Python:

python

Then:

import retro
env = retro.make(game="SuperMarioBros-Nes", state="Level1-1")
obs = env.reset()
print(obs.shape)

If this works, setup is correct.

---

# 3. Project Structure

mario-rl-agent-mmhe/

├── mario_env.py        # Custom environment wrapper
├── train.py            # PPO training script
├── play.py             # Run trained agent
│
├── models/             # Saved checkpoints (auto-created)
└── mario_ppo.zip       # Final trained model

---

# 4. Training the Agent

Run training:

python train.py

What happens:

* Uses 4 parallel environments
* PPO with CNN policy
* GPU acceleration if available
* Saves models every 50,000 steps

Saved models:

models/mario_ppo_0
models/mario_ppo_50000
models/mario_ppo_100000
...

---

# 5. Playing the Trained Model

Run:

python play.py

It will:

* Load mario_ppo.zip
* Run deterministic policy
* Render gameplay
* Reset automatically on death

---

# 6. Observation Space

Shape:

(4, 84, 84)

Meaning:

* 4 stacked grayscale frames
* 84x84 resolution

---

# 7. Action Space

Uses Retro action space:

* Left
* Right
* Jump
* Run
* Button combinations

---

# 8. Reward System

Current reward:

reward += x_pos * 0.001

This encourages forward movement.

---

# 9. Performance Tips (IMPORTANT)

For faster training on RTX 3050:

* Increase NUM_ENVS (4 → 6 or 8 if CPU allows)
* Use GPU (device="cuda")
* Keep n_steps = 1024 or higher
* Avoid rendering during training
* Close background apps

---

# 10. Saving Models

Auto-save is enabled:

SAVE_EVERY = 50000

You can adjust this in train.py.

---

# 11. Resume Training

To resume from checkpoint:

from stable_baselines3 import PPO

model = PPO.load("models/mario_ppo_50000", env=env)

---

# 12. Common Issues

## SubprocVecEnv crash

Must always use:

if **name** == "**main**":

## Retro errors

Run:

python -m retro.import .

## CNN policy errors

Ensure observation shape is:

(4, 84, 84)

---

# 13. Future Improvements

* Frame skipping (big speed boost)
* Better reward shaping (survival + distance)
* LSTM memory policy
* Curriculum learning (Level 1 → harder levels)
* Action repeat optimization

---

# 14. Run Order

1. python train.py
2. python play.py

---

