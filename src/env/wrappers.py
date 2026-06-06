import cv2
import numpy as np

def preprocess(obs):
    sky_mask = obs[:, :, 2] > 240
    obs[sky_mask] = 0

    obs = obs[40:224, 0:256]

    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    gray = cv2.Canny(gray, 100, 200)

    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_NEAREST)

    return np.expand_dims(resized, axis=0).astype(np.uint8)

def preprocess_old(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_NEAREST)

    return np.expand_dims(resized, axis=0)