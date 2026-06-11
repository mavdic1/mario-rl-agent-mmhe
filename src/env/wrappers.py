import cv2
import numpy as np


# Preprocess function is the improved preprocessing function
def preprocess(obs):
    # Mask out the blue sky to reduce visual noise
    sky_mask = obs[:, :, 2] > 240
    obs[sky_mask] = 0

    # Crop frame to remove status bar and bottom floor padding
    obs = obs[40:224, 0:256]

    # Convert to grayscale and extract edges using Canny
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    gray = cv2.Canny(gray, 100, 200)

    # Downsample to 84x84 resolution
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_NEAREST)

    # Add channel dimension and cast to uint8 for memory efficiency
    return np.expand_dims(resized, axis=0).astype(np.uint8)

# Preprocess_old function is the original preprocessing function
# It is used in the final evaluation is seeds marked as v1
def preprocess_old(obs):
    # Standard grayscale conversion for baseline version
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    
    # Resize directly without edge detection
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_NEAREST)

    # Add channel dimension and ensure consistent data type
    return np.expand_dims(resized, axis=0).astype(np.uint8)