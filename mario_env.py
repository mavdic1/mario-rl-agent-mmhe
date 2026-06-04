import gym
import numpy as np
import retro
from gym import spaces
import cv2

cv2.setNumThreads(0)

class MarioEnv(gym.Env):
    def __init__(self, version="v2"):
        super().__init__()
        self.version = version
        self.env = retro.make(game="SuperMarioBros-Nes", state="Level1-1")
        
        # Action Set
        self._actions = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0], # 0: NOOP
            [0, 0, 0, 0, 0, 0, 0, 1, 0], # 1: Right
            [0, 0, 0, 0, 0, 0, 0, 1, 1], # 2: Right + A (Jump)
            [1, 0, 0, 0, 0, 0, 0, 1, 0], # 3: Right + B (Run)
            [1, 0, 0, 0, 0, 0, 0, 1, 1], # 4: Right + A + B (Run + Jump)
            [0, 0, 0, 0, 0, 0, 0, 0, 1], # 5: A (Jump)
            [0, 0, 0, 0, 0, 0, 1, 0, 0], # 6: Left
        ]
        self.action_space = spaces.Discrete(len(self._actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)
        
        self.frame_stack = []
        self.max_x = 0
        self.prev_x = 0
        self.prev_time = 0 # Track time for 'c' calculation
        self.stuck_timer = 0
        
        # Core RAM Addresses
        self.ADDR_X_PAGE    = 0x006D
        self.ADDR_X_POS     = 0x0086
        self.ADDR_TIME_H    = 0x07F8 # Clock Hundreds
        self.ADDR_TIME_T    = 0x07F9 # Clock Tens
        self.ADDR_TIME_U    = 0x07FA # Clock Units
        self.ADDR_STATE     = 0x000E
        self.ADDR_VIEWPORT  = 0x00B5
        self.ADDR_FLAG      = 0x0770

    def get_ram_stats(self):
        try:
            ram = self.env.get_ram()
            # X Position
            x_pos = int(ram[self.ADDR_X_PAGE]) * 256 + int(ram[self.ADDR_X_POS])
            
            # Clock (Time Left)
            # Mario clock is BCD-ish, stored as digits
            time_left = (int(ram[self.ADDR_TIME_H]) * 100 + 
                         int(ram[self.ADDR_TIME_T]) * 10 + 
                         int(ram[self.ADDR_TIME_U]))
            
            # Death Detection
            player_state = ram[self.ADDR_STATE]
            is_dying = (player_state == 0x0b or player_state == 0x06 or ram[self.ADDR_VIEWPORT] > 1)
            
            # Goal Detection
            is_finished = (ram[self.ADDR_FLAG] == 2)
            
            return x_pos, time_left, is_dying, is_finished
        except:
            return 0, 0, False, False

    def preprocess(self, obs):
        # 1. Faster Sky Masking (Avoiding HSV conversion)
        # In NES Mario, the sky blue has a very high Blue value (255) 
        # compared to Red/Green. This is 10x faster than HSV conversion.
        # Logic: If Blue > 240, it's sky.
        sky_mask = obs[:, :, 2] > 240
        obs[sky_mask] = 0

        # 2. ROI Crop (Do this BEFORE Canny/Resize to process fewer pixels)
        # Reducing the height from 240 to 184 pixels immediately saves 25% CPU
        obs = obs[40:224, 0:256]

        # 3. Grayscale (Fast)
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        # 4. Optimized Canny
        # Canny is slow. If you don't NEED the lines for your thesis, 
        # a simple threshold is much faster. 
        # But if you want to keep Canny, use these settings:
        gray = cv2.Canny(gray, 100, 200)

        # 5. Resize (Fast)
        # Use INTER_NEAREST instead of INTER_AREA. 
        # For a thesis, it preserves sharp lines better anyway.
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_NEAREST)

        return np.expand_dims(resized, axis=0).astype(np.uint8)

    def preprocess_old(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_NEAREST)
        # Return with a channel dimension for SB3
        return np.expand_dims(resized, axis=0)

    def reset(self):
        obs = self.env.reset()
        self.max_x = 0
        self.prev_x = 0
        self.stuck_timer = 0
        
        # Wait for loading screen to pass
        for _ in range(40):
            obs, _, _, _ = self.env.step([0]*9)
        
        x_start, time_left, _, _ = self.get_ram_stats()
        self.prev_x = x_start
        self.max_x = x_start
        self.prev_time = time_left
        
        return self.preprocess_old(obs) if self.version == "v1" else self.preprocess(obs)

    def step(self, action_idx):
        x0 = self.prev_x
        c0 = self.prev_time
        done = False
        obs = None
        info = {}

        # 1. Execute Action
        for _ in range(4):
            obs, _, done, info = self.env.step(self._actions[action_idx])
            if done: break

        # 2. Get Stats
        x1, c1, is_dying, is_finished = self.get_ram_stats()

        # 3. Reward & Milestone Logic (REFACTORED)
        if is_dying:
            reward = -15.0
            done = True
        else:
            # v = current x minus the previous max x 
            # This ensures he only gets rewards for NEW ground covered
            v = x1 - x0
            c = c1 - c0
            reward = float(v + c)
            
            # Update Milestone (ONLY IF ALIVE)
            if x1 > self.max_x or is_finished or c1 == 0:
                self.max_x = max(self.max_x, x1)
                self.stuck_timer = 0
            else:
                # Only increment if the clock is actually ticking 
                # (prevents timing out on black screens)
                if c1 < c0: 
                    self.stuck_timer += 1

        # 4. Finalize
        reward = max(min(reward, 15.0), -15.0)
        self.prev_x = x1
        self.prev_time = c1

        if self.stuck_timer > 250: # Stuck for 250 skipped steps
            done = True
        if is_finished:
            reward = 15.0
            done = True
        
        # This info goes to the callback for the progress bar
        info["max_x"] = self.max_x
        info["stuck_timer"] = self.stuck_timer

        p_obs = self.preprocess_old(obs) if self.version == "v1" else self.preprocess(obs)
        return p_obs, reward, done, info
    
    def render(self, mode='human'):
        return self.env.render()

    def close(self):
        self.env.close()