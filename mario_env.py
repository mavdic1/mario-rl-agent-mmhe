import gym
import numpy as np
import retro
from gym import spaces
import cv2

class MarioEnv(gym.Env):
    def __init__(self):
        super().__init__()
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
        self.observation_space = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
        
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
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_NEAREST)

    def reset(self):
        # 1. Hardware Reset
        obs = self.env.reset()
        
        # 2. CLEAR TRACKERS IMMEDIATELY
        self.max_x = 0
        self.prev_x = 0
        self.stuck_timer = 0
        
        # 3. Buffer Period (Let the level load fully)
        for _ in range(10):
            obs, _, _, _ = self.env.step([0]*9)
        
        # 4. Get Actual Start Position after buffer
        x_start, time_left, _, _ = self.get_ram_stats()
        self.prev_x = x_start
        self.max_x = x_start  # Now we know for sure where we are
        self.prev_time = time_left
        
        # 5. Prepare Frame Stack
        processed = self.preprocess(obs)
        self.frame_stack = [processed for _ in range(4)]
        return np.array(self.frame_stack, dtype=np.uint8)

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
            if x1 > self.max_x:
                self.max_x = x1
                self.stuck_timer = 0
            else:
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

        processed = self.preprocess(obs)
        self.frame_stack.pop(0)
        self.frame_stack.append(processed)
        stacked_obs = np.array(self.frame_stack, dtype=np.uint8)
        
        # This info goes to the callback for the progress bar
        info["max_x"] = self.max_x
        return stacked_obs, reward, done, info
    
    def render(self, mode='human'):
        return self.env.render()

    def close(self):
        self.env.close()