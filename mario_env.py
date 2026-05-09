import gym
import numpy as np
import retro
import cv2

from gym import spaces


class MarioEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.env = retro.make(
            game="SuperMarioBros-Nes",
            state="Level1-1"
        )

        # =====================================================
        # ACTION SPACE
        # =====================================================

        self._actions = [

            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 NOOP

            [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 1 RIGHT

            [0, 0, 0, 0, 0, 0, 0, 1, 1],  # 2 RIGHT + JUMP

            [1, 0, 0, 0, 0, 0, 0, 1, 0],  # 3 RUN RIGHT

            [1, 0, 0, 0, 0, 0, 0, 1, 1],  # 4 RUN + JUMP

            [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 5 JUMP

            [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 6 LEFT
        ]

        self.action_space = spaces.Discrete(
            len(self._actions)
        )

        # =====================================================
        # OBSERVATION SPACE
        # =====================================================

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(4, 84, 84),
            dtype=np.uint8
        )

        # =====================================================
        # TRACKERS
        # =====================================================

        self.frame_stack = []

        self.max_x = 0
        self.prev_x = 0

        self.stuck_timer = 0

        # =====================================================
        # RAM ADDRESSES
        # =====================================================

        self.ADDR_X_PAGE = 0x006D
        self.ADDR_X_POS = 0x0086

        self.ADDR_PLAYER_STATE = 0x000E
        self.ADDR_VIEWPORT = 0x00B5

        self.ADDR_FLAGPOLE = 0x001D

    # =========================================================
    # RAM PARSING
    # =========================================================

    def get_ram_stats(self):

        try:

            ram = self.env.get_ram()

            # =================================================
            # GLOBAL X POSITION
            # =================================================

            x_pos = (
                int(ram[self.ADDR_X_PAGE]) * 256
                + int(ram[self.ADDR_X_POS])
            )

            # =================================================
            # PLAYER STATE
            # =================================================

            player_state = int(
                ram[self.ADDR_PLAYER_STATE]
            )

            # =================================================
            # DEATH DETECTION
            # =================================================

            is_dying = (
                player_state in [0x06, 0x0B]
                or ram[self.ADDR_VIEWPORT] > 1
            )

            # =================================================
            # LEVEL COMPLETE
            # =================================================

            flag_state = int(
                ram[self.ADDR_FLAGPOLE]
            )

            is_finished = (
                flag_state == 3
                or player_state == 0x05
            )

            return (
                x_pos,
                is_dying,
                is_finished
            )

        except Exception:

            return (
                0,
                False,
                False
            )

    # =========================================================
    # PREPROCESS
    # =========================================================

    def preprocess(self, obs):

        gray = cv2.cvtColor(
            obs,
            cv2.COLOR_RGB2GRAY
        )

        resized = cv2.resize(
            gray,
            (84, 84),
            interpolation=cv2.INTER_AREA
        )

        return resized

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        obs = self.env.reset()

        # =====================================================
        # RESET TRACKERS
        # =====================================================

        self.max_x = 0
        self.prev_x = 0

        self.stuck_timer = 0

        # =====================================================
        # BUFFER FRAMES
        # =====================================================

        for _ in range(10):

            obs, _, _, _ = self.env.step(
                [0] * 9
            )

        # =====================================================
        # INITIAL POSITION
        # =====================================================

        x_start, _, _ = self.get_ram_stats()

        self.prev_x = x_start
        self.max_x = x_start

        # =====================================================
        # FRAME STACK
        # =====================================================

        processed = self.preprocess(obs)

        self.frame_stack = [
            processed for _ in range(4)
        ]

        return np.array(
            self.frame_stack,
            dtype=np.uint8
        )

    # =========================================================
    # STEP
    # =========================================================

    def step(self, action_idx):

        x0 = self.prev_x

        done = False
        info = {}

        obs = None

        # =====================================================
        # FRAME SKIP
        # =====================================================

        for _ in range(4):

            obs, _, done, info = self.env.step(
                self._actions[action_idx]
            )

            if done:
                break

        # =====================================================
        # RAM STATS
        # =====================================================

        x1, is_dying, is_finished = (
            self.get_ram_stats()
        )

        # =====================================================
        # TERMINAL STATES
        # =====================================================

        if is_finished:

            reward = 100.0
            done = True

        elif is_dying:

            reward = -15.0
            done = True

        else:

            # =================================================
            # PROGRESS REWARD
            # =================================================

            reward = float(x1 - x0)

            # =================================================
            # STUCK DETECTION
            # =================================================

            if x1 > self.max_x:

                self.max_x = x1

                self.stuck_timer = 0

            else:

                self.stuck_timer += 1

        # =====================================================
        # STUCK TERMINATION
        # =====================================================

        if self.stuck_timer > 250:

            reward = -10.0
            done = True

        # =====================================================
        # REWARD CLIPPING
        # =====================================================

        reward = np.clip(
            reward,
            -15.0,
            100.0
        )

        # =====================================================
        # UPDATE TRACKERS
        # =====================================================

        self.prev_x = x1

        # =====================================================
        # FRAME STACK
        # =====================================================

        processed = self.preprocess(obs)

        self.frame_stack.pop(0)

        self.frame_stack.append(processed)

        stacked_obs = np.array(
            self.frame_stack,
            dtype=np.uint8
        )

        # =====================================================
        # CALLBACK INFO
        # =====================================================

        info["max_x"] = self.max_x

        return (
            stacked_obs,
            reward,
            done,
            info
        )

    # =========================================================
    # RENDER
    # =========================================================

    def render(self, mode="human"):

        return self.env.render()

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):

        self.env.close()