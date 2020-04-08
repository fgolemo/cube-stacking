from enum import Enum

import gym
import pybullet_data
import pybullet
import pybullet_utils.bullet_client as bc
import numpy as np

# rewards:
# 0 on stacking
# -1 on bad placement
# 1 on enemy tower falling after action
# -1 on own tower falling after action
from gym.spaces import Box

from cube_stacking import RandomPositions
from cube_stacking.sim import CubeStacking
from cube_stacking.utils import Player

MAX_HEIGHT = 15
TEST_STEPS = 50


class SinglePlayer(gym.Env):

    def __init__(self,
                 randomness,
                 headless=True,
                 max_height=MAX_HEIGHT,
                 arena=4,
                 relative_action=False,
                 eval=False):
        super().__init__()
        self.randomness = randomness
        self.arena = arena
        self.rel_action = relative_action
        self.max_height = max_height
        self.sim = CubeStacking(headless=headless)
        self.eval = eval

        self.action_space = Box(-1, 1, shape=(2,), dtype=np.float32)
        self.observation_space = Box(0, 255, shape=(84, 84, 3), dtype=np.uint8)

        self.last_cube_xy = None

    def step(self, action):
        assert len(action) == 2
        action = np.clip(action, -1, 1)
        if not self.rel_action:
            action *= self.arena
        else:
            action = self.last_cube_xy + action

        action = CubeStacking.apply_randomization(
            action, RandomPositions[self.randomness])

        # cube_xy = np.random.uniform(-4, 4, 2)
        if self.eval:
            player = Player.Player
        else:
            player = None

        higher = self.sim.place_cube(action, player)
        # self.last_cube_xy = action # <- if this is uncommented, PPO puts the cube always at -1,-1
        fall_player = self.sim.last_cube_fell(TEST_STEPS)

        if fall_player:
            obs = self.sim.render()
            return obs, -1, True, {"success": False}

        done = False
        reward = 0
        if not higher:
            reward = -1

        # opponent's turn
        opponent_xy = self.last_cube_xy + np.random.uniform(-1, 1, 2)

        if self.eval:
            player = Player.Enemy
        else:
            player = None

        self.sim.place_cube(opponent_xy, player)
        self.last_cube_xy = opponent_xy
        fall_opponent = self.sim.last_cube_fell(TEST_STEPS)
        obs = self.sim.render()

        misc = {}
        if higher and fall_opponent:
            reward = 1
            done = True
            misc["success"] = True

        if self.sim.current_max_z >= MAX_HEIGHT * 2:
            done = True

        return obs, reward, done, misc

    def reset(self):
        self.sim.reset()
        cube_xy = np.random.uniform(-self.arena, self.arena, 2)

        if self.eval:
            player = Player.Starter
        else:
            player = None

        self.sim.place_cube(cube_xy, player)
        self.last_cube_xy = cube_xy
        obs = self.sim.render()
        return obs

    def render(self, mode='human'):
        pass
        #TODO

    def seed(self, seed=None):
        np.random.seed(seed)
        return super().seed(seed)

    def close(self):
        self.sim.close()
        super().close()


if __name__ == '__main__':
    import cube_stacking

    env = gym.make("Cubestacc-TwoPlayer-RelativeAct-NonRandom-Graphical-v0")
    obs = env.reset()
    print(obs.shape)

    env.close()
