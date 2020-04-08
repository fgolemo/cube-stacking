import gym
import numpy as np

from gym.spaces import Box

from cube_stacking.sim import CubeStacking
from cube_stacking.utils import CAM_POSES, RandomPositions, Player

MAX_HEIGHT = 10
TEST_STEPS = 50

npa = np.array


class TowerStacc(gym.Env):

    def __init__(self,
                 randomness,
                 headless=True,
                 max_height=MAX_HEIGHT,
                 arena=1,
                 fall_disabled=False,
                 top_down=False,
                 dark=False):
        super().__init__()
        self.randomness = randomness
        self.arena = arena
        self.dark = dark
        self.arena_diag = np.sqrt(self.arena**2 + self.arena**2)
        self.max_height = max_height
        self.top_down = top_down
        self.fall_disabled = fall_disabled
        self._max_episode_steps = self.max_height

        cam = CAM_POSES["9.5_block_close"]
        four_colors = False
        if self.top_down:
            cam = CAM_POSES["top_down_4_block"]
            four_colors = True

        self.sim = CubeStacking(
            headless=headless, cam=cam, four_colors=four_colors, dark=dark)

        self.eval = eval

        self.action_space = Box(-1, 1, shape=(2,), dtype=np.float32)
        self.observation_space = Box(0, 255, shape=(84, 84, 3), dtype=np.uint8)

        self.last_cube_xy = None

    def step(self, action):
        distance = np.linalg.norm(self.ref_cube - np.clip(npa(action),-1,1), 1) # the 1 at the end means L1 distance, not L2
        fall, higher, obs = self.fall_higher_obs(action)

        if self.max_height == 1:
            # -Single- case
            return obs, -distance, True, {}

        # in case of self.fall_disabled, this is never true
        if fall:
            return obs, -1, True, {"success": False}

        # end the episode immediately when the block is not on the starter
        if self.top_down and not higher:
            return obs, -distance, True, {"success": False}

        if not higher or distance >= (2 / self.arena_diag):
            return obs, -distance, False, {}

        # if higher, which is the only left case

        done = False
        misc = {}
        if self.sim.current_max_z >= self.max_height * 2:
            done = True
            misc["success"] = True

        return obs, 1, done, misc

    def fall_higher_obs(self, action):
        assert len(action) == 2
        action = np.clip(action, -1, 1)
        action *= self.arena

        action = CubeStacking.apply_randomization(
            action, RandomPositions[self.randomness])

        higher = self.sim.place_cube(action)

        if not self.fall_disabled:
            fall = self.sim.last_cube_fell(TEST_STEPS)
        else:
            fall = False
        obs = self.sim.render()

        return fall, higher, obs

    def reset(self):
        self.sim.reset()

        cube_xy = np.random.uniform(-self.arena, self.arena, 2)
        self.sim.place_cube(cube_xy, Player.Starter)
        self.ref_cube = cube_xy / self.arena  # to bring into [-1,1]

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

    # env = gym.make("Cubestacc-Towerstacc-ArenaFour-TopDown-Single-NonRandom-Graphical-v0")
    env = gym.make("Cubestacc-Towerstacc-ArenaFour-NoPhys-NonRandom-Graphical-v0")

    for i in range(20):
        obs = env.reset()
        print(obs.shape)
        done = False
        action = env.action_space.sample()
        while not done:
            obs, rew, done, misc = env.step(action)
            print (action, rew, misc)

    env.close()
