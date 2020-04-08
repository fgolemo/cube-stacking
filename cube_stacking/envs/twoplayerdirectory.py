import os
from random import shuffle

import gym
import numpy as np
# rewards:
# 0 on stacking
# -1 on bad placement
# 1 on enemy tower falling after action
# -1 on own tower falling after action
import torch
from gym.spaces import Box

from cube_stacking import RandomPositions
from cube_stacking.self_play_policies import POLICY_DIR
from cube_stacking.sim import CubeStacking
from cube_stacking.utils import Player

MAX_HEIGHT = 15
TEST_STEPS = 50


class TwoPlayerDirectory(gym.Env):

    def __init__(self,
                 randomness,
                 headless=True,
                 max_height=MAX_HEIGHT,
                 arena=4,
                 dir=POLICY_DIR,
                 eval=False,
                 drl="ppo"):
        super().__init__()
        self.randomness = randomness
        self.arena = arena
        self.max_height = max_height
        self.dir = dir
        self.eval = eval
        self.drl = drl
        self.sim = CubeStacking(headless=headless)

        self.action_space = Box(-1, 1, shape=(2,), dtype=np.float32)
        self.observation_space = Box(0, 255, shape=(84, 84, 3), dtype=np.uint8)

        self.seed_val = np.random.randint(0, 100000000)

        self.last_cube_xy = None
        self.opponent = None  # this will be set to the PPO policy later

        self.subfolder = None

    def step(self, action):
        assert len(action) == 2
        action = np.clip(action, -1, 1)
        if not self.rel_action:
            action *= self.arena
        else:
            if self.last_cube_xy is None:
                self.last_cube_xy = np.random.uniform(-1, 1, 2) * self.arena
            action = self.last_cube_xy + action

        action = CubeStacking.apply_randomization(
            action, RandomPositions[self.randomness])

        # cube_xy = np.random.uniform(-4, 4, 2)
        if self.eval:
            player = Player.Player
        else:
            player = None
        higher = self.sim.place_cube(action, player)

        self.last_cube_xy = action  # <- if this is uncommented, PPO puts the cube always at -1,-1 # this should be prevented by uniform random noise

        fall_player = self.sim.last_cube_fell(TEST_STEPS)

        obs = self.sim.render()
        if fall_player:
            return obs, -1, True, {"success": False}

        done = False
        reward = 0
        if not higher:
            reward = -1

        # opponent's turn
        opponent_xy = self._play_opponent(obs)
        opponent_xy_rand = CubeStacking.apply_randomization(
            opponent_xy, RandomPositions[self.randomness])
        self.last_cube_xy += opponent_xy_rand

        if self.eval:
            player = Player.Enemy
        else:
            player = None
        self.sim.place_cube(self.last_cube_xy, player)
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

        # cube_xy = np.random.uniform(-self.arena, self.arena, 2)
        # self.sim.place_cube(cube_xy)
        # self.last_cube_xy = cube_xy

        if self.subfolder is not None:
            self.dir = os.path.join(self.dir, self.subfolder)
            print("switched loading directory to:", self.dir)
            # in order to trigger this only once
            self.subfolder = None

        self._init_opponent()

        # coin toss if player starts or opponent
        if np.random.rand() < .5:
            obs = self.sim.render()
            opponent_xy = self._play_opponent(obs)
            self.last_cube_xy = np.random.uniform(-1, 1, 2) * \
                                self.arena + opponent_xy

            if self.eval:
                player = Player.Enemy
            else:
                player = None
            self.sim.place_cube(self.last_cube_xy, player)
        else:
            self.last_cube_xy = None

        obs = self.sim.render()

        return obs

    def render(self, mode='human'):
        pass
        #TODO

    def seed(self, seed=None):
        self.seed = seed
        np.random.seed(seed)
        return super().seed(seed)

    def close(self):
        self.sim.close()
        super().close()

    def _play_opponent(self, obs):
        if self.opponent is None:
            opponent_xy = np.random.uniform(-1, 1, 2)
        else:
            obs = torch.from_numpy(obs).float().to('cpu')
            obs /= 255
            obs = obs.permute(2, 0, 1)

            if self.drl == "ppo":
                # move obs down on the stacc
                self.stacked_obs[:, :-3] = self.stacked_obs[:, 3:]
                # add new obs on top of stacc
                self.stacked_obs[:, -3:] = obs

                with torch.no_grad():
                    _, action, _, _ = self.opponent.act(
                        self.stacked_obs,
                        self.opp_recurrent_hidden_states,
                        self.opp_masks,
                        deterministic=True)
                opponent_xy = action.numpy()[0]

            elif self.drl == "td3":
                opponent_xy = self.opponent.select_action(np.array(obs), "cpu")

        return opponent_xy

    def _init_opponent(self):
        # get dire contents

        policies = [x for x in os.listdir(self.dir) if f"-{self.drl}.pt" in x]

        if len(policies) == 0:
            # print("ENV: no existing policies")
            self.opponent = None
            return

        # if there is only one policy and we've loaded it, we don't need to reload it
        # if there are 3 or fewer policies, then toss a coin to see if we need to relead the policy
        if self.opponent is not None and (len(policies) == 1 or
                                          (len(policies) <= 3 and
                                           np.random.rand() < .5)):
            if self.drl == "ppo":
                self.opp_masks = torch.zeros(1, 1)
                self.stacked_obs = torch.zeros(
                    (1, 12, 84, 84)).to(torch.device('cpu'))
            elif self.drl == "td3":
                pass
            return

        shuffle(policies)
        policy_path = os.path.join(self.dir, policies[0])
        # print(f"ENV: picking opponent policy '{policy_path}'")

        # We need to use the same statistics for normalization as used in training

        if self.drl == "ppo":
            # notice the tuple
            self.opponent, _ = \
                torch.load(policy_path, map_location='cpu')
        elif self.drl == "td3":
            self.opponent = \
                torch.load(policy_path, map_location='cpu')

        # print("GE: USING POLICY:", policy_path)

        if self.drl == "ppo":
            self.opp_recurrent_hidden_states = torch.zeros(
                1, self.opponent.recurrent_hidden_state_size)
            self.opp_masks = torch.zeros(1, 1)
            self.stacked_obs = torch.zeros((1, 12, 84, 84)).to(torch.device('cpu'))
        elif self.drl == "td3":
            self.opponent.actor.eval()


if __name__ == '__main__':
    import cube_stacking
    import time

    # Cubestacc-TwoPlayer-RelativeAct-NonRandom-Headless-v0

    env = gym.make("Cubestacc-TwoPlayer-RelativeAct-NonRandom-Graphical-Eval-v0")

    for i in range(3):
        obs = env.reset()
        time.sleep(1)
        done = False
        while not done:
            obs, rew, done, misc = env.step([-1, 0])
            time.sleep(1)
            print("reward", rew)

    env.close()
