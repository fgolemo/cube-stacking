import os
from collections import deque
from random import shuffle

import gym
import numpy as np
# rewards:
# 0 on stacking
# -1 on bad placement
# 1 on enemy tower falling after action
# -1 on own tower falling after action
import torch
from cube_stacking.assets import TEXTURES, get_tex
from gym.spaces import Box
import time
from cube_stacking.self_play_policies import POLICY_DIR
from cube_stacking.sim import CubeStacking
from cube_stacking.utils import CAM_POSES, Rewards, RandomPositions, Player

MAX_HEIGHT = 10
TEST_STEPS = 50

npa = np.array

REWARDS = {
    "v0": {
        Rewards.PlayerFall: -1,
        Rewards.EnemyFall: +1,
        Rewards.DistanceScale: 1,
        Rewards.Floor: 0,
        Rewards.Tie: 0
    },
    "v1": {
        Rewards.PlayerFall: 0,
        Rewards.EnemyFall: +1,
        Rewards.DistanceScale: 10,
        Rewards.Floor: 0,
        Rewards.Tie: 0
    },
    "v2": {
        Rewards.PlayerFall: -1,
        Rewards.EnemyFall: +1,
        Rewards.DistanceScale: 0,
        Rewards.Floor: -1,
        Rewards.Tie: 0
    },
}


class TwoPlayerDirectoryFullArena(gym.Env):

    def __init__(self,
                 randomness,
                 headless=True,
                 max_height=MAX_HEIGHT,
                 arena=4,
                 dir=POLICY_DIR,
                 eval=False,
                 drl="ppo",
                 reward_scheme=0,
                 no_floor=False,
                 textured=False):
        super().__init__()
        self.randomness = randomness
        self.arena = arena
        self.arena_diag = np.sqrt(self.arena**2 + self.arena**2)
        self.max_height = max_height
        self.dir = dir
        self.eval = eval
        self.drl = drl
        self.no_floor = no_floor
        self.textured = textured
        self.rewards = REWARDS[f"v{reward_scheme}"]
        self.stats = {
            "player_correct_stacks": 0,
            "opponent_correct_stacks": 0,
            "player_floor_placements": 0,
            "opponent_floor_placements": 0,
            "avg_tower_height": deque(maxlen=100),
            "avg_win_rate": deque(maxlen=100),
            "avg_cubes_placed_total": deque(maxlen=100),
            "avg_player_dist_to_ref": deque(maxlen=100),
            "avg_opponent_dist_to_ref": deque(maxlen=100),
            "opponnet_policies": 0
        }
        self.stats_tmp = {}

        if not self.textured:
            self.sim = CubeStacking(
                headless=headless, cam=CAM_POSES["9.5_block_close"])
        else:
            self.sim = CubeStacking(headless=headless, halfbox=True, cam=CAM_POSES["physnet"], four_colors=True)

        self.action_space = Box(-1, 1, shape=(2,), dtype=np.float32)
        self.observation_space = Box(0, 255, shape=(84, 84, 3), dtype=np.uint8)

        self.seed_val = np.random.randint(0, 100000000)

        self.ref_cube = None
        self.opponent = None  # this will be set to the PPO policy later

        self.subfolder = None

    def fall_higher_obs(self, action, player):
        assert len(action) == 2
        action = np.clip(action, -1, 1)
        action *= self.arena

        action = CubeStacking.apply_randomization(
            action, RandomPositions[self.randomness])

        if self.eval:
            color = player
        else:
            color = None

        higher = self.sim.place_cube(action, color)
        fall = self.sim.last_cube_fell(TEST_STEPS)
        obs = self.sim.render()

        return fall, higher, obs

    def _prep_stats(self, win):
        self.stats["avg_tower_height"].append((self.sim.current_max_z + 1) / 2)
        self.stats["avg_win_rate"].append(1 if win else 0)
        self.stats["avg_cubes_placed_total"] = len(self.sim.cubes)
        self.stats["success"] = win

    def step(self, action):
        # negative normalized distance to reference cube
        reward_justin_case = -np.linalg.norm(self.ref_cube -
                                             npa(action)) / self.arena_diag
        self.stats["avg_player_dist_to_ref"].append(reward_justin_case)

        fall_player, higher, obs = self.fall_higher_obs(action, Player.Player)

        if higher:
            self.stats["player_correct_stacks"] += 1
        else:
            self.stats["player_floor_placements"] += 1
            if self.no_floor:
                self._prep_stats(False)
                return obs, self.rewards[Rewards.Floor], True, self.stats

        if fall_player:
            self._prep_stats(False)
            return obs, self.rewards[Rewards.PlayerFall], True, self.stats

        if self.max_height == 4 and len(self.sim.cubes) == 4:
            self._prep_stats(False)
            return obs, self.rewards[Rewards.Tie], True, self.stats

        done = False
        reward = reward_justin_case * self.rewards[Rewards.DistanceScale]

        # opponent's turn
        if self.eval:
            time.sleep(.5)
        opponent_xy = self._play_opponent(obs)
        opp_dist = -np.linalg.norm(self.ref_cube -
                                   opponent_xy) / self.arena_diag
        self.stats["avg_opponent_dist_to_ref"].append(opp_dist)

        fall_opponent, higher_opp, obs = self.fall_higher_obs(
            opponent_xy, Player.Enemy)

        if higher_opp:
            self.stats["opponent_correct_stacks"] += 1
        else:
            self.stats["opponent_floor_placements"] += 1

        if higher and fall_opponent:
            self._prep_stats(True)

            # "avg_tower_height": deque(100),
            #             "avg_win_rate": deque(100)
            reward = self.rewards[Rewards.EnemyFall]
            done = True

        if self.sim.current_max_z >= MAX_HEIGHT * 2 - 0.01:
            done = True

        if self.max_height == 4 and len(self.sim.cubes) == 4:
            self._prep_stats(False)
            reward = self.rewards[Rewards.Tie]
            done = True

        return obs, reward, done, self.stats

    def reset(self):
        self.stats["player_correct_stacks"] = 0
        self.stats["opponent_correct_stacks"] = 0
        self.stats["player_floor_placements"] = 0
        self.stats["opponent_floor_placements"] = 0
        if "success" in self.stats:
            del self.stats["success"]

        self.sim.reset()

        if self.textured:
            self.sim.shuffle_textures()

        cube_xy = np.random.uniform(-self.arena, self.arena, 2)
        self.sim.place_cube(cube_xy, Player.Starter)
        self.ref_cube = cube_xy / self.arena  # to bring into [-1,1]

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
            _, _, obs = self.fall_higher_obs(opponent_xy, Player.Enemy)
        else:
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
            return np.random.uniform(-1, 1, 2)

        obs = torch.from_numpy(obs).float().to('cpu')
        # obs /= 255
        obs = obs.permute(2, 0, 1)

        if self.drl == "ppo":
            # move obs down on the stacc
            # self.stacked_obs[:, :-3] = self.stacked_obs[:, 3:]
            # add new obs on top of stacc
            # self.stacked_obs[:, -3:] = obs
            self.stacked_obs[:, :] = obs

            with torch.no_grad():
                _, action, _, _ = self.opponent.act(
                    self.stacked_obs,
                    self.opp_recurrent_hidden_states,
                    self.opp_masks,
                    deterministic=True)
            opponent_xy = action.numpy()[0]
            self.opp_masks.fill_(1.0)

        elif self.drl == "td3":
            opponent_xy = self.opponent.select_action(np.array(obs), "cpu")

        return opponent_xy

    def _init_opponent(self):
        # get dire contents

        # print (f"ENV: SEARCHING '{self.dir}', filtering for '-{self.drl.upper()}-', got:",os.listdir(self.dir))
        policies = [
            x for x in os.listdir(self.dir)
            if f"-{self.drl.upper()}-" in x and ".pt" in x[-3:]
        ]
        self.stats["opponnet_policies"] = len(policies)

        if len(policies) == 0:
            print("ENV: no existing policies")
            self.opponent = None
            return

        # if there is only one policy and we've loaded it, we don't need to reload it
        # if there are 3 or fewer policies, then toss a coin to see if we need to relead the policy
        if self.opponent is not None and (len(policies) == 1 or
                                          (len(policies) <= 3 and
                                           np.random.rand() < .5)):
            if self.drl == "ppo":
                self.opp_masks = torch.zeros(1, 1)
                # self.stacked_obs = torch.zeros(
                #     (1, 12, 84, 84)).to(torch.device('cpu'))
                self.stacked_obs = torch.zeros(
                    (1, 3, 84, 84)).to(torch.device('cpu'))
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
            # self.stacked_obs = torch.zeros(
            #     (1, 12, 84, 84)).to(torch.device('cpu'))
            self.stacked_obs = torch.zeros(
                (1, 3, 84, 84)).to(torch.device('cpu'))
        elif self.drl == "td3":
            self.opponent.actor.eval()


if __name__ == '__main__':
    import cube_stacking
    import time

    # Cubestacc-TwoPlayer-RelativeAct-NonRandom-Headless-v0

    env = gym.make("Cubestacc-TwoPlayer-Full-H4-NonRandom-PPO-Graphical-v2")

    for i in range(10):
        obs = env.reset()
        time.sleep(1)
        done = False
        while not done:
            obs, rew, done, misc = env.step([0, 0])
            time.sleep(1)
            print("reward", rew)

    env.close()
