from enum import Enum
import numpy as np
import torch

COLORS = {
    "default": [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
                [0, 1, 1], [.5, .5, .5]],
    "4stacc": [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
}

CAM_POSES = {
    "9.5_block_close": {
        "eye": [10, 10, 15],
        "lookat": [-18, -18, 0]
    },
    "default_15_block": {
        "eye": [12, 12, 25],
        "lookat": [-15, -15, 0]
    },
    "top_down_4_block": {
        "eye": [0, .5, 10],
        "lookat": [0, -.5, 0]
    },
    "physnet": {
        "eye": [-8, 8, 8],
        "lookat": [3, -3, 0]
    }
}

WEIGHT_COLORS = {0.1: [1, 0, 0, 1], 1.0001: [0, 1, 0, 1], 10: [0, 0, 1, 1]}


class RandomPositions(Enum):
    NonRandom = 0,
    Uniform01 = 1,
    Normal05 = 2,
    Uniform05 = 3


class Player(Enum):
    Player = 0
    Enemy = 1
    Starter = 2


class Rewards(Enum):
    PlayerFall = 0
    EnemyFall = 1
    DistanceScale = 2
    Floor = 3
    Tie = 4


def cube_bottom_corners(pos, azimuth):
    corners_from = []
    corners_to = []
    for x, y in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 1), (0, -1), (1, 0),
                 (-1, 0)]:
        # cx, cy - center of square coordinates
        # x, y - coordinates of a corner point of the square

        rotatedX = x * np.cos(azimuth) - y * np.sin(azimuth)
        rotatedY = x * np.sin(azimuth) + y * np.cos(azimuth)

        x = rotatedX + pos[0]
        y = rotatedY + pos[1]

        corners_from.append((x, y, 100))
        corners_to.append((x, y, -1))
    return corners_from, corners_to


def getImg(img, height, width):
    rgb = img[2]
    rgb = np.reshape(rgb, (height, width, 4))
    # rgb = rgb * (1. / 255.)
    rgb = rgb[:, :, :3]
    return rgb


def getSeg(img, height, width):
    seg = img[4]
    seg = np.reshape(seg, (height, width))
    return seg


class NpToTensor(object):

    def __call__(self, arr):
        return torch.from_numpy(arr)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MaskToTensor(object):

    def __call__(self, arr):
        return torch.from_numpy(arr).permute(0, 3, 1, 2)

    def __repr__(self):
        return self.__class__.__name__ + '()'
