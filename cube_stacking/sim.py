from enum import Enum
from random import shuffle

import pybullet_data
import pybullet
import pybullet_utils.bullet_client as bc
import numpy as np
from cube_stacking.assets import get_urdf, TEXTURES, get_tex

from cube_stacking.utils import cube_bottom_corners, getImg, CAM_POSES, Player, RandomPositions, getSeg, COLORS, \
    WEIGHT_COLORS

FREQUENCY = 100
FALL_TEST = 50
FALL_THRESH = .01
IMG_SIZE = (84, 84)
DEBUG = False


class CubeStacking(object):

    def __init__(self,
                 headless=True,
                 img_size=IMG_SIZE,
                 cam=None,
                 halfbox=False,
                 four_colors=False,
                 dark=False):
        """ Launch PyBullet instance
        
        :param headless: enable debugging window (very slow, don't use for training)
        :param img_size: tuple, width and height of image 
        """
        super().__init__()
        self.cam = cam
        self.four_colors = four_colors
        self.dark = dark

        if headless:
            mode = pybullet.DIRECT
        else:
            mode = pybullet.GUI

        self.p0 = bc.BulletClient(
            connection_mode=mode)  # pybullet.DIRECT or pybullet.GUI
        self.p0.setAdditionalSearchPath(pybullet_data.getDataPath())

        if not headless:
            self.p0.resetDebugVisualizerCamera(
                cameraDistance=20,
                cameraYaw=225,
                # cameraYaw=135,
                cameraPitch=-45,
                cameraTargetPosition=[0, 0., 0])

        self.p0.setGravity(0, 0, -10)  # good enough
        self.p0.setTimeStep(1 / FREQUENCY)
        self.p0.setRealTimeSimulation(0)

        dark_suffix = ""
        if self.dark:
            dark_suffix = "-dark"

        if not halfbox:
            self.floor = self.p0.loadURDF(get_urdf(f"plane{dark_suffix}"))
        else:
            orientation_right = self.p0.getQuaternionFromEuler(
                [-np.pi / 2, 0, np.pi / 2])
            orientation_left = self.p0.getQuaternionFromEuler(
                [-np.pi / 2, 0, 0])

            self.floor = self.p0.loadURDF(
                get_urdf(f"plane{dark_suffix}"), globalScaling=2)
            self.wall_right = self.p0.loadURDF(
                get_urdf("plane"), [15, 0, 15],
                orientation_right,
                globalScaling=2)
            self.wall_left = self.p0.loadURDF(
                get_urdf("plane"), [0, -15, 15],
                orientation_left,
                globalScaling=2)
            self.textures = []
            for t in TEXTURES:
                texUid = self.p0.loadTexture(get_tex(t))
                self.textures.append(texUid)

        self.colors = COLORS["default"]
        if self.four_colors:
            self.colors = COLORS["4stacc"]

        self.cube_collision = self.p0.createCollisionShape(
            shapeType=self.p0.GEOM_BOX, halfExtents=[1, 1, 1])

        self.cam_width = img_size[0]
        self.cam_height = img_size[1]
        self.setup_camera()

        # these will be reset in the reset function:
        self.color_idx = np.random.randint(0, len(self.colors) - 1) - 1
        self.cubes = []
        self.current_max_z = -1

    def setup_camera(self):
        if self.cam is None:
            self.cam = CAM_POSES["default_15_block"]

        self.cam_view = self.p0.computeViewMatrix(
            cameraEyePosition=self.cam["eye"],
            cameraTargetPosition=self.cam["lookat"],
            cameraUpVector=[0, 0, 1])
        self.cam_proj = self.p0.computeProjectionMatrixFOV(
            fov=90,
            aspect=self.cam_width / self.cam_height,
            nearVal=0.1,
            farVal=50)

    def reset(self):
        if not self.four_colors:
            self.color_idx = np.random.randint(
                0,
                len(self.colors) - 1) - 1  # bc we immediately increment
        else:
            self.color_idx = -1
            shuffle(self.colors)

        for cube in self.cubes:
            self.p0.removeBody(cube)
        self.cubes.clear()
        self.current_max_z = -1

    def place_cube(self,
                   cube_xy,
                   player=None,
                   weight=1,
                   azimuth=None,
                   return_azimuth=False):
        """ put down a new cube, automatically determine the height
        
        :param cube_xy: tuple of x/y position of new cube
        :return: True if the placed cube is higher than any existing cube 
        """

        self.color_idx += 1
        if self.color_idx == len(self.colors):
            self.color_idx = 0
        if azimuth is None:
            azimuth = np.random.randint(0, 180)
        else:
            assert azimuth >= 0 and azimuth <= 180
        cube_rot = self.p0.getQuaternionFromEuler([
            0, 0, np.deg2rad(azimuth)
        ])  # rotated around which axis? # np.deg2rad(90)

        alpha = 1  # this could be set to .5 for some transparency

        if weight == 1:
            if player is None or self.four_colors:
                color = self.colors[self.color_idx] + [alpha]
            elif player == Player.Player:
                color = [0, 0, 1, 1]
                if DEBUG:
                    print("Player putting down cube at", cube_xy)
            elif player == Player.Enemy:
                color = [1, 0, 0, 1]
                if DEBUG:
                    print("Opponent putting down cube at", cube_xy)
            elif player == Player.Starter:
                color = [0, 0, 0, 1]
                if self.dark:
                    color = [1, 1, 1, 1]
                if DEBUG:
                    print("Starter cube at", cube_xy)
        else:
            color = WEIGHT_COLORS[weight]

        max_z = self.find_highest_z(cube_xy, azimuth)

        cube_pos = [cube_xy[0], cube_xy[1], max_z + 1.0001]
        # print ("placing cube at",cube_pos)

        cube_visual = self.p0.createVisualShape(
            shapeType=self.p0.GEOM_BOX,
            rgbaColor=color,
            halfExtents=[1, 1, 1]
            # specularColor=[0.4, .4, 0],
        )

        cube = self.p0.createMultiBody(
            baseMass=weight,
            # baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=self.cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=cube_pos,
            baseOrientation=cube_rot,
            useMaximalCoordinates=True)

        self.cubes.append(cube)

        if max_z > self.current_max_z:
            self.current_max_z = np.around(max_z)
            out = True
        else:
            out = False

        if not return_azimuth:
            return out
        else:
            return out, azimuth

    def find_highest_z(self, cube_xy, azimuth):
        corners_from, corners_to = cube_bottom_corners(cube_xy, azimuth)

        max_z = 0
        for i in range(len(corners_from)):
            res = self.p0.rayTestBatch(
                rayFromPositions=corners_from,
                rayToPositions=corners_to,
            )
            for rayHit in res:
                object_id, link_index, hit_fraction, hit_position, hit_normal = rayHit
                if object_id != 0 and hit_position[2] > max_z:
                    max_z = hit_position[2]

        return max_z

    def step(self):
        self.p0.stepSimulation()

    def last_cube_fell(self, steps=FALL_TEST, threshold=FALL_THRESH):

        # check before sim
        pos_before, _ = self.p0.getBasePositionAndOrientation(self.cubes[-1])

        # roll out a couple steps
        for _ in range(steps - 1):
            self.step()

        # check after
        pos_after, rot_after = self.p0.getBasePositionAndOrientation(
            self.cubes[-1])
        rot_after = self.p0.getEulerFromQuaternion(rot_after)
        if rot_after[0] > threshold or rot_after[1] > threshold or (
                pos_before[2] - pos_after[2]) > threshold:
            return True

        return False

    def render(self, segmap=False):
        flags = 0
        if not segmap:  # for speed
            flags = flags | self.p0.ER_NO_SEGMENTATION_MASK
        img = self.p0.getCameraImage(
            self.cam_width,
            self.cam_height,
            self.cam_view,
            self.cam_proj,
            renderer=self.p0.ER_BULLET_HARDWARE_OPENGL,
            flags=flags
            # lightDirection=[-.5, -1, .5], lightDistance=1,
            # renderer=self.p0.ER_TINY_RENDERER
        )
        output_img = getImg(img, self.cam_height, self.cam_width)
        if not segmap:
            return output_img
        else:
            return output_img, getSeg(img, self.cam_height, self.cam_width)

        # if PLOTTING:
        #     ax0_img = getImg(img0, 200, 200)  # just placeholder for time estimation
        #     ax1_img = getImg(img1, 200, 200)
        #     ax0.set_data(ax0_img)
        #     ax1.set_data(ax1_img)
        #     fig.suptitle(f"step {i}", fontsize=16)
        #     # plot.plot([0])
        #     plt.pause(0.001)

    def close(self):
        self.p0.disconnect()

    @staticmethod
    def apply_randomization(action, randomness):
        # assume action is in [-1,1]
        if randomness == RandomPositions.NonRandom:
            return action
        elif randomness == RandomPositions.Uniform01:
            return action + np.random.uniform(-.1, .1, len(action))
        elif randomness == RandomPositions.Uniform05:
            return action + np.random.uniform(-.5, .5, len(action))
        elif randomness == RandomPositions.Normal05:
            return action + np.random.normal(0, .5)

    def shuffle_textures(self):
        tid = np.random.randint(0, len(self.textures))
        self.p0.changeVisualShape(
            self.floor, -1, textureUniqueId=self.textures[tid])
        tid = np.random.randint(0, len(self.textures))
        self.p0.changeVisualShape(
            self.wall_left, -1, textureUniqueId=self.textures[tid])
        tid = np.random.randint(0, len(self.textures))
        self.p0.changeVisualShape(
            self.wall_right, -1, textureUniqueId=self.textures[tid])


if __name__ == '__main__':
    import time

    cs = CubeStacking(headless=False, cam=CAM_POSES["9.5_block_close"])
    for i in range(3):
        cs.place_cube([-4, -4])
        cs.place_cube([4, 4])
    cs.render()
    time.sleep(10)
