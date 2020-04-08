import tkinter as tk
import pybullet as p
import time
import pybullet_data
import numpy as np

from cube_stacking.utils import cube_bottom_corners

FREQUENCY = 100
STEP = .1
FALL_TEST = 200
UPDATE_CUBE = 10
FALL_THRESH = .1


class Debugger(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self._setupSim()
        self._addButtons()

        # start sim
        self.tmp_cube = None
        self.cubes = []
        self._init_vars()

        self.parent.after(int(1000 / FREQUENCY), lambda: self._simulate())

    def _addButtons(self):
        for b in ["left", "right", "forward", "backward"]:
            func = getattr(self, f"move_{b}")
            btn = tk.Button(self, text=b)
            btn.pack()
            btn.bind('<ButtonPress-1>', func)
            btn.bind('<ButtonRelease-1>', self.move_stop)
        tk.Button(self, text='place', command=self.place).pack()
        tk.Button(self, text='reset', command=self.reset).pack()

        self.txt = tk.Label(self, text="===")
        self.txt.pack()

    def _move(self, x, y):
        self.trajectory = [x, y]

    def move_left(self, ev=None):
        self._move(-STEP, 0)

    def move_right(self, ev=None):
        self._move(STEP, 0)

    def move_forward(self, ev=None):
        self._move(0, STEP)

    def move_backward(self, ev=None):
        self._move(0, -STEP)

    def move_stop(self, ev=None):
        self.trajectory = None

    def place(self):
        self._addCube(place=True)

    def reset(self):
        for c in self.cubes:
            p.removeBody(c)
        self.cubes.clear()
        if self.tmp_cube is not None:
            p.removeBody(self.tmp_cube)
        self.tmp_cube = None
        self._init_vars()

    def _init_vars(self):
        self.dirty = False  # gets flipped when block is placed to wait if it falls
        self.trajectory = None
        self.update_counter = 0
        self.current_cube_xy = np.array(
            [np.random.uniform(-4, 4),
             np.random.uniform(-4, 4)])
        self._addCube(place=False)
        self.player = 1
        self.txt.configure(text=f"Player {self.player}, go!")
        self.times = []
        self.timer = time.time()

    def _setupSim(self):
        physicsClient = p.connect(
            p.GUI)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.resetDebugVisualizerCamera(
            cameraDistance=20,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0., 0])

        p.setGravity(0, 0, -10)  # good enough
        p.setTimeStep(1 / FREQUENCY)
        p.setRealTimeSimulation(0)
        p.loadURDF("plane.urdf")

        self.colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
                       [0, 1, 1], [1, 1, 1]]

        self.color_idx = -1  # bc we immediately increment

        self.cube_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX, halfExtents=[1, 1, 1])

    def _simulate(self):
        if self.trajectory is not None:
            self.update_counter += 1
            if self.update_counter == UPDATE_CUBE:
                self.current_cube_xy += self.trajectory
                self._addCube(place=False)
                self.update_counter = 0

        p.stepSimulation()
        if self.dirty:
            pos_before, _ = p.getBasePositionAndOrientation(self.cubes[-1])
            for _ in range(FALL_TEST - 1):
                p.stepSimulation()
            self.dirty = False
            pos_after, rot_after = p.getBasePositionAndOrientation(
                self.cubes[-1])
            rot_after = p.getEulerFromQuaternion(rot_after)
            if rot_after[0] > FALL_THRESH or rot_after[
                    1] > FALL_THRESH or (pos_before[2] - pos_after[2]) > FALL_THRESH:
                print("cube fell")

        self.times.append(time.time() - self.timer)
        if len(self.times) == 1000:
            print(
                f"avg {np.mean(self.times)}s per loop, {1/np.mean(self.times)}Hz"
            )
            self.times.clear()

        self.timer = time.time()

        self.parent.after(int(1000 / FREQUENCY), lambda: self._simulate())

    def _addCube(self, place):
        if self.tmp_cube is not None:
            p.removeBody(self.tmp_cube)
            # and reuse previous color
        else:
            self.color_idx += 1
            if self.color_idx == len(self.colors):
                self.color_idx = 0
            self.azimuth = np.random.randint(0, 180)
            self.cube_rot = p.getQuaternionFromEuler([
                0, 0, np.deg2rad(self.azimuth)
            ])  # rotated around which axis? # np.deg2rad(90)

        alpha = 1 if place else .5
        color = self.colors[self.color_idx] + [alpha]

        corners_from, corners_to = cube_bottom_corners(self.current_cube_xy,
                                                       self.azimuth)
        max_z = 0
        for i in range(len(corners_from)):
            res = p.rayTestBatch(
                rayFromPositions=corners_from,
                rayToPositions=corners_to,
            )
            for rayHit in res:
                object_id, link_index, hit_fraction, hit_position, hit_normal = rayHit
                if object_id != 0 and hit_position[2] > max_z:
                    max_z = hit_position[2]

        cube_pos = [
            self.current_cube_xy[0], self.current_cube_xy[1], max_z + 1.0001
        ]

        cube_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            rgbaColor=color,
            halfExtents=[1, 1, 1]
            # specularColor=[0.4, .4, 0],
        )

        if place:
            cube = p.createMultiBody(
                baseMass=1,
                # baseInertialFramePosition=[0, 0, 0],
                baseCollisionShapeIndex=self.cube_collision,
                baseVisualShapeIndex=cube_visual,
                basePosition=cube_pos,
                baseOrientation=self.cube_rot,
                useMaximalCoordinates=True)
            self.tmp_cube = None
            self.current_cube_xy = np.array(
                [np.random.uniform(-4, 4),
                 np.random.uniform(-4, 4)])
            self._addCube(place=False)
            if self.player == 1:
                self.player = 2
            else:
                self.player = 1
            self.txt.configure(text=f"Player {self.player}, go!")
            self.dirty = True
            self.cubes.append(cube)
        else:
            cube = p.createMultiBody(
                baseMass=0,
                # baseInertialFramePosition=[0, 0, 0],
                baseVisualShapeIndex=cube_visual,
                basePosition=cube_pos,
                baseOrientation=self.cube_rot,
                useMaximalCoordinates=True)
            self.tmp_cube = cube


root = tk.Tk()
Debugger(root).pack(side="top", fill="both", expand=True)
root.mainloop()
