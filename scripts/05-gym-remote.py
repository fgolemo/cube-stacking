import gym
import cube_stacking
import numpy as np
import tkinter as tk


class Debugger(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.env = gym.make("Cubestacc-SinglePlayer-Heuristic-ArenaFour-NonRandom-Graphical-v0")

        self.obs = self.env.reset()
        self.rew = 0
        self.don = False

        self.motors = []
        for i in range(2):
            m = tk.Scale(
                self, from_=-1, to=1, orient=tk.HORIZONTAL, resolution=0.1)
            m.pack()
            self.motors.append(m)

        tk.Button(self, text='place cube', command=self.step).pack()
        tk.Button(self, text='reset', command=self.reset).pack()
        # tk.Button(self, text='observe', command=self.observe).pack()

    def getActions(self):
        action = [m.get() for m in self.motors]

        return action

    def step(self):
        action = self.getActions()
        self.obs, self.rew, self.done, self.misc = self.env.step(action)
        print(np.around(self.rew, 3), f"done: {self.done}", self.misc)

    def reset(self):
        self.env.reset()

    # def observe(self):
    #     # print(env.unwrapped._get_obs(), env.unwrapped._getReward())
    #     print(np.around(self.obs, 3), np.around(self.rew, 3))


root = tk.Tk()
Debugger(root).pack(side="top", fill="both", expand=True)
root.mainloop()
