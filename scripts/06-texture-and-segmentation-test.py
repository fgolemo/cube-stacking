from cube_stacking.assets import get_tex, get_urdf, TEXTURES
from cube_stacking.sim import CubeStacking
import matplotlib.pyplot as plt
import numpy as np

WID = 256

cam = {"eye": [-8, 8, 8], "lookat": [3, -3, 0]}

sim = CubeStacking(False, (WID, WID), halfbox=True, cam=cam)
# ASSETS

# texUid = sim.p0.loadTexture(get_tex("grass"))
# sim.p0.changeVisualShape(sim.floor, -1, textureUniqueId=texUid)

textures = []
for t in TEXTURES:
    texUid = sim.p0.loadTexture(get_tex(t))
    textures.append(texUid)

sim.place_cube([-4, 4])
sim.place_cube([-3.5, 4])
sim.place_cube([-3.7, 4])
i = 0
while True:
    sim.step()
    i += 1
    if i == 10000:
        tid = np.random.randint(0, len(textures))
        sim.p0.changeVisualShape(sim.floor, -1, textureUniqueId=textures[tid])

        tid = np.random.randint(0, len(textures))
        sim.p0.changeVisualShape(
            sim.wall_left, -1, textureUniqueId=textures[tid])

        tid = np.random.randint(0, len(textures))
        sim.p0.changeVisualShape(
            sim.wall_right, -1, textureUniqueId=textures[tid])
        i = 0
        img, seg = sim.render(segmap=True)
        # print (seg.min(), seg.max(), np.unique(seg))
        segmap_bg = np.zeros((WID, WID))
        segmap_bg[np.logical_or(np.logical_or(seg == 0, seg == 1),
                                seg == 2)] = 1

        segmap_1 = np.zeros((WID, WID))
        segmap_1[seg == 3] = 1
        segmap_2 = np.zeros((WID, WID))
        segmap_2[seg == 4] = 1
        segmap_3 = np.zeros((WID, WID))
        segmap_3[seg == 5] = 1

        fig, axs = plt.subplots(2, 3)
        axs[0, 0].imshow(img)

        axs[0, 1].imshow(segmap_bg)

        axs[1, 0].imshow(segmap_1)
        axs[1, 1].imshow(segmap_2)
        axs[1, 2].imshow(segmap_3)
        plt.show()
