import os

ASSETS = os.path.dirname(os.path.realpath(__file__))
TEXTURES = [
    "birch",
    "bricks",
    "concrete",
    "grass",
    "mud",
    "sand",
    "stone",
    "wood"
]

def get_tex(name, smol=True):
    smolness = ""
    if smol:
        smolness = "-smol"
    path = os.path.join(ASSETS,"textures",f"{name}{smolness}.png")
    assert os.path.isfile(path)
    return path

def get_urdf(name):
    path = os.path.join(ASSETS, "urdf", f"{name}.urdf.xml")
    assert os.path.isfile(path)
    return path

def get_xacro(name):
    path = os.path.join(ASSETS, "urdf", f"{name}.xacro.xml")
    assert os.path.isfile(path)
    return path

