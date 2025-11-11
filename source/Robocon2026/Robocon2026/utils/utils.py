import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

def euler2quaternion(euler):
    r = R.from_euler("xyz", euler, degrees=True)
    quaternion = r.as_quat()
    return quaternion

def quaternion2euler(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler("xyz", degrees=True)
    return euler

def quaternion2direction(quaternion):
    if quaternion.type == torch.Tensor:
        quaternion = quaternion.cpu().numpy()
    euler = quaternion2euler(quaternion)
    direction = np.array([np.cos(euler[0]) * np.cos(euler[1]), np.sin(euler[0]) * np.cos(euler[1]), np.sin(euler[1])])
    return direction


def print_green(x):
    return print("\033[92m{}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m{}\033[0m".format(x))

def print_red(x):
    return print("\033[91m{}\033[0m".format(x))
