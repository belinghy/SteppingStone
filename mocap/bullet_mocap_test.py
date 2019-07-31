import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, parent_dir)

import gym
import numpy as np
import pybullet

from environments.bullet_objects import VSphere
from environments.bullet_utils import BulletClient, Camera, SinglePlayerStadiumScene


def extract_joints_xyz(frame, offset=None):
    x_indices = slice(3, 69, 3)
    y_indices = slice(4, 69, 3)
    z_indices = slice(5, 69, 3)

    offset = np.zeros(3) if offset is None else offset

    pose = np.stack(
        (
            frame[x_indices] + offset[0],
            frame[z_indices] + offset[1],
            frame[y_indices] + offset[2],
        ),
        axis=1,
    )

    return pose


class MocapRobot:
    def __init__(self, bc):
        num_joints = 22

        colours = [
            (0.9411764705882353, 0.9725490196078431, 1.0, 1.0),
            (0.9803921568627451, 0.9215686274509803, 0.8431372549019608, 1.0),
            (0.0, 1.0, 1.0, 1.0),
            (0.4980392156862745, 1.0, 0.8313725490196079, 1.0),
            (0.9411764705882353, 1.0, 1.0, 1.0),
            (0.9607843137254902, 0.9607843137254902, 0.8627450980392157, 1.0),
            (1.0, 0.8941176470588236, 0.7686274509803922, 1.0),
            (0.0, 0.0, 0.0, 1.0),
            (1.0, 0.9215686274509803, 0.803921568627451, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (0.5411764705882353, 0.16862745098039217, 0.8862745098039215, 1.0),
            (0.6470588235294118, 0.16470588235294117, 0.16470588235294117, 1.0),
            (0.8705882352941177, 0.7215686274509804, 0.5294117647058824, 1.0),
            (0.37254901960784315, 0.6196078431372549, 0.6274509803921569, 1.0),
            (0.4980392156862745, 1.0, 0.0, 1.0),
            (0.8235294117647058, 0.4117647058823529, 0.11764705882352941, 1.0),
            (1.0, 0.4980392156862745, 0.3137254901960784, 1.0),
            (0.39215686274509803, 0.5843137254901961, 0.9294117647058824, 1.0),
            (1.0, 0.9725490196078431, 0.8627450980392157, 1.0),
            (0.8627450980392157, 0.0784313725490196, 0.23529411764705882, 1.0),
            (0.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.5450980392156862, 1.0),
        ]

        self.ordered_joints = [VSphere(bc, radius=0.065) for i in range(num_joints)]
        self.root_pos = np.zeros(3)
        self.root_facing = 0

    def apply_action(self, root_delta, root_facing, joints_pos):
        self.root_pos[0:2] += root_delta
        self.root_facing += root_facing

        for i, j in enumerate(self.ordered_joints):
            j.set_position(joints_pos[i] + self.root_pos)


def main():
    mocap_file = os.path.join(current_dir, "mocap_all.npy")
    # Walk including 180 degrees turn
    # mocap_data = np.load(mocap_file)[19626:20117]
    mocap_data = np.load(mocap_file)

    p = BulletClient(connection_mode=pybullet.GUI)
    p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

    camera = Camera(p, fps=30, dist=10)
    scene = SinglePlayerStadiumScene(p, gravity=9.8, timestep=1 / 30, frame_skip=1)
    scene.initialize()

    robot = MocapRobot(p)
    index = 0

    while True:
        index = (index + 1) % mocap_data.shape[0]
        frame = mocap_data[index]

        yaw = -robot.root_facing
        matrix2d = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        root_delta = np.matmul(matrix2d, frame[0:2])

        matrix3d = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )

        pose = extract_joints_xyz(frame)
        pose = np.matmul(matrix3d, pose.T).T

        robot.apply_action(root_delta * 0.3048, frame[2], pose * 0.3048)

        camera.track(pos=robot.root_pos)


if __name__ == "__main__":
    main()
