import numpy as np

from mocca_envs.env_locomotion import Walker3DCustomEnv
from mocca_envs.bullet_objects import Chair, Bench


class Walker3DChairEnv(Walker3DCustomEnv):
    robot_random_start = False
    robot_init_position = [0, 0, 1.1]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.robot.set_base_pose(pose="sit")

    def create_terrain(self):
        self.chair = Chair(self._p)
        self.bench = Bench(self._p, pos=np.array([3, 0, 0]))

    def randomize_target(self):
        self.dist = 3.0
        self.angle = 0
        self.stop_frames = 1000
