import gym
import gym.utils.seeding
import numpy as np
import pybullet

from environments.bullet_utils import BulletClient, Camera, SinglePlayerStadiumScene


class EnvBase(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}
    _render_width = 320 * 3
    _render_height = 240 * 3

    def __init__(self, robot_class, render=False):
        self.scene = None
        self.physics_client_id = -1
        self.owns_physics_client = 0
        self.state_id = -1

        self.metadata["video.frames_per_second"] = int(1 / self.control_step)

        self.is_render = render
        self.robot_class = robot_class

        self.seed()
        self.initialize_scene_and_robot()

    def close(self):
        if self.owns_physics_client and self.physics_client_id >= 0:
            self._p.disconnect()
        self.physics_client_id = -1

    def initialize_scene_and_robot(self):

        self.owns_physics_client = True

        bc_mode = pybullet.GUI if self.is_render else pybullet.DIRECT
        self._p = BulletClient(connection_mode=bc_mode)

        if self.is_render:
            self.camera = Camera(self._p, 1 / self.control_step)
            if hasattr(self, "create_target"):
                self.create_target()

        self.physics_client_id = self._p._client
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

        self.scene = SinglePlayerStadiumScene(
            self._p,
            gravity=9.8,
            timestep=self.control_step / self.llc_frame_skip / self.sim_frame_skip,
            frame_skip=self.sim_frame_skip,
        )
        self.scene.initialize()

        # Create floor
        self.ground_ids = {(self.scene.ground_plane_mjcf[0], -1)}

        # Create robot object
        self.robot = self.robot_class(self._p)
        self.robot.initialize()
        self.robot.np_random = self.np_random

        # Create terrain
        if hasattr(self, "create_terrain"):
            self.create_terrain()

        self.state_id = self._p.saveState()

    def set_env_params(self, params_dict):
        for k, v in params_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def set_robot_params(self, params_dict):
        for k, v in params_dict.items():
            if hasattr(self.robot, k):
                setattr(self.robot, k, v)

        # Right now only power can be set
        # Make sure to recalculate torque limit
        self.robot.calc_torque_limits()

    def render(self, mode="human"):
        # Taken care of by pybullet
        if not self.is_render:
            self.is_render = True
            self._p.disconnect()
            self.initialize_scene_and_robot()
            self.reset()

        if mode != "rgb_array":
            return np.array([])

        yaw, pitch, dist, lookat = self._p.getDebugVisualizerCamera()[-4:]

        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=lookat,
            distance=dist,
            yaw=yaw,
            pitch=pitch,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self._render_width) / self._render_height,
            nearVal=0.1,
            farVal=100.0,
        )
        (_, _, px, _, _) = self._p.getCameraImage(
            width=self._render_width,
            height=self._render_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
        )
        rgb_array = np.array(px)
        rgb_array = np.reshape(
            np.array(px), (self._render_height, self._render_width, -1)
        )
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def reset(self):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, a):
        raise NotImplementedError

    def _handle_keyboard(self):
        keys = self._p.getKeyboardEvents()
        # keys is a dict, so need to check key exists
        if ord("d") in keys and keys[ord("d")] == self._p.KEY_WAS_RELEASED:
            self.debug = True if not hasattr(self, "debug") else not self.debug
        elif ord("r") in keys and keys[ord("r")] == self._p.KEY_WAS_RELEASED:
            self.done = True
        elif ord("z") in keys and keys[ord("z")] == self._p.KEY_WAS_RELEASED:
            while True:
                keys = self._p.getKeyboardEvents()
                if ord("z") in keys and keys[ord("z")] == self._p.KEY_WAS_RELEASED:
                    break
