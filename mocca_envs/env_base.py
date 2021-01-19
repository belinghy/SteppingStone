import datetime

import gym
import gym.utils.seeding
import numpy as np
import pybullet

from mocca_envs.bullet_utils import BulletClient, Camera, SinglePlayerStadiumScene


class EnvBase(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}
    _render_width = 320 * 3
    _render_height = 240 * 3

    def __init__(
        self,
        robot_class,
        robot_kwargs={},
        render=False,
        remove_ground=False,
        use_egl=False,
        use_ffmpeg=False,
        **kwargs
    ):
        self.robot_kwargs = robot_kwargs
        self.robot_class = robot_class

        self.is_rendered = render
        self.remove_ground = remove_ground
        self.use_egl = use_egl
        self.use_ffmpeg = use_ffmpeg

        self.scene = None
        self.physics_client_id = -1
        self.owns_physics_client = 0
        self.state_id = -1

        self.metadata["video.frames_per_second"] = int(1 / self.control_step)

        self.seed()
        self.initialize_scene_and_robot()

    def close(self):
        if self.owns_physics_client and self.physics_client_id >= 0:
            self._p.disconnect()
        self.physics_client_id = -1

    def initialize_scene_and_robot(self):

        self.owns_physics_client = True

        bc_mode = pybullet.GUI if self.is_rendered else pybullet.DIRECT
        render_fps = 1 / self.control_step * self.llc_frame_skip
        self._p = BulletClient(bc_mode, use_ffmpeg=self.use_ffmpeg, fps=render_fps)

        if self.is_rendered or self.use_egl:
            self.camera = Camera(self._p, render_fps, use_egl=self.use_egl)
            if hasattr(self, "create_target"):
                self.create_target()

        if self.use_egl:
            import pkgutil

            egl = pkgutil.get_loader("eglRenderer")
            self.egl = self._p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

        self.physics_client_id = self._p._client

        pc = self._p
        pc.configureDebugVisualizer(pc.COV_ENABLE_RENDERING, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_GUI, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        pc.configureDebugVisualizer(pc.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

        self.scene = SinglePlayerStadiumScene(
            self._p,
            gravity=9.8,
            timestep=self.control_step / self.llc_frame_skip / self.sim_frame_skip,
            frame_skip=self.sim_frame_skip,
        )
        self.scene.initialize(self.remove_ground)

        # Create floor
        if hasattr(self.scene, "ground_plane_mjcf"):
            self.ground_ids = {(self.scene.ground_plane_mjcf[0], -1)}

        # Create robot object
        self.robot = self.robot_class(self._p, **self.robot_kwargs)
        self.robot.initialize()
        self.robot.np_random = self.np_random

        # Create terrain
        if hasattr(self, "create_terrain"):
            self.create_terrain()

        pc.configureDebugVisualizer(pc.COV_ENABLE_RENDERING, int(self.is_rendered))

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

    def get_env_param(self, param_name, default):
        return getattr(self, param_name, default)

    def render(self, mode="human"):
        # Taken care of by pybullet
        if not self.is_rendered:
            self.is_rendered = True
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
            renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
        )
        rgb_array = np.array(px)
        rgb_array = np.reshape(
            np.array(px), (self._render_height, self._render_width, -1)
        )
        rgb_array = rgb_array[:, :, :3]
        return rgb_array.astype(np.uint8)

    def reset(self):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, a):
        raise NotImplementedError

    def _handle_keyboard(self, keys=None, callback=None):
        if keys is None:
            keys = self._p.getKeyboardEvents()

        RELEASED = self._p.KEY_WAS_RELEASED
        self.keypress_status = keys

        # keys is a dict, so need to check key exists
        if keys.get(ord("d")) == RELEASED:
            self.debug = True if not hasattr(self, "debug") else not self.debug
        elif keys.get(ord("r")) == RELEASED:
            self.done = True
        elif keys.get(self._p.B3G_F1) == RELEASED:
            from imageio import imwrite

            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            imwrite("{}.png".format(now), self.camera.dump_rgb_array())
        elif keys.get(pybullet.B3G_F2) == RELEASED:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self._p.startStateLogging(
                self._p.STATE_LOGGING_VIDEO_MP4, "{}.mp4".format(now)
            )
        elif keys.get(ord(" ")) == RELEASED:
            self._p.configureDebugVisualizer(
                self._p.COV_ENABLE_SINGLE_STEP_RENDERING, 0
            )
            while True:
                keys = self._p.getKeyboardEvents()
                if keys.get(ord(" ")) == RELEASED:
                    break
        else:
            if callback is not None:
                callback(keys)
