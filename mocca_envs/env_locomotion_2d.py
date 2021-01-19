import gym
import numpy as np

from mocca_envs.bullet_objects import VSphere
from mocca_envs.env_base import EnvBase
from mocca_envs.robots import Crab2D, Walker2D


class Walker2DCustomEnv(EnvBase):

    control_step = 1 / 60
    llc_frame_skip = 1
    sim_frame_skip = 4

    robot_class = Walker2D
    termination_height = 0.5
    robot_random_start = True
    robot_init_position = [0, 0, 1.2]
    robot_init_velocity = None

    def __init__(self, **kwargs):
        # initialize simulator, robot, and ground
        super().__init__(self.robot_class, **kwargs)

        # used for calculating energy penalty
        self.electricity_cost = 2.0
        self.stall_torque_cost = 0.1
        self.joints_at_limit_cost = 0.1

        # goal is to walk as far forward as possible
        # +x axis is forward, set to some large value
        self.walk_target = np.array([1000, 0, 0])

        # Observation space is just robot's state, see robot.calc_state()
        high = np.inf * np.ones(self.robot.observation_space.shape[0])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # Action space is for the robot
        self.action_space = self.robot.action_space

    def reset(self):
        self.done = False

        # restoreState resets simulation to a saved checkpoint
        # including robot position and pose, and all other objects
        if self.state_id >= 0:
            self._p.restoreState(self.state_id)

        # set robot's initial pose
        # randomness can help with local minima
        self.robot_state = self.robot.reset(
            random_pose=self.robot_random_start,
            pos=self.robot_init_position,
            vel=self.robot_init_velocity,
        )

        # make camera point to robot's root joint
        if self.is_rendered or self.use_egl:
            self.camera.lookat(self.robot.body_xyz)

        # calculate distance to the target
        # needs to be done every time robot or target position is manually moved
        # or else the reward will not be correct
        self.calc_potential()

        # save a checkpoint after first environment reset
        if not self.state_id >= 0:
            self.state_id = self._p.saveState()

        # environment's observation space is just robot's state
        # see robot.calc_state() for what's included in the state
        return self.robot_state

    def step(self, action):
        # set torque for each joint
        # action is normalized to between [-1, 1], wrt max. torque for each joint
        # control frequency is 60Hz, i.e. everytime this function is called, 1/60 seconds has passed in simulation
        self.robot.apply_action(action)

        # step simulation forward after torque is set
        # simulation period is `control_step / sim_frame_skip`, or 240Hz in this case
        # increase sim_frame_skip to make simulation more stable, if necessary
        self.scene.global_step()

        # recalculate robot state after simulation has stepped forward
        # ground_ids is passed to calculate contact state
        self.robot_state = self.robot.calc_state(self.ground_ids)

        # calculate rewards and if episode should be terminated
        self.calc_env_state(action)

        reward = self.progress - self.energy_penalty
        reward += self.tall_bonus - self.joints_penalty

        # for rendering only, in the pybullet gui, press
        # <space> to pause, 'r' to reset, etc
        if self.is_rendered or self.use_egl:
            self._handle_keyboard()
            self.camera.track(pos=self.robot.body_xyz)

        # step() should return observation, reward, done (boolean), and info (dict)
        # info can be anything, some people return individual reward components
        # like, {"progress": self.progress, "energy": self.energy_penalty, ...}
        return self.robot_state, reward, self.done, {}

    def calc_potential(self):

        walk_target_delta = self.walk_target - self.robot.body_xyz

        self.distance_to_target = (
            walk_target_delta[0] ** 2 + walk_target_delta[1] ** 2
        ) ** (1 / 2)

        # reward is sum of progress, scaling by dt here makes sum equal to distance travelled
        self.linear_potential = -self.distance_to_target / self.scene.dt

    def calc_env_state(self, action):
        # in case if neural net explodes
        if not np.isfinite(self.robot_state).all():
            print("~INF~", self.robot_state)
            self.done = True

        # calculate rewards
        # main reward is progress, i.e. how much closer we're towards target
        old_linear_potential = self.linear_potential
        self.calc_potential()
        linear_progress = self.linear_potential - old_linear_potential
        self.progress = linear_progress

        # standard energy penalty based on applied action
        self.energy_penalty = self.electricity_cost * float(
            np.abs(action * self.robot.joint_speeds).mean()
        )
        self.energy_penalty += self.stall_torque_cost * float(np.square(action).mean())

        # penalize joints near maximum range of motion
        # helps prevent getting stuck in local minima
        # also pybullet doesn't enforce joint limits, so kind of necessary
        self.joints_penalty = float(
            self.joints_at_limit_cost * self.robot.joints_at_limit
        )

        # tall bonus encourages robot to stay upright
        self.tall_bonus = 1.0 if self.robot_state[0] > self.termination_height else -1.0

        # since getting up can be hard to learn
        # terminate the episode when robot falls
        self.done = self.done or self.tall_bonus < 0


class Crab2DCustomEnv(Walker2DCustomEnv):

    # same implementation as Walker2D
    # only changing the robot class and robot specific settings
    robot_class = Crab2D
    termination_height = 0.5
    robot_init_position = [0, 0, 1.2]