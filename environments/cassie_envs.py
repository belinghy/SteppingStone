import gym
import numpy as np

from environments.env_base import EnvBase
from environments.robots import Cassie


class CassieEnv(EnvBase):

    control_step = 1 / 30
    llc_frame_skip = 50
    sim_frame_skip = 1

    ## PD gains:
    kp = np.array([100, 100, 88, 96, 98, 98, 50, 100, 100, 88, 96, 98, 98, 50])
    kd = kp / 15
    kd[[6, 13]] /= 10

    def __init__(self, render=False):
        super(CassieEnv, self).__init__(Cassie, render)

        high = np.inf * np.ones(self.robot.observation_space.shape[0] + 2)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = self.robot.action_space

    def calc_potential(self, body_xyz):
        target_dist = (
            (self.walk_target[1] - body_xyz[1]) ** 2
            + (self.walk_target[0] - body_xyz[0]) ** 2
        ) ** (1 / 2)

        return -target_dist / self.control_step

    def reset(self):
        self.done = False
        self.walk_target = np.array([1000.0, 0.0, 0.0])

        self._p.restoreState(self.state_id)

        self.robot_state = self.robot.reset()

        if self.is_render:
            self.camera.lookat(self.robot.body_xyz)

        self.potential = self.calc_potential(self.robot.body_xyz)
        delta = self.walk_target - self.robot.body_xyz
        walk_target_theta = np.arctan2(delta[1], delta[0])
        delta_theta = walk_target_theta - self.robot.body_rpy[2]
        rot = np.array(
            [
                [np.cos(-delta_theta), -np.sin(-delta_theta), 0.0],
                [np.sin(-delta_theta), np.cos(-delta_theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        target = np.matmul(rot, self.walk_target)
        state = np.concatenate((self.robot_state, target[0:2]))
        return state

    def pd_control(self, target_angles, target_speeds):
        curr_angles = self.robot.to_radians(self.robot.joint_angles)
        curr_speeds = self.robot.joint_speeds

        perror = target_angles - curr_angles
        verror = np.clip(target_speeds - curr_speeds, -5, 5)

        # print(', '.join(['%3.0f' % s for s in perror]), end='   |   ')
        # print(', '.join(['%4.0f' % s for s in verror]))
        # import time
        # time.sleep(0.1)

        return self.kp * perror + self.kd * verror

    def step(self, a):
        target_angles = np.zeros(14)
        ## `knee_to_shin` and `ankle_joint` joints (both sides) do not have a motor
        ## we don't know how to set the constraints for them so we're using PD with fixed target instead
        target_angles[[0, 1, 2, 3, 6, 7, 8, 9, 10, 13]] = a / 2
        # target_angles = self.robot.to_radians(target_angles)
        target_angles += self.robot.base_joint_angles
        target_angles[4] = 0
        target_angles[5] = -target_angles[3] + 0.227  # -q_3 + 13 deg
        target_angles[11] = 0
        target_angles[12] = -target_angles[10] + 0.227  # -q_10 + 13 deg

        for _ in range(self.llc_frame_skip):
            target_speeds = target_angles * 0
            torque = self.pd_control(target_angles, target_speeds)
            self.robot.apply_action(torque)
            self.scene.global_step()
            robot_state = self.robot.calc_state()

        done = False
        if not np.isfinite(robot_state).all():
            print("~INF~", robot_state)
            done = True

        old_potential = self.potential
        self.potential = self.calc_potential(self.robot.body_xyz)
        progress = self.potential - old_potential

        tall_bonus = (
            2.0
            if self.robot.body_xyz[2] - np.min(self.robot.feet_xyz[:, 2]) > 0.6
            else -1.0
        )

        if tall_bonus < 0:
            done = True

        if self.is_render:
            self.camera.track(pos=self.robot.body_xyz)
            self._handle_keyboard()
            done = done or self.done

        self.rewards = [tall_bonus, progress]

        delta = self.walk_target - self.robot.body_xyz
        walk_target_theta = np.arctan2(delta[1], delta[0])
        delta_theta = walk_target_theta - self.robot.body_rpy[2]

        rot = np.array(
            [
                [np.cos(-delta_theta), -np.sin(-delta_theta), 0.0],
                [np.sin(-delta_theta), np.cos(-delta_theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        target = np.matmul(rot, self.walk_target)
        state = np.concatenate((robot_state, target[0:2]))
        return state, sum(self.rewards), done, {}
