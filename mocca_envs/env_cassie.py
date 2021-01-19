import os
import gym
import copy
import pickle
import numpy as np

from mocca_envs.env_base import EnvBase
from mocca_envs import current_dir

from .loadstep import CassieTrajectory


class Cassie:
    model_path = os.path.join(
        current_dir, "data", "robots", "cassie", "urdf", "cassie_collide.urdf"
    )
    base_position = (0.0, 0.0, 1.085)
    base_orientation = (0.0, 0.0, 0.0, 1.0)

    base_joint_angles = [
        # left:
        0.035615837,
        -0.01348790,
        0.391940848,
        -0.95086160,
        -0.08376049,
        1.305643634,
        -1.61174064,
        # right:
        0.035615837,
        -0.01348790,
        0.391940848,
        -0.95086160,
        -0.08376049,
        1.305643634,
        -1.61174064,
    ]

    rod_joint_angles = [-0.8967891835, 0.063947468, -0.8967891835, -0.063947468]

    power_coef = {
        "hip_abduction_left": 112.5,
        "hip_rotation_left": 112.5,
        "hip_flexion_left": 195.2,
        "knee_joint_left": 195.2,
        "knee_to_shin_right": 200,  # not sure how to set, using PD instead of a constraint
        "ankle_joint_right": 200,  # not sure how to set, using PD instead of a constraint
        "toe_joint_left": 45.0,
        "hip_abduction_right": 112.5,
        "hip_rotation_right": 112.5,
        "hip_flexion_right": 195.2,
        "knee_joint_right": 195.2,
        "knee_to_shin_left": 200,  # not sure how to set, using PD instead of a constraint
        "ankle_joint_left": 200,  # not sure how to set, using PD instead of a constraint
        "toe_joint_right": 45.0,
    }
    joint_damping = [1, 1, 1, 1, 0.1, 0, 1, 1, 1, 1, 1, 0.1, 0, 1]

    powered_joint_inds = [0, 1, 2, 3, 6, 7, 8, 9, 10, 13]
    spring_joint_inds = [4, 11]

    def __init__(self, bc, power=1.0):
        self._p = bc
        self.base_power = power
        self.rod_joints = {}

        self.parts = None
        self.jdict = None
        self.object_id = None
        self.ordered_joints = None
        self.robot_body = None
        self.foot_names = ["right_toe", "left_toe"]

        action_dim = 10
        high = np.ones(action_dim)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)
        state_dim = (action_dim + 4) * 2 + 6
        high = np.inf * np.ones(state_dim)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def load_robot_model(self):
        flags = (
            self._p.URDF_USE_SELF_COLLISION
            | self._p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
            | self._p.URDF_USE_INERTIA_FROM_FILE
        )
        self.object_id = (
            self._p.loadURDF(
                self.model_path,
                basePosition=self.base_position,
                baseOrientation=self.base_orientation,
                useFixedBase=False,
                flags=flags,
            ),
        )

        self.parse_joints_and_links(self.object_id)
        self.powered_joints = np.array(self.ordered_joints)[
            self.powered_joint_inds
        ].tolist()
        self.spring_joints = np.array(self.ordered_joints)[
            self.spring_joint_inds
        ].tolist()

        # Set Initial pose
        self._p.resetBasePositionAndOrientation(
            self.object_id[0], posObj=self.base_position, ornObj=self.base_orientation
        )

        self.reset_joint_positions(
            self.base_joint_angles, [0 for _ in self.base_joint_angles]
        )

        self._p.createConstraint(
            self.object_id[self.parts["left_tarsus"].bodyIndex],
            self.parts["left_tarsus"].bodyPartIndex,
            self.object_id[self.parts["left_achilles_rod"].bodyIndex],
            self.parts["left_achilles_rod"].bodyPartIndex,
            jointType=self._p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[-0.22735404, 0.05761813, 0.00711836],
            childFramePosition=[0.254001, 0, 0],
            parentFrameOrientation=[0.000000, 0.000000, 0.000000, 1.000000],
            childFrameOrientation=[0.000000, 0.000000, 0.000000, 1.000000],
        )
        self._p.createConstraint(
            self.object_id[self.parts["right_tarsus"].bodyIndex],
            self.parts["right_tarsus"].bodyPartIndex,
            self.object_id[self.parts["right_achilles_rod"].bodyIndex],
            self.parts["right_achilles_rod"].bodyPartIndex,
            jointType=self._p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[-0.22735404, 0.05761813, -0.00711836],
            childFramePosition=[0.254001, 0, 0],
            parentFrameOrientation=[0.000000, 0.000000, 0.000000, 1.000000],
            childFrameOrientation=[0.000000, 0.000000, 0.000000, 1.000000],
        )
        for part_name in [
            "left_achilles_rod",
            "left_achilles_rod_y",
            "right_achilles_rod",
            "right_achilles_rod_y",
        ]:
            self._p.setCollisionFilterGroupMask(
                self.object_id[self.parts[part_name].bodyIndex],
                self.parts[part_name].bodyPartIndex,
                0,
                0,
            )

    def reset_joint_positions(self, positions, velocities):
        for j, q, v in zip(self.ordered_joints, positions, velocities):
            j.reset_current_position(q, v)

    def parse_joints_and_links(self, bodies):
        self.parts = {}
        self.jdict = {}
        self.ordered_joints = []
        bodies = [bodies] if np.isscalar(bodies) else bodies

        # We will overwrite this if a "pelvis" is found
        self.robot_body = BodyPart(self._p, "root", bodies, 0, -1)

        for i in range(len(bodies)):
            for j in range(self._p.getNumJoints(bodies[i])):
                self._p.setJointMotorControl2(
                    bodies[i],
                    j,
                    self._p.POSITION_CONTROL,
                    positionGain=0.1,
                    velocityGain=0.1,
                    force=0,
                )
                jointInfo = self._p.getJointInfo(bodies[i], j)
                joint_name = jointInfo[1]
                part_name = jointInfo[12]

                joint_name = joint_name.decode("utf8")
                part_name = part_name.decode("utf8")

                self.parts[part_name] = BodyPart(self._p, part_name, bodies, i, j)

                if part_name == "pelvis":
                    self.robot_body = self.parts[part_name]

                joint = Joint(self._p, joint_name, bodies, i, j, torque_limit=0)

                if "achilles" in joint_name:
                    joint.reset_position(self.rod_joint_angles[len(self.rod_joints)], 0)
                    self.rod_joints[joint_name] = joint

                if joint_name[:5] != "fixed":
                    joint.set_torque_limit(
                        self.base_power * self.power_coef[joint_name]
                    )
                    self.jdict[joint_name] = joint
                    self._p.changeDynamics(
                        bodies[i],
                        j,
                        jointDamping=self.joint_damping[len(self.ordered_joints)],
                    )
                    self.ordered_joints.append(self.jdict[joint_name])

    def make_robot_utils(self):
        # Make utility functions for converting from normalized to radians and vice versa
        # Inputs are thetas (normalized angles) directly from observation
        # weight is the range of motion, thigh can move 90 degrees, etc
        weight = np.array([j.upperLimit - j.lowerLimit for j in self.ordered_joints])
        # bias is the angle corresponding to -1
        bias = np.array([j.lowerLimit for j in self.ordered_joints])
        self.to_radians = lambda thetas: weight * (thetas + 1) / 2 + bias
        self.to_normalized = lambda angles: 2 * (angles - bias) / weight - 1

    def initialize(self):
        self.load_robot_model()
        self.make_robot_utils()

    def reset(self):
        self.feet = [self.parts[f] for f in self.foot_names]
        self.feet_xyz = np.zeros((len(self.foot_names), 3))
        self.initial_z = None
        state = self.calc_state()
        return state

    def apply_action(self, a):
        assert np.isfinite(a).all()
        # for n, j in enumerate(self.ordered_joints):
        for n, j in enumerate(self.powered_joints + self.spring_joints):
            # j.set_position(self.base_joint_angles[n])
            j.set_motor_torque(float(np.clip(a[n], -j.torque_limit, j.torque_limit)))

        # self.ordered_joints[4].set_position(0)
        # self.ordered_joints[11].set_position(0)
        # angles = self.to_radians(self.joint_angles)
        # self.ordered_joints[5].set_position(-angles[3] + 0.227)  # -q_3 + 13 deg
        # self.ordered_joints[12].set_position(-angles[10] + 0.227)  # -q_10 + 13 deg

    def calc_state(self):
        j = np.array(
            [j.current_relative_position() for j in self.ordered_joints],
            dtype=np.float32,
        )

        self.joint_angles = j[:, 0]
        self.rad_joint_angles = self.to_radians(self.joint_angles)
        self.joint_speeds = j[:, 1]
        self.joints_at_limit = np.count_nonzero(np.abs(self.joint_angles) > 0.99)

        body_pose = self.robot_body.pose()
        self.body_xyz = body_pose.xyz()
        self.body_angular_speed = self.robot_body.angular_speed()

        z = self.body_xyz[2]
        if self.initial_z is None:
            self.initial_z = z

        self.body_rpy = body_pose.rpy()
        roll, pitch, yaw = self.body_rpy

        rot = np.array(
            [
                [np.cos(-yaw), -np.sin(-yaw), 0],
                [np.sin(-yaw), np.cos(-yaw), 0],
                [0, 0, 1],
            ]
        )
        self.body_velocity = np.dot(rot, self.robot_body.speed())

        vx, vy, vz = self.body_velocity
        more = np.array([z - self.initial_z, vx, vy, vz, roll, pitch], dtype=np.float32)

        for i, p in enumerate(self.feet):
            # Need this to calculate done, might as well calculate it
            self.feet_xyz[i] = p.pose().xyz()

        return np.concatenate((more, self.joint_angles, self.joint_speeds))


class Cassie2D(Cassie):
    model_path = os.path.join(
        current_dir, "data", "cassie", "urdf", "cassie_collide_2d.urdf"
    )


class CassieEnv(EnvBase):

    control_step = 0.03
    llc_frame_skip = 50
    sim_frame_skip = 1

    ## PD gains:
    kp = np.array(
        [
            ## left:
            100,
            100,
            88,
            96,
            # 500,
            # 450,
            50,
            ## right:
            100,
            100,
            88,
            96,
            # 500,
            # 450,
            50,
            ## knee_to_shin springs:
            400,
            400,
        ]
    )
    kp = kp / 1.9
    # kp[[4, 9]] /= 2
    kd = kp / 10

    jvel_alpha = min(10 / llc_frame_skip, 1)

    initial_velocity = [0, 0, 0]

    def __init__(
        self,
        render=False,
        planar=False,
        power_coef=1.0,
        residual_control=True,
        rsi=True,
    ):
        """
        :params render: enables GUI rendering
        :params planar: constrains the robot movement to a 2D plane (rather than the full 3D motion)
        :params power_coef: multiplying factor that determines the torque limit
        :params residual_control: if set to `True`, will add `self.base_angles` to the action
        :params rsi: if set to `True`, will do Random State Initialization [https://www.cs.ubc.ca/~van/papers/2018-TOG-deepMimic/]
        """
        self.planar = planar
        self.residual_control = residual_control
        self.rsi = rsi
        robot_class = Cassie2D if planar else Cassie
        super(CassieEnv, self).__init__(robot_class, render, power=power_coef)

        high = np.inf * np.ones(self.robot.observation_space.shape[0] + 2)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = self.robot.action_space

    def calc_potential(self, body_xyz):
        target_dist = (
            (self.walk_target[1] - body_xyz[1]) ** 2
            + (self.walk_target[0] - body_xyz[0]) ** 2
        ) ** (1 / 2)

        return -target_dist / self.control_step

    def resetJoints(self):
        self.robot.reset_joint_positions(self.base_angles(), self.base_velocities())
        self.jvel = self.base_velocities()

    def mocap_time(self):
        return self.istep * self.control_step / self.llc_frame_skip

    def reset(self, istep=0):
        self.done = False
        self.istep = istep if self.rsi else 0
        self.walk_target = np.array([1000.0, 0.0, 0.0])

        self._p.restoreState(self.state_id)
        self.resetJoints()
        self.robot.robot_body.reset_velocity(self.initial_velocity)

        self.robot_state = self.robot.reset()

        if self.is_rendered:
            self.camera.lookat(self.robot.body_xyz)

        self.potential = self.calc_potential(self.robot.body_xyz)
        return self.get_obs(self.robot_state)

    def pd_control(self, target_angles, target_speeds):
        self.istep += 1
        joint_inds = self.robot.powered_joint_inds + self.robot.spring_joint_inds
        curr_angles = self.robot.rad_joint_angles[joint_inds]
        curr_speeds = self.jvel[joint_inds]
        # curr_speeds = self.robot.joint_speeds[joint_inds]

        perror = target_angles - curr_angles
        verror = np.clip(target_speeds - curr_speeds, -5, 5)

        # print(", ".join(["%4.1f" % s for s in perror]), end="   |   ")
        # print(", ".join(["%4.1f" % s for s in verror]))

        return self.kp * perror + self.kd * verror

    def base_angles(self):
        return np.array(self.robot.base_joint_angles)

    def base_velocities(self):
        return np.array([0 for _ in self.robot.base_joint_angles])

    def compute_rewards(self, action, torques):
        old_potential = self.potential
        self.potential = self.calc_potential(self.robot.body_xyz)
        progress = self.potential - old_potential

        tall_bonus = (
            2.0
            if self.robot.body_xyz[2] - np.min(self.robot.feet_xyz[:, 2]) > 0.6
            else -1.0
        )

        dead = tall_bonus < 0

        return dead, {"AliveRew": tall_bonus, "ProgressRew": progress}

    def get_obs(self, robot_state):
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
        return state

    def step(self, a):
        if self.residual_control:
            ## `knee_to_shin` and `ankle_joint` joints (both sides) do not have a driving motor
            target_angles = self.base_angles()[self.robot.powered_joint_inds]
        else:
            target_angles = 0

        target_angles += a
        target_angles = np.concatenate(
            [target_angles, [0 for _ in self.robot.spring_joint_inds]]
        )

        torques = []
        done = False

        jpos = self.robot.rad_joint_angles

        for _ in range(self.llc_frame_skip):
            self.jvel = (
                1 - self.jvel_alpha
            ) * self.jvel + self.jvel_alpha * self.robot.joint_speeds
            target_speeds = target_angles * 0
            torque = self.pd_control(target_angles, target_speeds)
            torques.append(torque)
            self.robot.apply_action(torque)
            self.scene.global_step()
            robot_state = self.robot.calc_state()
            if self.is_rendered:
                self.camera.track(
                    pos=self.robot.body_xyz, smooth_coef=np.array([0.1, 0.01, 0.01])
                )
                self._handle_keyboard()
                done = done or self.done

        self.jpos = self.robot.rad_joint_angles
        self.jvel = np.subtract(self.jpos, jpos) / self.control_step

        # TODO: how to get more stable jvel for using in PD?

        if not np.isfinite(robot_state).all():
            print("~INF~", robot_state)
            done = True

        dead, rewards = self.compute_rewards(a, torques)
        done = done or dead

        return self.get_obs(robot_state), sum(rewards.values()), done, rewards


class CassieMocapRewEnv(CassieEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = {
            "SpeedRew": 0.1,
            "CoMRew": 0.02 if self.planar else 0.05,
            "OrientationRew": 0 if self.planar else 0.05,
            "AngularSpeedRew": 0.1,
        }
        wleft = 1 - sum(self.weights.values())
        self.weights["JPosRew"] = wleft / 5 * 4
        self.weights["JVelRew"] = wleft / 5

    def compute_rewards(self, action, torques):
        dead, rewards = super(CassieMocapRewEnv, self).compute_rewards(action, torques)
        # TODO: use self.initial_velocity
        vel_error = (self.robot.body_velocity[0] - 0.8) ** 2

        kin_jpos = self.base_angles()
        kin_jvel = self.base_velocities()

        joint_penalty = np.sqrt(
            np.sum((kin_jpos - self.jpos)[self.robot.powered_joint_inds] ** 2)
        )
        jvel_penalty = np.sqrt(
            np.sum((kin_jvel - self.jvel)[self.robot.powered_joint_inds] ** 2)
        )
        # print(kin_jvels, dyn_jvels)
        orientation_penalty = np.sum(np.power(self.robot.body_rpy, 2))
        angular_speed_penalty = np.sum(np.power(self.robot.body_angular_speed, 2))
        com_penalty = np.sum(
            np.subtract(self.robot.body_xyz[1:], self.robot.base_position[1:]) ** 2
        )

        self.robot.robot_body.angular_speed()

        rewards = {}
        # rewards["EnergyUseRew"] = np.mean(np.power(torques, 2))
        # rewards["DeviationRew"] = np.mean(np.power(action, 2))
        rewards["SpeedRew"] = np.exp(-4 * vel_error)
        rewards["JPosRew"] = np.exp(-4 * joint_penalty)
        rewards["JVelRew"] = np.exp(-0.4 * jvel_penalty)
        rewards["OrientationRew"] = np.exp(-4 * orientation_penalty)
        rewards["AngularSpeedRew"] = np.exp(-4 * angular_speed_penalty)
        rewards["CoMRew"] = np.exp(-4 * com_penalty)

        weighted_rewards = {k: v * self.weights[k] for k, v in rewards.items()}

        return dead, weighted_rewards


class CassieMoccaEnv(CassieMocapRewEnv):
    initial_velocity = [0.8, 0, 0]

    mirror_indices = {
        "neg_obs_inds": [
            # y
            0,
            # quat x
            3,
            # quat z
            5,
            # y velocity
            21,
            # x angular speed
            23,
            # z angular speed
            25,
        ],
        "sideneg_obs_inds": [
            # left abduction
            6,
            # left yaw
            7,
            # left abduction speed
            26,
            # left yaw speed
            27,
        ],
        "com_obs_inds": [1, 2, 4, 20, 22, 24],
        "left_obs_inds": list(range(6, 13)) + list(range(26, 33)),
        "right_obs_inds": list(range(13, 20)) + list(range(33, 40)),
        # action:
        "com_act_inds": [],
        "left_act_inds": list(range(0, 5)),
        "right_act_inds": list(range(5, 10)),
        "neg_act_inds": [],
        "sideneg_act_inds": [0, 1],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        high = np.inf * np.ones(40)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.traj = CassieTrajectory()

    def reset(self, istep=None):
        if istep is None:
            istep = self.np_random.randint(0, 10000)  # 3971 # 5152
        return super().reset(istep=istep)

    def resetJoints(self):
        super().resetJoints()
        t = self.mocap_time()
        rod_joint_names = [
            "fixed_right_achilles_rod_joint_z",
            "fixed_right_achilles_rod_joint_y",
            "fixed_left_achilles_rod_joint_z",
            "fixed_left_achilles_rod_joint_y",
        ]
        for jname, angle in zip(rod_joint_names, self.traj.rod_joint_angles(t)):
            self.robot.rod_joints[jname].reset_current_position(angle, 0)

    def base_angles(self):
        return self.traj.joint_angles(self.mocap_time())

    def base_velocities(self):
        return self.traj.joint_speeds(self.mocap_time())

    def get_obs(self, _):
        quaternion = self._p.getQuaternionFromEuler(self.robot.body_rpy)

        return np.concatenate(
            [
                ## pos
                self.robot.body_xyz[1:],
                [quaternion[-1]],
                quaternion[:-1],
                # [1, 0, 0, 0],
                self.robot.rad_joint_angles,
                ## vel
                self.robot.body_velocity,
                self.robot.body_angular_speed,
                # [0, 0, 0],
                self.jvel,
            ]
        )


class CassiePhaseMoccaEnv(CassieMoccaEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        high = np.inf * np.ones(42)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.mirror_indices = copy.deepcopy(self.mirror_indices)
        self.mirror_indices["left_obs_inds"] += [40]
        self.mirror_indices["right_obs_inds"] += [41]

    def get_obs(self, robot_state):
        dyn_state = super().get_obs(robot_state)

        phase_l = (self.mocap_time() / self.traj.max_time()) % 1
        phase_r = (phase_l + 0.5) % 1

        return np.concatenate([dyn_state, [phase_l, phase_r]])


class CassiePhaseMirrorEnv(CassiePhaseMoccaEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neg_inds = (
            self.mirror_indices["neg_obs_inds"]
            + self.mirror_indices["sideneg_obs_inds"]
        )
        self.lr_inds = (
            self.mirror_indices["left_obs_inds"] + self.mirror_indices["right_obs_inds"]
        )
        self.rl_inds = (
            self.mirror_indices["right_obs_inds"] + self.mirror_indices["left_obs_inds"]
        )

    def get_obs(self, robot_state):
        obs = super().get_obs(robot_state)
        phase = obs[-2]
        if phase > 0.5:
            obs[self.lr_inds] = obs[self.rl_inds]
            obs[self.neg_inds] *= -1
        return obs


# class CassieOSUEnv(CassieMoccaEnv):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         high = np.inf * np.ones(80)
#         self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

#         self.mirror_indices = copy.deepcopy(self.mirror_indices)

#         for key in self.mirror_indices.keys():
#             if "obs" in key:
#                 self.mirror_indices[key] = np.concatenate(
#                     [self.mirror_indices[key], np.array(self.mirror_indices[key]) + 40]
#                 ).tolist()

#     def get_obs(self, robot_state):
#         t = (self.istep + 1) * self.control_step / self.llc_frame_skip
#         kin_state = self.traj.state(t)
#         kin_state[0] = 0  # fix desired y to be exactly 0

#         dyn_state = super().get_obs(robot_state)

#         return np.concatenate([dyn_state, kin_state])
