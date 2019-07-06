import os

current_dir = os.path.dirname(os.path.realpath(__file__))

import gym
import numpy as np

from environments.bullet_utils import BodyPart, Joint

DEG2RAD = np.pi / 180


class Cassie:
    model_path = os.path.join(
        current_dir, "data", "cassie", "urdf", "cassie_collide.urdf"
    )
    base_position = (0.0, 0.0, 1.057)
    base_orientation = (0.0, 0.0, 0.0, 1.0)

    base_joint_angles = [
        3.56592126e-02,
        -1.30443918e-02,
        3.55475724e-01,
        -9.15456176e-01,
        -8.37604925e-02,
        1.37208855e00,
        -1.61174064e00,
        3.56592126e-02,
        -1.30443918e-02,
        3.55475724e-01,
        -9.15456176e-01,
        -8.37604925e-02,
        1.37208855e00,
        -1.61174064e00,
    ]

    power_coef = {
        "hip_abduction_left": 112.5,
        "hip_rotation_left": 112.5,
        "hip_flexion_left": 195.2,
        "knee_joint_left": 195.2,
        "knee_to_shin_right": 100,  # not sure how to set, using PD instead of a constraint
        "ankle_joint_right": 100,  # not sure how to set, using PD instead of a constraint
        "toe_joint_left": 45.0,
        "hip_abduction_right": 112.5,
        "hip_rotation_right": 112.5,
        "hip_flexion_right": 195.2,
        "knee_joint_right": 195.2,
        "knee_to_shin_left": 100,  # not sure how to set, using PD instead of a constraint
        "ankle_joint_left": 100,  # not sure how to set, using PD instead of a constraint
        "toe_joint_right": 45.0,
    }

    def __init__(self, bc, power=1.0):
        self._p = bc
        self.power = power

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
        self.object_id = (
            self._p.loadURDF(
                self.model_path,
                basePosition=self.base_position,
                baseOrientation=self.base_orientation,
                useFixedBase=False,
            ),
        )

        self.parse_joints_and_links(self.object_id)

        self.torque_limits = self.power * np.array(
            [self.power_coef[j.joint_name] for j in self.ordered_joints]
        )

        # Set Initial pose
        self._p.resetBasePositionAndOrientation(
            self.object_id[0], posObj=self.base_position, ornObj=self.base_orientation
        )

        for j, q in zip(self.ordered_joints, self.base_joint_angles):
            j.reset_current_position(q, 0)

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

                if joint_name[:5] != "fixed":
                    self.jdict[joint_name] = Joint(self._p, joint_name, bodies, i, j)
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
        x = np.clip(a, -self.torque_limits, self.torque_limits)
        for n, j in enumerate(self.ordered_joints):
            # j.set_position(self.base_joint_angles[n])
            j.set_motor_torque(float(x[n]))

    def calc_state(self):
        j = np.array(
            [j.current_relative_position() for j in self.ordered_joints],
            dtype=np.float32,
        )

        self.joint_angles = j[:, 0]
        self.joint_speeds = j[:, 1]
        self.joints_at_limit = np.count_nonzero(np.abs(self.joint_angles) > 0.99)

        body_pose = self.robot_body.pose()
        self.body_xyz = body_pose.xyz()

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


class WalkerBase:
    def apply_action(self, a):
        assert np.isfinite(a).all()
        x = np.clip(a, -1, 1)
        for n, j in enumerate(self.ordered_joints):
            j.set_motor_torque(self.torque_limits[n] * float(x[n]))

    def calc_state(self, contact_object_ids=None):
        j = np.array(
            [j.current_relative_position() for j in self.ordered_joints],
            dtype=np.float32,
        )

        self.joint_angles = j[:, 0]
        self.joint_speeds = 0.1 * j[:, 1]  # Normalize
        self.joints_at_limit = np.count_nonzero(np.abs(self.joint_angles) > 0.99)

        body_pose = self.robot_body.pose()

        # parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()])
        # self.body_xyz = parts_xyz.mean(axis=0)
        # # pelvis z is more informative than mean z
        # self.body_xyz[2] = body_pose.xyz()[2]

        # Faster if we don't use true CoM
        self.body_xyz = body_pose.xyz()

        if self.initial_z is None:
            self.initial_z = self.body_xyz[2]

        self.body_rpy = body_pose.rpy()
        roll, pitch, yaw = self.body_rpy

        rot = np.array(
            [
                [np.cos(-yaw), -np.sin(-yaw), 0],
                [np.sin(-yaw), np.cos(-yaw), 0],
                [0, 0, 1],
            ]
        )
        self.body_vel = np.dot(rot, self.robot_body.speed())
        vx, vy, vz = self.body_vel

        more = np.array(
            [self.body_xyz[2] - self.initial_z, vx, vy, vz, roll, pitch],
            dtype=np.float32,
        )

        if contact_object_ids is not None:
            for i, f in enumerate(self.feet):
                self.feet_xyz[i] = f.pose().xyz()
                contact_ids = set((x[2], x[4]) for x in f.contact_list())
                if contact_object_ids & contact_ids:
                    self.feet_contact[i] = 1.0
                else:
                    self.feet_contact[i] = 0.0

        state = np.concatenate(
            (more, self.joint_angles, self.joint_speeds, self.feet_contact)
        )

        return np.clip(state, -5, +5)

    def initialize(self):
        self.load_robot_model()
        self.make_robot_utils()

    def load_robot_model(self, model_path, flags, root_link_name=None):
        self.object_id = self._p.loadMJCF(model_path, flags=flags)

        self.parse_joints_and_links(self.object_id)
        if root_link_name is not None:
            self.robot_body = self.parts[root_link_name]
        else:
            self.robot_body = BodyPart(self._p, "root", self.object_id, 0, -1)

        self.feet = [self.parts[f] for f in self.foot_names]
        self.feet_contact = np.zeros(len(self.foot_names), dtype=np.float32)
        self.feet_xyz = np.zeros((len(self.foot_names), 3))
        self.calc_torque_limits()

    def calc_torque_limits(self):
        self.torque_limits = self.power * np.array(
            [self.power_coef[j.joint_name] for j in self.ordered_joints]
        )

    def make_robot_utils(self):
        # utility functions for converting from normalized to radians and vice versa
        # Inputs are thetas (normalized angles) directly from observation
        # weight is the range of motion, thigh can move 90 degrees, etc
        weight = np.array([j.upperLimit - j.lowerLimit for j in self.ordered_joints])
        # bias is the angle corresponding to -1
        bias = np.array([j.lowerLimit for j in self.ordered_joints])
        self.to_radians = lambda thetas: weight * (thetas + 1) / 2 + bias
        self.to_normalized = lambda angles: 2 * (angles - bias) / weight - 1

    def parse_joints_and_links(self, bodies):
        self.parts = {}
        self.jdict = {}
        self.ordered_joints = []
        bodies = [bodies] if np.isscalar(bodies) else bodies

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

                if joint_name[:6] == "ignore":
                    Joint(self._p, joint_name, bodies, i, j).disable_motor()
                    continue

                if joint_name[:8] != "jointfix":
                    self.jdict[joint_name] = Joint(self._p, joint_name, bodies, i, j)
                    self.ordered_joints.append(self.jdict[joint_name])

    def reset(self):
        self.feet_contact.fill(0.0)
        self.feet_xyz.fill(0.0)
        self.initial_z = None

        robot_state = self.calc_state()
        return robot_state


class Walker3D(WalkerBase):

    foot_names = ["right_foot", "left_foot"]

    power_coef = {
        "abdomen_z": 60,
        "abdomen_y": 80,
        "abdomen_x": 60,
        "right_hip_x": 80,
        "right_hip_z": 60,
        "right_hip_y": 100,
        "right_knee": 90,
        "right_ankle": 60,
        "left_hip_x": 80,
        "left_hip_z": 60,
        "left_hip_y": 100,
        "left_knee": 90,
        "left_ankle": 60,
        "right_shoulder_x": 60,
        "right_shoulder_z": 60,
        "right_shoulder_y": 50,
        "right_elbow": 60,
        "left_shoulder_x": 60,
        "left_shoulder_z": 60,
        "left_shoulder_y": 50,
        "left_elbow": 60,
    }

    def __init__(self, bc):
        self._p = bc
        self.power = 1.0

        self.action_dim = 21
        high = np.ones(self.action_dim)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # globals + angles + speeds + contacts
        self.state_dim = 6 + self.action_dim * 2 + 2
        high = np.inf * np.ones(self.state_dim)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def load_robot_model(self):
        flags = (
            self._p.MJCF_COLORS_FROM_FILE
            | self._p.URDF_USE_SELF_COLLISION
            | self._p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        )
        model_path = os.path.join(current_dir, "data", "custom", "walker3d.xml")
        root_link_name = None

        # Need to call this first to parse body
        super(Walker3D, self).load_robot_model(model_path, flags, root_link_name)

        # T-pose
        self.base_joint_angles = np.zeros(self.action_dim)
        self.base_position = (0, 0, 1.32)
        self.base_orientation = (0, 0, 0, 1)

        # Need this to set pose and mirroring
        # hip_[x,z,y], knee, ankle, shoulder_[x,z,y], elbow
        self._right_joint_indices = np.array(
            [3, 4, 5, 6, 7, 13, 14, 15, 16], dtype=np.int64
        )
        self._left_joint_indices = np.array(
            [8, 9, 10, 11, 12, 17, 18, 19, 20], dtype=np.int64
        )
        # abdomen_[x,z]
        self._negation_joint_indices = np.array([0, 2], dtype=np.int64)

        self._rl = np.concatenate((self._right_joint_indices, self._left_joint_indices))
        self._lr = np.concatenate((self._left_joint_indices, self._right_joint_indices))

    def set_base_pose(self, pose=None):
        self.base_joint_angles[:] = 0  # reset

        if pose == "running_start":
            self.base_joint_angles[[5, 6]] = -np.pi / 8  # Right leg
            self.base_joint_angles[10] = np.pi / 10  # Left leg back
            self.base_joint_angles[[13, 17]] = np.pi / 3  # Shoulder x
            self.base_joint_angles[[14]] = -np.pi / 6  # Right shoulder back
            self.base_joint_angles[[18]] = np.pi / 6  # Left shoulder forward
            self.base_joint_angles[[16, 20]] = np.pi / 3  # Elbow
        elif pose == "sit":
            self.base_joint_angles[[5, 10]] = -np.pi / 2  # hip
            self.base_joint_angles[[6, 11]] = -np.pi / 2  # knee
        elif pose == "squat":
            angle = -20 * DEG2RAD
            self.base_joint_angles[[5, 10]] = -np.pi / 2  # hip
            self.base_joint_angles[[6, 11]] = -np.pi / 2  # knee
            self.base_joint_angles[[7, 12]] = angle  # ankles
            self.base_orientation = self._p.getQuaternionFromEuler([0, -angle, 0])
        elif pose == "crawl":
            self.base_joint_angles[[13, 17]] = np.pi / 2  # shoulder x
            self.base_joint_angles[[14, 18]] = np.pi / 2  # shoulder z
            self.base_joint_angles[[16, 20]] = np.pi / 3  # Elbow
            self.base_joint_angles[[5, 10]] = -np.pi / 2  # hip
            self.base_joint_angles[[6, 11]] = -120 * DEG2RAD  # knee
            self.base_joint_angles[[7, 12]] = -20 * DEG2RAD  # ankles
            self.base_orientation = self._p.getQuaternionFromEuler([0, 90 * DEG2RAD, 0])

    def reset(self, random_pose=True, pos=None, quat=None):

        if random_pose:
            # Mirror initial pose
            if self.np_random.rand() < 0.5:
                self.base_joint_angles[self._rl] = self.base_joint_angles[self._lr]
                self.base_joint_angles[self._negation_joint_indices] *= -1

            # Add small deviations
            ds = self.np_random.uniform(low=-0.1, high=0.1, size=self.action_dim)
            ps = self.to_normalized(self.base_joint_angles + ds)
            ps = self.to_radians(np.clip(ps, -0.95, 0.95))

            for i, j in enumerate(self.ordered_joints):
                j.reset_current_position(ps[i], 0)

        pos = self.base_position if pos is None else pos
        quat = self.base_orientation if quat is None else quat

        self._p.resetBasePositionAndOrientation(
            self.object_id[0], posObj=pos, ornObj=quat
        )

        return super(Walker3D, self).reset()


class Walker2D(WalkerBase):

    foot_names = ["foot", "foot_left"]

    power_coef = {
        "torso_joint": 100,
        "thigh_joint": 100,
        "leg_joint": 100,
        "foot_joint": 50,
        "thigh_left_joint": 100,
        "leg_left_joint": 100,
        "foot_left_joint": 50,
    }

    def __init__(self, bc):
        self._p = bc
        self.power = 1.0

        self.action_dim = 7
        high = np.ones(self.action_dim)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # globals + angles + speeds + contacts
        self.state_dim = 6 + self.action_dim * 2 + 2
        high = np.inf * np.ones(self.state_dim)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def load_robot_model(self):
        flags = self._p.MJCF_COLORS_FROM_FILE
        model_path = os.path.join(current_dir, "data", "custom", "walker2d.xml")
        root_link_name = "pelvis"
        super(Walker2D, self).load_robot_model(model_path, flags, root_link_name)


class Crab2D(WalkerBase):

    foot_names = ["foot", "foot_left"]

    power_coef = {
        "thigh_left_joint": 100,
        "leg_left_joint": 100,
        "foot_left_joint": 50,
        "thigh_joint": 100,
        "leg_joint": 100,
        "foot_joint": 50,
    }

    def __init__(self, bc):
        self._p = bc
        self.power = 1.0

        self.action_dim = 6
        high = np.ones(self.action_dim)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # globals + angles + speeds + contacts
        self.state_dim = 6 + self.action_dim * 2 + 2
        high = np.inf * np.ones(self.state_dim)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def load_robot_model(self):
        flags = (
            self._p.MJCF_COLORS_FROM_FILE
            | self._p.URDF_USE_SELF_COLLISION
            | self._p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        )

        model_path = os.path.join(current_dir, "data", "custom", "crab2d.xml")
        root_link_name = "pelvis"
        super(Crab2D, self).load_robot_model(model_path, flags, root_link_name)
