import math
import os

current_dir = os.path.dirname(os.path.realpath(__file__))

import gym
import numpy as np
import pybullet

from mocca_envs.bullet_utils import BodyPart, Joint

DEG2RAD = np.pi / 180


class WalkerBase:

    mirrored = False
    applied_gain = 1.0

    def __init__(self, bc, power=1.0):
        self._p = bc
        self.base_power = power

        self.action_dim = len(self.power_coef)
        high = np.ones(self.action_dim)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # globals + angles + speeds + contacts
        self.state_dim = 6 + self.action_dim * 2 + len(self.foot_names)
        high = np.inf * np.ones(self.state_dim)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def apply_action(self, a):
        forces = (self.ordered_joint_base_gains * a).dot(self.applied_gain)
        pybullet.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.ordered_joint_ids,
            controlMode=pybullet.TORQUE_CONTROL,
            forces=forces,
            physicsClientId=self._p._client,
        )

    def calc_state(self, contact_object_ids=None):

        # get joint states for all robot's joints from pybullet
        joint_info = pybullet.getJointStates(
            self.id, self.ordered_joint_ids, physicsClientId=self._p._client
        )

        # first column is joint angles, second column is joint speeds
        # each joint is a 1 DoF hinge joint, so angle and speed are scalar values
        # 0.1 is an arbitrary scaling factor to make sure values are not too large
        # large values can be bad for neural networks
        joint_angle_and_vel = np.array([j[0:2] for j in joint_info], dtype=np.float32)
        self.joint_angles = joint_angle_and_vel[:, 0]
        self.joint_speeds = 0.1 * joint_angle_and_vel[:, 1]

        # normalize joint angles to between [-1, 1] wrt to max. range of motion
        # same thing, small values are good
        self.normalized_joint_angles = self.to_normalized(self.joint_angles)
        self.joints_at_limit = np.count_nonzero(
            np.abs(self.normalized_joint_angles) > 0.99
        )

        # get robot's root position and orientation
        body_pose = self.robot_body.pose()
        self.body_xyz = body_pose.xyz()
        self.body_rpy = body_pose.rpy()
        roll, pitch, yaw = self.body_rpy

        # get root's global velocity, and convert to local velocity
        # shouldn't matter for 2d
        vxg, vyg, vzg = self.robot_body.speed()
        yaw_cos = math.cos(-yaw)
        yaw_sin = math.sin(-yaw)
        self.body_vel = (
            yaw_cos * vxg - yaw_sin * vyg,
            yaw_sin * vxg + yaw_cos * vyg,
            vzg,
        )
        vx, vy, vz = self.body_vel

        # get global positions for feet, and calculate contact with other objects
        self.feet_xyz = np.array([f.pose().xyz() for f in self.feet])
        if contact_object_ids is not None:
            self.feet_contact = np.array(
                [
                    min(
                        len(
                            contact_object_ids
                            & set((x[2], x[4]) for x in f.contact_list())
                        ),
                        1,
                    )
                    for f in self.feet
                ]
            )

        # height is root relative to the lower foot
        height = self.body_xyz[2] - self.feet_xyz[:, 2].min()

        # state information for the root include height, velocity, and orientation
        # note that yaw is not included
        more = np.array([height, vx, vy, vz, roll, pitch], dtype=np.float32)

        # complete state is root's info, normalized joint angles, joint speeds, and contact
        state = np.concatenate(
            (more, self.normalized_joint_angles, self.joint_speeds, self.feet_contact)
        )

        # clip state to be safe, in case simulation explodes
        return state.clip(-5, +5)

    def initialize(self):
        self.load_robot_model()
        self.make_robot_utils()

    def load_robot_model(self, model_path, flags, root_link_name=None):
        self.object_id = self._p.loadMJCF(model_path, flags=flags)
        self.id = self.object_id[0]

        self.parse_joints_and_links(self.object_id)
        if root_link_name is not None:
            self.robot_body = self.parts[root_link_name]
        else:
            self.robot_body = BodyPart(self._p, "root", self.object_id, 0, -1)

        self.feet = [self.parts[f] for f in self.foot_names]
        self.feet_contact = np.zeros(len(self.foot_names), dtype=np.float32)
        self.feet_xyz = np.zeros((len(self.foot_names), 3))

        self.base_joint_angles = np.zeros(self.action_dim)
        self.base_joint_speeds = np.zeros(self.action_dim)
        self.base_position = np.array([0, 0, 0])
        self.base_orientation = np.array([0, 0, 0, 1])
        self.base_velocity = np.array([0, 0, 0])
        self.base_angular_velocity = np.array([0, 0, 0])

    def make_robot_utils(self):
        # utility functions for converting from normalized to radians and vice versa
        # Inputs are thetas (normalized angles) directly from observation
        # weight is the range of motion, thigh can move 90 degrees, etc
        weight = np.array(
            [j.upperLimit - j.lowerLimit for j in self.ordered_joints], dtype=np.float32
        )
        # bias is the angle corresponding to -1
        bias = np.array([j.lowerLimit for j in self.ordered_joints], dtype=np.float32)
        self.to_radians = lambda thetas: weight * (thetas + 1) / 2 + bias
        self.to_normalized = lambda angles: 2 * (angles - bias) / weight - 1

    def parse_joints_and_links(self, bodies):
        self.parts = {}
        self.jdict = {}
        self.ordered_joints = []
        self.ordered_joint_ids = []
        self.ordered_joint_base_gains = []
        bodies = [bodies] if np.isscalar(bodies) else bodies

        bc = self._p

        for i in range(len(bodies)):
            for j in range(bc.getNumJoints(bodies[i])):
                bc.setJointMotorControl2(
                    bodies[i],
                    j,
                    bc.POSITION_CONTROL,
                    positionGain=0.1,
                    velocityGain=0.1,
                    force=0,
                )
                jointInfo = bc.getJointInfo(bodies[i], j)
                joint_name = jointInfo[1]
                part_name = jointInfo[12]

                joint_name = joint_name.decode("utf8")
                part_name = part_name.decode("utf8")

                self.parts[part_name] = BodyPart(bc, part_name, bodies, i, j)

                if joint_name[:6] == "ignore":
                    Joint(bc, joint_name, bodies, i, j).disable_motor()
                    continue

                if joint_name[:8] != "jointfix":
                    gain = self.base_power * self.power_coef[joint_name]
                    self.jdict[joint_name] = Joint(bc, joint_name, bodies, i, j, gain)
                    self.ordered_joints.append(self.jdict[joint_name])
                    self.ordered_joint_ids.append(j)
                    self.ordered_joint_base_gains.append(gain)

        # need to use it to calculate torques later
        self.ordered_joint_base_gains = np.array(self.ordered_joint_base_gains)
        self._zeros = [0 for _ in self.ordered_joint_ids]
        self._gains = [0.1 for _ in self.ordered_joint_ids]

    def reset(self, random_pose=True, pos=None, quat=None, vel=None, ang_vel=None):
        base_joint_angles = np.copy(self.base_joint_angles)
        base_orientation = np.copy(self.base_orientation)
        if self.np_random.rand() < 0.5:
            self.mirrored = True
            base_joint_angles[self._rl] = base_joint_angles[self._lr]
            base_joint_angles[self._negation_joint_indices] *= -1
            base_orientation[0:3] *= -1
        else:
            self.mirrored = False

        if random_pose:
            # Add small deviations
            ds = self.np_random.uniform(low=-0.1, high=0.1, size=self.action_dim)
            ps = self.to_normalized(base_joint_angles + ds)
            base_joint_angles = self.to_radians(np.clip(ps, -0.95, 0.95))

        self.reset_joint_states(base_joint_angles, self.base_joint_speeds)

        pos = pos or self.base_position
        quat = quat or self.base_orientation
        self.robot_body.reset_pose(pos, quat)

        vel = vel or self.base_velocity
        ang_vel = ang_vel or self.base_angular_velocity
        self.robot_body.reset_velocity(vel, ang_vel)

        self.feet_contact.fill(0.0)
        self.feet_xyz.fill(0.0)

        robot_state = self.calc_state()
        return robot_state

    def reset_joint_states(self, positions, velocities):
        bc = self._p

        for j, pos, vel in zip(self.ordered_joint_ids, positions, velocities):
            bc.resetJointState(self.id, j, targetValue=pos, targetVelocity=vel)

        bc.setJointMotorControlArray(
            bodyIndex=self.id,
            jointIndices=self.ordered_joint_ids,
            controlMode=bc.POSITION_CONTROL,
            targetPositions=self._zeros,
            targetVelocities=self._zeros,
            positionGains=self._gains,
            velocityGains=self._gains,
            forces=self._zeros,
        )


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

    def load_robot_model(self, model_path=None, flags=None, root_link_name=None):
        if flags is None:
            flags = (
                self._p.MJCF_COLORS_FROM_FILE
                | self._p.URDF_USE_SELF_COLLISION
                | self._p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
            )

        if model_path is None:
            model_path = os.path.join(current_dir, "data", "robots", "walker3d.xml")

        # Need to call this first to parse body
        super(Walker3D, self).load_robot_model(model_path, flags, root_link_name)

        # T-pose
        self.base_joint_angles = np.zeros(self.action_dim)
        self.base_joint_speeds = np.zeros(self.action_dim)
        self.base_position = np.array([0, 0, 1.32])
        self.base_orientation = np.array([0, 0, 0, 1])
        self.base_velocity = np.array([0, 0, 0])
        self.base_angular_velocity = np.array([0, 0, 0])

        # Need this to set pose and mirroring
        # hip_[x,z,y], knee, ankle, shoulder_[x,z,y], elbow
        self._right_joint_indices = np.array(
            [3, 4, 5, 6, 7, 13, 14, 15, 16], dtype=np.int64
        )
        self._left_joint_indices = np.array(
            [8, 9, 10, 11, 12, 17, 18, 19, 20], dtype=np.int64
        )
        self._negation_joint_indices = np.array([0, 2], dtype=np.int64)  # abdomen_[x,z]
        self._rl = np.concatenate((self._right_joint_indices, self._left_joint_indices))
        self._lr = np.concatenate((self._left_joint_indices, self._right_joint_indices))

    def set_base_pose(self, pose=None):
        self.base_joint_angles[:] = 0  # reset
        self.base_orientation = np.array([0, 0, 0, 1])

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
            self.base_orientation = np.array(
                self._p.getQuaternionFromEuler([0, -angle, 0])
            )
        elif pose == "crawl":
            self.base_joint_angles[[13, 17]] = np.pi / 2  # shoulder x
            self.base_joint_angles[[14, 18]] = np.pi / 2  # shoulder z
            self.base_joint_angles[[16, 20]] = np.pi / 3  # Elbow
            self.base_joint_angles[[5, 10]] = -np.pi / 2  # hip
            self.base_joint_angles[[6, 11]] = -120 * DEG2RAD  # knee
            self.base_joint_angles[[7, 12]] = -20 * DEG2RAD  # ankles
            self.base_orientation = np.array(
                self._p.getQuaternionFromEuler([0, 90 * DEG2RAD, 0])
            )


class Child3D(Walker3D):
    def __init__(self, bc):
        super().__init__(bc, power=0.4)

    def load_robot_model(self, model_path=None, flags=None, root_link_name=None):
        if model_path is None:
            model_path = os.path.join(current_dir, "data", "robots", "child3d.xml")

        super().load_robot_model(model_path)
        self.base_position = (0, 0, 0.38)


class Walker2D(WalkerBase):

    foot_names = ["right_foot", "left_foot"]

    power_coef = {
        "torso_joint": 40,
        "right_thigh_joint": 70,
        "right_leg_joint": 70,
        "right_foot_joint": 50,
        "left_thigh_joint": 70,
        "left_leg_joint": 70,
        "left_foot_joint": 50,
    }

    def set_base_pose(self, pose=None):
        self.base_joint_angles[:] = 0  # reset
        self.base_orientation = np.array([0, 0, 0, 1])

    def load_robot_model(self, model_path=None, flags=0, root_link_name=None):
        # Need to call this first to parse body
        flags |= self._p.MJCF_COLORS_FROM_FILE
        model_path = model_path or os.path.join(
            current_dir, "data", "robots", "walker2d.xml"
        )
        super().load_robot_model(model_path, flags, "pelvis")

        # Need this to set pose and mirroring
        self._right_joint_indices = np.array([1, 2, 3], dtype=np.int64)
        self._left_joint_indices = np.array([4, 5, 6], dtype=np.int64)
        self._negation_joint_indices = np.array([], dtype=np.int64)
        self._rl = np.concatenate((self._right_joint_indices, self._left_joint_indices))
        self._lr = np.concatenate((self._left_joint_indices, self._right_joint_indices))


class Crab2D(WalkerBase):

    foot_names = ["right_foot", "left_foot"]

    power_coef = {
        "right_thigh_joint": 70,
        "right_leg_joint": 70,
        "right_foot_joint": 50,
        "left_thigh_joint": 70,
        "left_leg_joint": 70,
        "left_foot_joint": 50,
    }

    def set_base_pose(self, pose=None):
        self.base_joint_angles[:] = 0  # reset
        self.base_orientation = np.array([0, 0, 0, 1])

    def load_robot_model(self, model_path=None, flags=None, root_link_name=None):
        flags = (
            self._p.MJCF_COLORS_FROM_FILE
            | self._p.URDF_USE_SELF_COLLISION
            | self._p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        )

        model_path = os.path.join(current_dir, "data", "robots", "crab2d.xml")
        super(Crab2D, self).load_robot_model(model_path, flags, "pelvis")

        # Need this to set pose and mirroring
        self._right_joint_indices = np.array([0, 1, 2], dtype=np.int64)
        self._left_joint_indices = np.array([3, 4, 5], dtype=np.int64)
        self._negation_joint_indices = np.array([], dtype=np.int64)
        self._rl = np.concatenate((self._right_joint_indices, self._left_joint_indices))
        self._lr = np.concatenate((self._left_joint_indices, self._right_joint_indices))


class Monkey3D(Walker3D):
    foot_names = ["right_hand", "left_hand"]

    power_coef = {
        "abdomen_z": 60,
        "abdomen_y": 60,
        "abdomen_x": 60,
        "right_hip_x": 50,
        "right_hip_z": 50,
        "right_hip_y": 50,
        "right_knee": 30,
        "right_ankle": 10,
        "left_hip_x": 50,
        "left_hip_z": 50,
        "left_hip_y": 50,
        "left_knee": 30,
        "left_ankle": 10,
        "right_shoulder_x": 100,
        "right_shoulder_y": 100,
        "right_elbow_z": 60,
        "right_elbow_y": 100,
        "right_hand": 80,
        "left_shoulder_x": 100,
        "left_shoulder_y": 100,
        "left_elbow_z": 60,
        "left_elbow_y": 100,
        "left_hand": 80,
    }

    def __init__(self, bc, power=0.7):
        super().__init__(bc, power=power)

    def load_robot_model(self, model_path=None, flags=None, root_link_name=None):
        if model_path is None:
            model_path = os.path.join(current_dir, "data", "robots", "monkey3d.xml")

        super().load_robot_model(model_path)
        self.base_position = (0, 0, 0.7)

        # Need this to set pose and mirroring
        # hip_[x,z,y], knee, ankle, shoulder_[x,y], elbow
        self._right_joint_indices = np.array(
            [3, 4, 5, 6, 7, 13, 14, 15, 16, 17], dtype=np.int64
        )
        self._left_joint_indices = np.array(
            [8, 9, 10, 11, 12, 18, 19, 20, 21, 22], dtype=np.int64
        )
        self._negation_joint_indices = np.array([0, 2], dtype=np.int64)  # abdomen_[x,z]
        self._rl = np.concatenate((self._right_joint_indices, self._left_joint_indices))
        self._lr = np.concatenate((self._left_joint_indices, self._right_joint_indices))

    def set_base_pose(self, pose=None):
        self.base_joint_angles[:] = 0  # reset
        self.base_orientation = np.array([0, 0, 0, 1])

        if pose == "monkey_start":
            self.base_joint_angles[[14]] = -140 * DEG2RAD  # shoulder y
            self.base_joint_angles[[15]] = 180 * DEG2RAD  # elbow z
            self.base_joint_angles[[17]] = -90 * DEG2RAD  # finger
            self.base_joint_angles[[19]] = -170 * DEG2RAD  # shoulder y
            self.base_joint_angles[[20]] = 180 * DEG2RAD  # elbow z
            self.base_joint_angles[[22]] = -90 * DEG2RAD  # finger
            self.base_joint_angles[[6, 11]] = -90 * DEG2RAD  # ankles
            self.base_joint_angles[[7, 12]] = -90 * DEG2RAD  # knees
            self.base_orientation = np.array(self._p.getQuaternionFromEuler([0, 0, 0]))


class Mike(Walker3D):
    foot_names = ["right_foot", "left_foot"]

    power_coef = {
        "abdomen_z": 0,
        "abdomen_y": 0,
        "abdomen_x": 0,
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
        "right_shoulder_x": 30,
        "right_shoulder_z": 30,
        "right_shoulder_y": 25,
        "right_elbow": 30,
        "left_shoulder_x": 30,
        "left_shoulder_z": 30,
        "left_shoulder_y": 25,
        "left_elbow": 30,
    }

    def load_robot_model(self, model_path=None, flags=None, root_link_name=None):
        model_path = model_path or os.path.join(
            current_dir, "data", "robots", "mike.xml"
        )
        super().load_robot_model(model_path)

        waist_part = self.parts["waist"]
        body_id = waist_part.bodies[waist_part.bodyIndex]
        part_id = waist_part.bodyPartIndex
        self._p.changeDynamics(body_id, part_id, mass=8)

    def decorate(self):
        f = lambda x: os.path.join(current_dir, "data", "objects", "misc", x)

        glasses_shape = self._p.createVisualShape(
            shapeType=self._p.GEOM_MESH,
            fileName=f("glasses.obj"),
            meshScale=[0.02, 0.02, 0.02],
            visualFrameOrientation=[0, 0, 1, 1],
        )
        self.glasses_id = self._p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=glasses_shape, useMaximalCoordinates=True
        )

        hardhat_shape = self._p.createVisualShape(
            shapeType=self._p.GEOM_MESH,
            fileName=f("hardhat.obj"),
            meshScale=[0.02, 0.02, 0.02],
        )
        self.hardhat_id = self._p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=hardhat_shape, useMaximalCoordinates=True
        )
        self._p.changeVisualShape(
            self.hardhat_id, -1, textureUniqueId=self._p.loadTexture(f("hardhat.jpg"))
        )

    def calc_state(self, contact_object_ids=None):
        state = super().calc_state(contact_object_ids)

        if hasattr(self, "glasses_id") and hasattr(self, "glasses_id"):

            quat = self.robot_body.pose().orientation()
            mat = np.array(self._p.getMatrixFromQuaternion(quat)).reshape(3, 3)

            glasses_xyz = self.body_xyz + np.matmul(mat, [0.25, 0, 0])
            self._p.resetBasePositionAndOrientation(self.glasses_id, glasses_xyz, quat)

            hardhat_xyz = self.body_xyz + np.matmul(mat, [0, -0.4, 0.15])
            self._p.resetBasePositionAndOrientation(self.hardhat_id, hardhat_xyz, quat)

        return state


class Laikago(WalkerBase):

    mirrored = False
    applied_gain = 1.0

    foot_names = ["toeFR", "toeFL", "toeRR", "toeRL"]

    power_coef = {
        "FR_hip_motor_2_chassis_joint": 40,
        "FR_upper_leg_2_hip_motor_joint": 40,
        "FR_lower_leg_2_upper_leg_joint": 40,
        "FL_hip_motor_2_chassis_joint": 40,
        "FL_upper_leg_2_hip_motor_joint": 40,
        "FL_lower_leg_2_upper_leg_joint": 40,
        "RR_hip_motor_2_chassis_joint": 40,
        "RR_upper_leg_2_hip_motor_joint": 40,
        "RR_lower_leg_2_upper_leg_joint": 40,
        "RL_hip_motor_2_chassis_joint": 40,
        "RL_upper_leg_2_hip_motor_joint": 40,
        "RL_lower_leg_2_upper_leg_joint": 40,
    }

    # Need this to set pose and mirroring
    # hip_[x,z,y], knee, ankle, shoulder_[x,z,y], elbow
    _right_joint_indices = np.array([0, 1, 2, 6, 7, 8], dtype=np.int64)
    _left_joint_indices = np.array([3, 4, 5, 9, 10, 11], dtype=np.int64)
    _negation_joint_indices = np.array([], dtype=np.int64)
    _rl = np.concatenate((_right_joint_indices, _left_joint_indices))
    _lr = np.concatenate((_left_joint_indices, _right_joint_indices))

    def initialize(self):
        bc = self._p

        base_path = os.path.join(current_dir, "data", "robots", "laikago")
        model_path = os.path.join(base_path, "laikago_toes_limits.urdf")

        self.base_position = np.array([0, 0, 0.6])
        self.base_orientation = np.array([0, 0, 0, 1])
        self.base_velocity = np.array([0, 0, 0])
        self.base_angular_velocity = np.array([0, 0, 0])

        self.id = bc.loadURDF(
            model_path,
            baseOrientation=self.base_orientation,
            flags=bc.URDF_USE_SELF_COLLISION,
            useFixedBase=False,
        )

        self.parts = {}
        self.ordered_joint_ids = []
        self.ordered_joint_base_gains = []

        weight = []
        bias = []

        for j in range(bc.getNumJoints(self.id)):
            info = bc.getJointInfo(self.id, j)
            joint_name = info[1].decode("utf8")
            part_name = info[12].decode("utf8")

            self.parts[part_name] = BodyPart(bc, part_name, (self.id,), 0, j)

            joint_type = info[2]
            lower_limit = info[8]
            upper_limit = info[9]

            # Laikago only has revolute and fixed joints
            if joint_type == bc.JOINT_PRISMATIC or joint_type == bc.JOINT_REVOLUTE:
                gain = self.base_power * self.power_coef[joint_name]
                self.ordered_joint_ids.append(j)
                self.ordered_joint_base_gains.append(gain)
                weight.append(upper_limit - lower_limit)
                bias.append(lower_limit)

        # weight is the range of motion, thigh can move 90 degrees, etc
        # bias is the angle corresponding to -1
        weight = np.array(weight)
        bias = np.array(bias)
        # utility functions for converting from normalized to radians and vice versa
        # Inputs are thetas (normalized angles) directly from observation
        self.to_radians = lambda thetas: weight * (thetas + 1) / 2 + bias
        self.to_normalized = lambda angles: 2 * (angles - bias) / weight - 1

        # need to use it to calculate torques later
        self.ordered_joint_base_gains = np.array(self.ordered_joint_base_gains)
        self._zeros = [0 for _ in self.ordered_joint_ids]
        self._gains = [0.1 for _ in self.ordered_joint_ids]

        self.base_joint_angles = np.zeros(self.action_dim)
        self.base_joint_speeds = np.zeros(self.action_dim)

        self.robot_body = BodyPart(self._p, "root", (self.id,), 0, -1)
        self.feet = [self.parts[f] for f in self.foot_names]
        self.feet_contact = np.zeros(len(self.foot_names), dtype=np.float32)
        self.feet_xyz = np.zeros((len(self.foot_names), 3))

    def set_base_pose(self, pose=None):
        self.base_joint_angles = np.zeros(self.action_dim)
        self.base_orientation = np.array([0, 0, 0, 1])

        if pose == "running_start":
            self.base_joint_angles[[2, 5, 8, 11]] = -np.pi / 6  # Lower legs
            # self.base_joint_angles[[1, 10]] = -np.pi / 9  # diagonal upper legs
            # self.base_joint_angles[[4, 7]] = np.pi / 9  # other diagonal upper legs
