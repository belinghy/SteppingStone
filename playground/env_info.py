import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

from common.misc_utils import make_gym_env, rad_to_deg
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple PD control for driving characters in environments"
    )
    parser.add_argument(
        "--env", default="Walker2DCustomEnv-v0", help="Name of Gym environment to run"
    )
    return parser.parse_args()


args = parse_args()
env = make_gym_env(args.env)

env.reset()
bullet_client = env.unwrapped._p
robot_id = -1

# Bodies
num_bodies = bullet_client.getNumBodies()
print("Bodies:")
for i in range(num_bodies):
    name, obj = bullet_client.getBodyInfo(i)
    name = name.decode()
    obj = obj.decode()
    print(i, name, obj)
    robot_id = (
        i if (obj == "humanoid" or obj == "walker2d" or obj == "crab2d") else robot_id
    )

# Base
print("Base Position / Orientation:")
base_pos, base_orn = env.unwrapped._p.getBasePositionAndOrientation(robot_id)
print("Position: ", base_pos)
print("Orientation: ", base_orn)

# Joints
num_joints = bullet_client.getNumJoints(robot_id)
print("\nJoints: name, damping, friction, lower limit, upper limit, link name, parent")
for i in range(num_joints):
    joint_info = bullet_client.getJointInfo(robot_id, i)
    name = joint_info[1].decode()
    damping = joint_info[6]
    friction = joint_info[7]
    lower_limit = int(rad_to_deg(joint_info[8]))
    upper_limit = int(rad_to_deg(joint_info[9]))
    link_name = joint_info[12].decode()
    parent_id = joint_info[16]
    print(
        "{:2} {:15} {} {} {:4} {:4} {:15} {}".format(
            i, name, damping, friction, lower_limit, upper_limit, link_name, parent_id
        )
    )

# Links
max_z = float("-inf")
min_z = float("inf")
print("\nLinks:")
for i in range(num_joints):
    link_state = bullet_client.getLinkState(robot_id, i)
    _, _, z = link_state[4]
    max_z = z if z > max_z else max_z
    min_z = z if z < min_z else min_z
print("Height: ", max_z - min_z)

# Dynamics
print("\nDynamics:")
total_mass = 0
for key, part in env.unwrapped.robot.parts.items():
    if key == "floor":
        continue
    dynamics_info = bullet_client.getDynamicsInfo(robot_id, part.bodyPartIndex)
    mass = dynamics_info[0]
    total_mass += mass
    if mass != 0:
        print("{:15} {:0.4f}".format(key, mass))
print("Total Mass: ", total_mass)
