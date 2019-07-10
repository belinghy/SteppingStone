import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import gym
import numpy as np

import environments


DEG2RAD = np.pi / 180


env_name = "Walker3DPlannerEnv-v0"
env = gym.make(env_name, render=True)
action_dim = env.action_space.shape[0]
offset = 6

# Disable gravity
# env.unwrapped._p.setGravity(0, 0, 0)

obs = env.reset()


bc = env.unwrapped._p
robot_id = env.unwrapped.robot.object_id[0]
num_joints = bc.getNumJoints(robot_id)


# Bodies
num_bodies = bc.getNumBodies()
print("\nBodies:")
for i in range(num_bodies):
    name, obj = bc.getBodyInfo(i)
    name = name.decode()
    obj = obj.decode()
    print(i, name, obj)


# Links
max_z = float("-inf")
min_z = float("inf")
for i in range(num_joints):
    link_state = bc.getLinkState(robot_id, i)
    _, _, z = link_state[4]
    max_z = z if z > max_z else max_z
    min_z = z if z < min_z else min_z
print("\nHeight: {:.2f} meters".format(max_z - min_z))


# Dynamics
print("\nWeights:")
total_mass = 0
for key, part in env.unwrapped.robot.parts.items():
    if key == "floor":
        continue
    dynamics_info = bc.getDynamicsInfo(robot_id, part.bodyPartIndex)
    mass = dynamics_info[0]
    total_mass += mass
    if mass != 0:
        print("{:25} {:.4f}".format(key, mass))
print("{:25} {:.4f} kg".format("Total Mass:", total_mass))


while True:
    ## uncomment to drive the base/standing position as the action instead
    # to_normalized = env.unwrapped.robot.to_normalized
    # base_angles = env.unwrapped.robot.base_joint_angles
    # base_pose_action = to_normalized(base_angles)[[0,1,2,3,6, 7,8,9,10,13]]
    # obs, reward, done, info = env.step(base_pose_action)

    obs, reward, done, info = env.step(env.action_space.sample())

    if done:
        env.reset()
