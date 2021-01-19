import gym
import os


def register(id, **kvargs):
    if id in gym.envs.registration.registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, **kvargs)


# fixing package path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)


register(
    id="CassieEnv-v0",
    entry_point="mocca_envs.env_cassie:CassieEnv",
    max_episode_steps=1000,
)

register(
    id="Cassie2DEnv-v0",
    entry_point="mocca_envs.env_cassie:CassieEnv",
    max_episode_steps=1000,
    kwargs={"planar": True},
)

register(
    id="CassiePhaseMocca2DEnv-v0",
    entry_point="mocca_envs.env_cassie:CassiePhaseMoccaEnv",
    max_episode_steps=1000,
    kwargs={"planar": True},
)

register(
    id="CassiePhaseMirror2DEnv-v0",
    entry_point="mocca_envs.env_cassie:CassiePhaseMirrorEnv",
    max_episode_steps=1000,
    kwargs={"planar": True},
)

register(
    id="Child3DCustomEnv-v0",
    entry_point="mocca_envs.env_locomotion:Child3DCustomEnv",
    max_episode_steps=1000,
)


register(
    id="Walker3DCustomEnv-v0",
    entry_point="mocca_envs.env_locomotion:Walker3DCustomEnv",
    max_episode_steps=1000,
)

register(
    id="Walker3DStepperEnv-v0",
    entry_point="mocca_envs.env_locomotion:Walker3DStepperEnv",
    max_episode_steps=1000,
)

register(
    id="MikeStepperEnv-v0",
    entry_point="mocca_envs.env_locomotion:MikeStepperEnv",
    max_episode_steps=1000,
)

register(
    id="LaikagoCustomEnv-v0",
    entry_point="mocca_envs.env_locomotion:LaikagoCustomEnv",
    max_episode_steps=1000,
)

register(
    id="LaikagoStepperEnv-v0",
    entry_point="mocca_envs.env_locomotion:LaikagoStepperEnv",
    max_episode_steps=1000,
)

register(
    id="Walker3DPlannerEnv-v0",
    entry_point="mocca_envs.env_locomotion:Walker3DPlannerEnv",
    max_episode_steps=1000,
)

register(
    id="MikePlannerEnv-v0",
    entry_point="mocca_envs.env_locomotion:MikePlannerEnv",
    max_episode_steps=1000,
)

register(
    id="Monkey3DCustomEnv-v0",
    entry_point="mocca_envs.env_locomotion:Monkey3DCustomEnv",
    max_episode_steps=1000,
)

register(
    id="Walker3DChairEnv-v0",
    entry_point="mocca_envs.env_interaction:Walker3DChairEnv",
    max_episode_steps=1000,
)

register(
    id="Walker2DCustomEnv-v0",
    entry_point="mocca_envs.env_locomotion_2d:Walker2DCustomEnv",
    max_episode_steps=1000,
)

register(
    id="Crab2DCustomEnv-v0",
    entry_point="mocca_envs.env_locomotion_2d:Crab2DCustomEnv",
    max_episode_steps=1000,
)
