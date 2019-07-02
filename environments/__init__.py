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
    entry_point="environments.cassie_envs:CassieEnv",
    max_episode_steps=1000,
)

register(
    id="Walker3DCustomEnv-v0",
    entry_point="environments.walker3d_envs:Walker3DCustomEnv",
    max_episode_steps=1000,
)

register(
    id="Walker3DTerrainEnv-v0",
    entry_point="environments.walker3d_envs:Walker3DTerrainEnv",
    max_episode_steps=1000,
)
