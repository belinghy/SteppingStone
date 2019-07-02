import copy
import multiprocessing
import os
import time
from collections import deque
from glob import glob

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import numpy as np
import torch

from algorithms.ppo import PPO
from algorithms.storage import RolloutStorage
from common.controller import SoftsignActor, Policy
from common.envs_utils import (
    make_env,
    make_vec_envs,
    cleanup_log_dir,
    get_mirror_function,
)
from common.misc_utils import linear_decay, exponential_decay, set_optimizer_lr


def main():
    env_name = "Walker3DTerrainEnv-v0"
    dummy_env = make_env(env_name, render=False)

    # Auxiliary configurations
    num_frames = 6e7
    seed = 16
    cuda = torch.cuda.is_available()
    save_every = 1e7
    log_interval = 1
    save_dir = os.path.join(current_dir, "models")
    log_dir = "./logs/"
    load_saved_controller = False
    mirror_trajectory = True

    # Sampling parameters
    episode_steps = 50000
    num_processes = multiprocessing.cpu_count()
    num_steps = episode_steps // num_processes
    mini_batch_size = 1024
    num_mini_batch = episode_steps // mini_batch_size

    # Algorithm hyper-parameters
    use_gae = True
    use_clipped_value_loss = False
    lr_decay_type = "exponential"
    robot_power_decay_type = "exponential"
    entropy_coef = 0.0
    value_loss_coef = 1.0
    ppo_epoch = 10
    gamma = 0.99
    gae_lambda = 0.95
    clip_param = 0.2
    lr = 0.0003
    eps = 1e-5
    max_grad_norm = 2.0

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    cleanup_log_dir(log_dir)

    try:
        os.makedirs(save_dir)
    except OSError:
        pass

    torch.set_num_threads(1)

    envs = make_vec_envs(env_name, seed, num_processes, log_dir)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0], *obs_shape[1:])

    if load_saved_controller:
        best_model = "{}_best.pt".format(env_name)
        model_path = os.path.join(current_dir, "models", best_model)
        print("Loading model {}".format(best_model))
        actor_critic = torch.load(model_path)
    else:
        controller = SoftsignActor(dummy_env)
        actor_critic = Policy(controller)

    mirror_function = None
    if mirror_trajectory:
        indices = dummy_env.unwrapped.get_mirror_indices()
        mirror_function = get_mirror_function(indices)

    if cuda:
        actor_critic.cuda()

    agent = PPO(
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=lr,
        eps=eps,
        max_grad_norm=max_grad_norm,
        use_clipped_value_loss=use_clipped_value_loss,
        mirror_function=mirror_function,
    )

    rollouts = RolloutStorage(
        num_steps,
        num_processes,
        obs_shape,
        envs.action_space.shape[0],
        actor_critic.state_size,
    )
    current_obs = torch.zeros(num_processes, *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        current_obs[:, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)

    rollouts.observations[0].copy_(current_obs)

    if cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    episode_rewards = deque(maxlen=num_processes)
    num_updates = int(num_frames) // num_steps // num_processes

    start = time.time()
    next_checkpoint = save_every
    max_ep_reward = float("-inf")

    for j in range(num_updates):

        if lr_decay_type == "linear":
            scheduled_lr = linear_decay(j, num_updates, lr, final_value=0)
        elif lr_decay_type == "exponential":
            scheduled_lr = exponential_decay(j, 0.99, lr, final_value=3e-5)
        else:
            scheduled_lr = lr

        set_optimizer_lr(agent.optimizer, scheduled_lr)

        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, states = actor_critic.act(
                    rollouts.observations[step],
                    rollouts.states[step],
                    rollouts.masks[step],
                )
            cpu_actions = action.squeeze(1).cpu().numpy()

            obs, reward, done, infos = envs.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()

            bad_masks = np.ones((num_processes, 1))
            for p_index, info in enumerate(infos):
                keys = info.keys()
                # This information is added by algorithms.utils.TimeLimitMask
                if "bad_transition" in keys:
                    bad_masks[p_index] = 0.0
                # This information is added by baselines.bench.Monitor
                if "episode" in keys:
                    episode_rewards.append(info["episode"]["r"])

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.from_numpy(bad_masks)

            update_current_obs(obs)
            rollouts.insert(
                current_obs,
                states,
                action,
                action_log_prob,
                value,
                reward,
                masks,
                bad_masks,
            )

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.observations[-1], rollouts.states[-1], rollouts.masks[-1]
            ).detach()

        rollouts.compute_returns(next_value, use_gae, gamma, gae_lambda)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        frame_count = (j + 1) * num_steps * num_processes
        if (frame_count >= next_checkpoint or j == num_updates - 1) and save_dir != "":
            model_name = "{}_{:d}.pt".format(env_name, int(next_checkpoint))
            next_checkpoint += save_every
        else:
            model_name = "{}_latest.pt".format(env_name)

        # A really ugly way to save a model to CPU
        save_model = actor_critic
        if cuda:
            save_model = copy.deepcopy(actor_critic).cpu()

        torch.save(save_model, os.path.join(save_dir, model_name))

        if len(episode_rewards) > 1 and np.mean(episode_rewards) > max_ep_reward:
            model_name = "{}_best.pt".format(env_name)
            max_ep_reward = np.mean(episode_rewards)
            torch.save(save_model, os.path.join(save_dir, model_name))

        if j % log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            total_num_steps = (j + 1) * num_processes * num_steps
            print(
                "Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".format(
                    j,
                    total_num_steps,
                    int(total_num_steps / (end - start)),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                    dist_entropy,
                    value_loss,
                    action_loss,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
