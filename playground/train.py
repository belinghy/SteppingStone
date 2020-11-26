"""
Usage:
```bash
./scripts/local_run_playground_train.sh <EXPERIMENT_NAME> env=<ENV>
or
python train.py with env=<ENV>
```
"""

import os
import time
from collections import deque

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, parent_dir)

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
from common.csv_utils import ConsoleCSVLogger
from common.sacred_utils import ex, init


@ex.config
def configs():
    env = "mocca_envs:Walker2DCustomEnv-v0"

    # Sampling parameters
    num_frames = 1e7
    episode_steps = 40000
    num_processes = 125 if os.name != "nt" else torch.multiprocessing.cpu_count()
    num_steps = episode_steps // num_processes
    mini_batch_size = 1024
    num_mini_batch = episode_steps // mini_batch_size

    # Auxiliary configurations
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed = 16
    save_every = int(num_frames // 5)
    log_interval = 2
    net = None

    # Algorithm hyper-parameters
    use_gae = True
    lr_decay_type = "exponential"
    gamma = 0.99
    gae_lambda = 0.95
    lr = 3e-4

    ppo_params = {
        "use_clipped_value_loss": False,
        "num_mini_batch": num_mini_batch,
        "entropy_coef": 0.0,
        "value_loss_coef": 1.0,
        "ppo_epoch": 10,
        "clip_param": 0.2,
        "lr": lr,
        "eps": 1e-5,
        "max_grad_norm": 2.0,
    }


@ex.automain
def main(_seed, _config, _run):
    args = init(_seed, _config, _run)

    env_name = args.env

    env_name_parts = env_name.split(":")
    save_name = "-".join(env_name_parts) if len(env_name_parts) > 1 else env_name

    env_kwargs = {}
    dummy_env = make_env(env_name, **env_kwargs)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    torch.set_num_threads(1)

    envs = make_vec_envs(
        env_name, args.seed, args.num_processes, args.log_dir, **env_kwargs
    )

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0], *obs_shape[1:])
    action_dim = envs.action_space.shape[0]

    if args.net is not None:
        print(f"Loading model {args.net}")
        actor_critic = torch.load(args.net)
    else:
        controller = SoftsignActor(dummy_env)
        actor_critic = Policy(controller)

    actor_critic = actor_critic.to(args.device)
    agent = PPO(actor_critic, **args.ppo_params)

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, action_dim)
    rollouts.to(args.device)

    obs = envs.reset()
    rollouts.observations[0].copy_(torch.from_numpy(obs))

    episode_rewards = deque(maxlen=args.num_processes)
    num_updates = int(args.num_frames) // args.num_steps // args.num_processes

    start = time.time()
    next_checkpoint = args.save_every
    max_ep_reward = float("-inf")

    logger = ConsoleCSVLogger(
        log_dir=args.experiment_dir, console_log_interval=args.log_interval
    )

    for iteration in range(num_updates):

        if args.lr_decay_type == "linear":
            scheduled_lr = linear_decay(iteration, num_updates, args.lr, final_value=0)
        elif args.lr_decay_type == "exponential":
            scheduled_lr = exponential_decay(iteration, 0.99, args.lr, final_value=3e-5)
        else:
            scheduled_lr = args.lr

        set_optimizer_lr(agent.optimizer, scheduled_lr)

        # Disable gradient for data collection
        with torch.no_grad():
            for step in range(args.num_steps):
                value, action, action_log_prob = actor_critic.act(
                    rollouts.observations[step]
                )
                cpu_actions = action.squeeze(1).cpu().numpy()

                obs, reward, done, infos = envs.step(cpu_actions)
                reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()

                bad_masks = np.ones((args.num_processes, 1))
                for p_index, info in enumerate(infos):
                    # This information is added by common.envs_utils.TimeLimitMask
                    if "bad_transition" in info:
                        bad_masks[p_index] = 0.0
                    # This information is added by common.envs_utils.Monitor
                    if "episode" in info:
                        episode_rewards.append(info["episode"]["r"])

                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.from_numpy(bad_masks)

                rollouts.insert(
                    torch.from_numpy(obs),
                    action,
                    action_log_prob,
                    value,
                    reward,
                    masks,
                    bad_masks,
                )

            next_value = actor_critic.get_value(rollouts.observations[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        frame_count = (iteration + 1) * args.num_steps * args.num_processes
        if frame_count >= next_checkpoint or iteration == num_updates - 1:
            model_name = f"{save_name}_{int(next_checkpoint)}.pt"
            next_checkpoint += args.save_every
        else:
            model_name = f"{save_name}_latest.pt"

        torch.save(actor_critic, os.path.join(args.save_dir, model_name))

        if len(episode_rewards) > 1 and np.mean(episode_rewards) > max_ep_reward:
            max_ep_reward = np.mean(episode_rewards)
            model_name = f"{save_name}_best.pt"
            torch.save(actor_critic, os.path.join(args.save_dir, model_name))

        if len(episode_rewards) > 1:
            end = time.time()
            logger.log_epoch(
                {
                    "iter": iteration + 1,
                    "total_num_steps": frame_count,
                    "fps": int(frame_count / (end - start)),
                    "entropy": dist_entropy,
                    "value_loss": value_loss,
                    "action_loss": action_loss,
                    "stats": {"rew": episode_rewards},
                }
            )
