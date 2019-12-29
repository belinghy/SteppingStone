import copy
import multiprocessing
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

import gym

'''plotting script python3 -m playground.plot_from_csv --load_paths runs/2019_12_13__10_45_21__curriculum_run1/ runs/2019_12_13__09_33_28__threshold_sampling_run1/ --columns mean_rew test_mean_rew --smooth 2

 python3 -m playground.plot_from_csv --load_paths runs/*curriculum_run1/ runs/*curriculum_run3/ runs/*threshold_sampling_run1/ runs/*threshold_sampling_run3/ --columns mean_rew test_mean_rew --name_regex ".*__(.*)_run*" --group 1 --smooth 2
 '''


@ex.config
def configs():
    env_name = "CassieStepper-v1"

    # Auxiliary configurations
    num_frames = 6e7
    seed = 8
    cuda = torch.cuda.is_available()
    save_every = 1e7
    log_interval = 1
    load_saved_controller = False
    use_mirror = False

    use_phase_mirror = False

    use_curriculum = False
    use_adaptive_sampling = False
    use_specialist = False
    use_threshold_sampling = False
    plot_prob = False

    # Sampling parameters
    episode_steps = 40000
    num_processes = 100  # multiprocessing.cpu_count()
    num_steps = episode_steps // num_processes
    mini_batch_size = 1024
    num_mini_batch = episode_steps // mini_batch_size
    num_tests = 4
    num_ensembles = 1
    sampling_scale = 150

    # Algorithm hyper-parameters
    use_gae = True
    lr_decay_type = "exponential"
    robot_power_decay_type = "exponential"
    gamma = 0.99
    gae_lambda = 0.95
    lr = 0.0003

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

    env_name = args.env_name

    dummy_env = make_env(env_name, render=False)

    cleanup_log_dir(args.log_dir)
    cleanup_log_dir(args.log_dir + "_test")

    try:
        os.makedirs(args.save_dir)
    except OSError:
        pass

    torch.set_num_threads(1)

    envs = make_vec_envs(env_name, args.seed, args.num_processes, args.log_dir)
    envs.set_mirror(args.use_phase_mirror)
    test_envs = make_vec_envs(env_name, args.seed, args.num_tests, args.log_dir + "_test")
    test_envs.set_mirror(args.use_phase_mirror)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.use_curriculum:
        curriculum = 0
        print("curriculum", curriculum)
        envs.update_curriculum(curriculum)
    if args.use_specialist:
        specialist = 0
        print("specialist", specialist)
        envs.update_specialist(specialist)
    if args.use_threshold_sampling:
        sampling_threshold = 200
        first_sampling = False
        uniform_sampling = True
        uniform_every = 500000
        uniform_counter = 1
        evaluate_envs = make_env(env_name, render=False)
        evaluate_envs.set_mirror(args.use_phase_mirror)
        evaluate_envs.update_curriculum(0)
        prob_filter = np.zeros((11, 11))
        prob_filter[5, 5] = 1
    if args.use_adaptive_sampling:
        evaluate_envs = make_env(env_name, render=False)
        evaluate_envs.set_mirror(args.use_phase_mirror)
        evaluate_envs.update_curriculum(0)
    if args.plot_prob:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.show(block=False)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0], *obs_shape[1:])

    if args.load_saved_controller:
        best_model = "{}_base.pt".format(env_name)
        model_path = os.path.join(current_dir, "models", best_model)
        print("Loading model {}".format(best_model))
        actor_critic = torch.load(model_path)
        actor_critic.reset_dist()
    else:
        controller = SoftsignActor(dummy_env)
        actor_critic = Policy(controller, num_ensembles=args.num_ensembles)

    mirror_function = None
    if args.use_mirror:
        indices = dummy_env.unwrapped.get_mirror_indices()
        mirror_function = get_mirror_function(indices)

    device = "cuda:0" if args.cuda else "cpu"
    if args.cuda:
        actor_critic.cuda()

    agent = PPO(actor_critic, mirror_function=mirror_function, **args.ppo_params)

    rollouts = RolloutStorage(
        args.num_steps,
        args.num_processes,
        obs_shape,
        envs.action_space.shape[0],
        actor_critic.state_size,
    )


    current_obs = torch.zeros(args.num_processes, *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        current_obs[:, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)

    rollouts.observations[0].copy_(current_obs)

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    episode_rewards = deque(maxlen=args.num_processes)
    test_episode_rewards = deque(maxlen=args.num_tests)
    num_updates = int(args.num_frames) // args.num_steps // args.num_processes

    start = time.time()
    next_checkpoint = args.save_every
    max_ep_reward = float("-inf")

    logger = ConsoleCSVLogger(
        log_dir=args.experiment_dir, console_log_interval=args.log_interval
    )

    update_values = False

    for j in range(num_updates):

        if args.lr_decay_type == "linear":
            scheduled_lr = linear_decay(j, num_updates, args.lr, final_value=0)
        elif args.lr_decay_type == "exponential":
            scheduled_lr = exponential_decay(j, 0.99, args.lr, final_value=3e-5)
        else:
            scheduled_lr = args.lr

        set_optimizer_lr(agent.optimizer, scheduled_lr)

        ac_state_dict = copy.deepcopy(actor_critic).cpu().state_dict()

        if update_values and args.use_threshold_sampling:
            envs.update_curriculum(5)
        elif (not update_values) and args.use_threshold_sampling and first_sampling:
            envs.update_specialist(0)
        
        # if args.use_threshold_sampling and not uniform_sampling:
        #     obs = evaluate_envs.reset()
        #     total_metric = torch.zeros(1, 121).to(device)
        #     evaluate_counter = 0
        #     while True:
        #         obs = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        #         with torch.no_grad():
        #             _, action, _, _ = actor_critic.act(
        #             obs, None, None, deterministic=True
        #             )
        #         cpu_actions = action.squeeze().cpu().numpy()
        #         obs, reward, done, info = evaluate_envs.step(cpu_actions)
        #         if done:
        #             obs = evaluate_envs.reset()
        #         if evaluate_envs.update_terrain:
        #             evaluate_counter += 1
        #             temp_states = evaluate_envs.create_temp_states()
        #             with torch.no_grad():
        #                 temp_states = torch.from_numpy(temp_states).float().to(device)
        #                 value_samples = actor_critic.get_ensemble_values(temp_states, None, None)
        #                 size = dummy_env.yaw_samples.shape[0]
        #                 mean = value_samples.mean(dim=-1)
        #                 #mean = value_samples.min(dim=-1)[0]
        #                 metric = mean.clone()
        #                 metric = metric.view(size, size)
        #                 #metric = metric / (metric.abs().max())
        #                 metric = metric.view(1, size*size)
        #                 total_metric += metric
        #         if evaluate_counter >= 5:
        #             total_metric /= (total_metric.abs().max())
        #             total_metric[total_metric < 0.7] = 0
        #             print("metric", total_metric)
        #             sampling_probs = (-10*(total_metric-0.85).abs()).softmax(dim=1).view(size, size) #threshold1:150, 0.9 l2, threshold2: 10, 0.85 l1, threshold3: 10, 0.85, l1, 0.40 gap
        #             #threshold 4: 20, 0.85, l1, yaw 10
        #             sample_probs = np.zeros((args.num_processes, size, size))
        #             #print("prob", sampling_probs)
        #             for i in range(args.num_processes):
        #                 sample_probs[i, :, :] = np.copy(sampling_probs.cpu().numpy().astype(np.float64))
        #             envs.update_sample_prob(sample_probs)
        #             break
        # elif args.use_threshold_sampling and uniform_sampling:
        #     envs.update_curriculum(5)
        if args.use_threshold_sampling and not uniform_sampling:
            obs = evaluate_envs.reset()
            total_metric = torch.zeros(1, 11**3).to(device)
            evaluate_counter = 0
            while True:
                obs = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    _, action, _, _ = actor_critic.act(
                    obs, None, None, deterministic=True
                    )
                cpu_actions = action.squeeze().cpu().numpy()
                obs, reward, done, info = evaluate_envs.step(cpu_actions)
                if done:
                    obs = evaluate_envs.reset()
                if evaluate_envs.update_terrain:
                    evaluate_counter += 1
                    temp_states = evaluate_envs.create_temp_states()
                    with torch.no_grad():
                        temp_states = torch.from_numpy(temp_states).float().to(device)
                        value_samples = actor_critic.get_ensemble_values(temp_states, None, None)
                        size = dummy_env.yaw_samples.shape[0]
                        mean = value_samples.mean(dim=-1)
                        #mean = value_samples.min(dim=-1)[0]
                        metric = mean.clone()
                        metric = metric.view(size, size, size)
                        #metric = metric / (metric.abs().max())
                        metric = metric.view(1, size*size*size)
                        total_metric += metric
                if evaluate_counter >= 5:
                    total_metric /= (total_metric.abs().max())
                    total_metric[total_metric < 0.7] = 0
                    print("metric", total_metric)
                    sampling_probs = (-10*(total_metric-0.85).abs()).softmax(dim=1).view(size, size, size) #threshold1:150, 0.9 l2, threshold2: 10, 0.85 l1, threshold3: 10, 0.85, l1, 0.40 gap
                    #threshold 4: 3d grid, 10, 0.85, l1
                    sample_probs = np.zeros((args.num_processes, size, size, size))
                    #print("prob", sampling_probs)
                    for i in range(args.num_processes):
                        sample_probs[i, :, :, :] = np.copy(sampling_probs.cpu().numpy().astype(np.float64))
                    envs.update_sample_prob(sample_probs)
                    break
        elif args.use_threshold_sampling and uniform_sampling:
            envs.update_curriculum(5)

        if args.use_adaptive_sampling:
            obs = evaluate_envs.reset()
            total_metric = torch.zeros(1, 121).to(device)
            evaluate_counter = 0
            while True:
                obs = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    _, action, _, _ = actor_critic.act(
                    obs, None, None, deterministic=True
                    )
                cpu_actions = action.squeeze().cpu().numpy()
                obs, reward, done, info = evaluate_envs.step(cpu_actions)
                if done:
                    obs = evaluate_envs.reset()
                if evaluate_envs.update_terrain:
                    evaluate_counter += 1
                    temp_states = evaluate_envs.create_temp_states()
                    with torch.no_grad():
                        temp_states = torch.from_numpy(temp_states).float().to(device)
                        value_samples = actor_critic.get_ensemble_values(temp_states, None, None)
                        size = dummy_env.yaw_samples.shape[0]
                        mean = value_samples.mean(dim=-1)
                        metric = mean.clone()
                        metric = metric.view(size, size)
                        #metric = metric / metric.abs().max()
                        metric = metric.view(1, size*size)
                        total_metric += metric
                        # sampling_probs = (-30*metric).softmax(dim=1).view(size, size)
                        # sample_probs = np.zeros((args.num_processes, size, size))
                        # for i in range(args.num_processes):
                        #     sample_probs[i, :, :] = np.copy(sampling_probs.cpu().numpy().astype(np.float64))
                        # envs.update_sample_prob(sample_probs)
                if evaluate_counter >= 5:
                    total_metric /= (total_metric.abs().max())
                    print("metric", total_metric)
                    sampling_probs = (-100*total_metric).softmax(dim=1).view(size, size)
                    sample_probs = np.zeros((args.num_processes, size, size))
                    for i in range(args.num_processes):
                        sample_probs[i, :, :] = np.copy(sampling_probs.cpu().numpy().astype(np.float64))
                    envs.update_sample_prob(sample_probs)
                    break

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, states = actor_critic.act(
                    rollouts.observations[step],
                    rollouts.states[step],
                    rollouts.masks[step], deterministic=update_values
                )
            cpu_actions = action.squeeze(1).cpu().numpy()

            obs, reward, done, infos = envs.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()

            if args.plot_prob and step == 0:
                temp_states = envs.create_temp_states()
                with torch.no_grad():
                    temp_states = torch.from_numpy(temp_states).float().to(device)
                    value_samples = actor_critic.get_value(temp_states, None, None)
                size = dummy_env.yaw_samples.shape[0]
                v = value_samples.mean(dim=0).view(size, size).cpu().numpy()
                vs = value_samples.var(dim=0).view(size, size).cpu().numpy()
                ax1.pcolormesh(v)
                ax2.pcolormesh(vs)
                print(np.round(v, 2))
                fig.canvas.draw()

            # if args.use_adaptive_sampling:
            #     temp_states = envs.create_temp_states()
            #     with torch.no_grad():
            #         temp_states = torch.from_numpy(temp_states).float().to(device)
            #         value_samples = actor_critic.get_value(temp_states, None, None)

            #     size = dummy_env.yaw_samples.shape[0]
            #     sample_probs = (-value_samples / 5).softmax(dim=1).view(args.num_processes, size, size)
            #     envs.update_sample_prob(sample_probs.cpu().numpy())

            # if args.use_threshold_sampling and not uniform_sampling:
            #     temp_states = envs.create_temp_states()
            #     with torch.no_grad():
            #         temp_states = torch.from_numpy(temp_states).float().to(device)
            #         value_samples = actor_critic.get_ensemble_values(temp_states, None, None)
            #     size = dummy_env.yaw_samples.shape[0]
            #     mean = value_samples.mean(dim=-1)
            #     std = value_samples.std(dim=-1)
                
                #using std
                # metric = std.clone()
                # metric = metric.view(args.num_processes, size, size)
                # value_filter = torch.ones(args.num_processes, 11, 11).to(device) * -1e5
                # value_filter[:, 5 - curriculum: 5 + curriculum + 1, 5 - curriculum: 5 + curriculum + 1] = 0
                # metric = metric / metric.max() + value_filter
                # metric = metric.view(args.num_processes, size*size)
                # sample_probs = (30*metric).softmax(dim=1).view(args.num_processes, size, size)

                #using value estimate
                # metric = mean.clone()
                # metric = metric.view(args.num_processes, size, size)
                # value_filter = torch.ones(args.num_processes, 11, 11).to(device) * -1e5
                # value_filter[:, 5 - curriculum: 5 + curriculum + 1, 5 - curriculum: 5 + curriculum + 1] = 0
                # metric = metric / metric.abs().max() - value_filter
                # metric = metric.view(args.num_processes, size*size)
                # sample_probs = (-30*metric).softmax(dim=1).view(args.num_processes, size, size)

                # if args.plot_prob and step == 0:
                #     #print(sample_probs.cpu().numpy()[0, :, :])
                #     ax.pcolormesh(sample_probs.cpu().numpy()[0, :, :])
                #     print(np.round(sample_probs.cpu().numpy()[0, :, :], 4))
                #     fig.canvas.draw()
                # envs.update_sample_prob(sample_probs.cpu().numpy())

                #using value threshold
                # metric = mean.clone()
                # metric = metric.view(args.num_processes, size, size)
                # metric = metric / metric.abs().max()# - value_filter
                # metric = metric.view(args.num_processes, size*size)
                # sample_probs = (-30*(metric-0.8)**2).softmax(dim=1).view(args.num_processes, size, size)

                # if args.plot_prob and step == 0:
                #     ax.pcolormesh(sample_probs.cpu().numpy()[0, :, :])
                #     print(np.round(sample_probs.cpu().numpy()[0, :, :], 4))
                #     fig.canvas.draw()
                # envs.update_sample_prob(sample_probs.cpu().numpy())

            bad_masks = np.ones((args.num_processes, 1))
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


        obs = test_envs.reset()
        if args.use_threshold_sampling:
            if uniform_counter % uniform_every == 0:
                uniform_sampling = True
                uniform_counter = 0
            else:
                uniform_sampling = False
            uniform_counter += 1
            if uniform_sampling:
                envs.update_curriculum(5)
                print("uniform")
        
        #print("max_step", dummy_env._max_episode_steps)
        for step in range(dummy_env._max_episode_steps):
            # Sample actions
            with torch.no_grad():
                obs = torch.from_numpy(obs).float().to(device)
                _, action, _, _ = actor_critic.act(obs, None, None, deterministic=True)
            cpu_actions = action.squeeze(1).cpu().numpy()

            obs, reward, done, infos = test_envs.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()

            for p_index, info in enumerate(infos):
                keys = info.keys()
                # This information is added by baselines.bench.Monitor
                if "episode" in keys:
                    #print(info["episode"]["r"])
                    test_episode_rewards.append(info["episode"]["r"])


        if args.use_curriculum and np.mean(episode_rewards) > 1000 and curriculum <= 4:
            curriculum += 1
            print("curriculum", curriculum)
            envs.update_curriculum(curriculum)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.observations[-1], rollouts.states[-1], rollouts.masks[-1]
            ).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)

        if update_values:
            value_loss = agent.update_values(rollouts)
        else:
            value_loss, action_loss, dist_entropy = agent.update(rollouts)
        #update_values = (not update_values)

        rollouts.after_update()

        frame_count = (j + 1) * args.num_steps * args.num_processes
        if (
            frame_count >= next_checkpoint or j == num_updates - 1
        ) and args.save_dir != "":
            model_name = "{}_{:d}.pt".format(env_name, int(next_checkpoint))
            next_checkpoint += args.save_every
        else:
            model_name = "{}_latest.pt".format(env_name)

        # A really ugly way to save a model to CPU
        save_model = actor_critic
        if args.cuda:
            save_model = copy.deepcopy(actor_critic).cpu()

        if args.use_specialist and np.mean(episode_rewards) > 1000 and specialist <= 4:
            specialist_name = "{}_specialist_{:d}.pt".format(env_name, int(specialist))
            specialist_model = actor_critic
            if args.cuda:
                specialist_model = copy.deepcopy(actor_critic).cpu()
            torch.save(specialist_model, os.path.join(args.save_dir, specialist_name))
            specialist += 1
            envs.update_specialist(specialist)
        # if args.use_threshold_sampling and np.mean(episode_rewards) > 1000 and curriculum <= 4:
        #     first_sampling = False
        #     curriculum += 1
        #     print("curriculum", curriculum)
        #     envs.update_curriculum(curriculum)
        #     prob_filter[5-curriculum:5+curriculum+1, 5-curriculum:5+curriculum+1] = 1

        torch.save(save_model, os.path.join(args.save_dir, model_name))

        if len(episode_rewards) > 1 and np.mean(episode_rewards) > max_ep_reward:
            model_name = "{}_best.pt".format(env_name)
            max_ep_reward = np.mean(episode_rewards)
            torch.save(save_model, os.path.join(args.save_dir, model_name))

        if len(episode_rewards) > 1:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            logger.log_epoch(
                {
                    "iter": j + 1,
                    "total_num_steps": total_num_steps,
                    "fps": int(total_num_steps / (end - start)),
                    "entropy": dist_entropy,
                    "value_loss": value_loss,
                    "action_loss": action_loss,
                    "stats": {"rew": episode_rewards},
                    "test_stats": {"rew": test_episode_rewards},
                }
            )
