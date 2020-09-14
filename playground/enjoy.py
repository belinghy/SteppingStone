import matplotlib
matplotlib.use('TkAgg')

import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)
import argparse
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import argparse
from glob import glob
import math
import types

import numpy as np
import torch
import gym

from common.controller import SoftsignActor, Policy
from common.envs_utils import make_env
from common.render_utils import StatsVisualizer


def euler2quat(z=0, y=0, x=0):
    z = z / 2.0
    y = y / 2.0
    x = x / 2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    result = np.array(
        [
            cx * cy * cz - sx * sy * sz,
            cx * sy * sz + cy * cz * sx,
            cx * cz * sy - sx * cy * sz,
            cx * cy * sz + sx * cz * sy,
        ]
    )
    if result[0] < 0:
        result = -result
    return result
def update_terrain_info(env):
    # print(env.next_step_index)
    next_next_step = env.next_step_index + 1
    # env.terrain_info[next_next_step, 2] = 30    
    env.sample_next_next_step()
    #print("first", env.get_temp_state()[80:])
    env.sample_next_next_step()
    #print("second", env.get_temp_state()[80:])
    # +1 because first body is worldBody
    body_index = next_next_step % env.rendered_step_count + 1
    env.model.body_pos[body_index, :] = env.terrain_info[next_next_step, 0:3]
    # account for half height
    env.model.body_pos[body_index, 2] -= env.step_half_height    
    phi, x_tilt, y_tilt = env.terrain_info[next_next_step, 3:6]
    env.model.body_quat[body_index, :] = euler2quat(phi, y_tilt, x_tilt)
    #print(x_tilt, y_tilt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="mocca_envs:Walker3DStepperEnv-v0", help="Environment ID"
    )
    parser.add_argument(
        "--net", type=str, default=None, help="Path to trained network file"
    )
    parser.add_argument("--mirror", action='store_true')
    parser.add_argument("--use_specialist", action='store_true')
    parser.add_argument("--plot_different_value", action='store_true')
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="If true, plot stats in real-time",
    )
    parser.add_argument(
        "--plot_value",
        action="store_true",
        default=False,
        help="If true, plot value in real-time",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        default=False,
        help="If true, dump camera rgb array",
    )
    args = parser.parse_args()

    env_name = args.env

    env = make_env(env_name, render=True)
    env.seed(1093)

    if args.net is None:
        best_model = "{}_latest.pt".format(env_name)
        model_path = os.path.join(current_dir, "models", best_model)
    else:
        model_path = args.net

    if args.use_specialist:
        current_specialist = 0
        specialist_paths = []
        for i in range(5):
            specialist_path = "{}_specialist_{}.pt".format(env_name, i)
            specialist_path = os.path.join(current_dir, "models", specialist_path)
            specialist_paths.append(specialist_path)
    if args.mirror:
        env.set_mirror(True)
        print("mirror")
    else:
        env.set_mirror(False)

    print("Env: {}".format(env_name))
    print("Model: {}".format(os.path.basename(model_path)))

    if args.dump:
        args.plot = False
        max_steps = 2000
        dump_dir = os.path.join(current_dir, "dump")
        image_sequence = []

        try:
            os.makedirs(dump_dir)
        except OSError:
            files = glob(os.path.join(dump_dir, "*.png"))
            for f in files:
                os.remove(f)

    else:
        max_steps = float("inf")

    if args.plot:
        num_steps = env.spec.max_episode_steps
        plotter = StatsVisualizer(100, num_steps)

    #from common.controller import SoftsignActor, Policy

    actor_critic = torch.load(model_path)
    actor_critic.reset_dist()
    if args.use_specialist:
        specialist_models = []
        for path in specialist_paths:
            specialist_models.append(torch.load(path))
    #controller = SoftsignActor(env)
    #actor_critic = Policy(controller)

    states = torch.zeros(1, actor_critic.state_size)
    masks = torch.zeros(1, 1)

    obs = env.reset()
    #env.update_curriculum(0)
    env.render()

    ep_reward = 0
    prev_contact = False
    step = 0

    if args.plot_value:
        import matplotlib
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.show(block=False)
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel("yaw")
        ax1.set_ylabel("pitch")
        ax1.set_xticks(np.arange(len(env.yaw_samples)) + 0.5)
        ax1.set_xticklabels(np.round(env.yaw_samples * 180 / np.pi, 0))
        ax1.set_yticks(np.arange(len(env.pitch_samples)) + 0.5)
        ax1.set_yticklabels(np.round(env.pitch_samples * 180 / np.pi, 0))
        p = np.meshgrid(np.power(np.linspace(0, 1, 10), 0.3))
        heatmap = ax1.pcolor(p, cmap="Reds")
        cbar = plt.colorbar(heatmap)
        # ax1.set_xticklabels(env.yaw_samples * 180 / np.pi)
        # ax1.set_yticklabels(env.pitch_samples * 180 / np.pi)
        #ax2 = fig.add_subplot(122)

    if args.plot_different_value:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.show()
        ax1 = fig.add_subplot(111)
        policy_list = []
        policy_list2 = []
        next_checkpoint = 4e5
        num_values = 30
        for i in range(num_values):
            temp_model_name = "{}_{:d}.pt".format(env_name, int(next_checkpoint))
            next_checkpoint += 4e5
            temp_model_path = os.path.join(current_dir, "../runs/2019_12_23__09_38_02__threshold6/models", temp_model_name)
            temp_policy = torch.load(temp_model_path)
            policy_list.append(temp_policy)

            temp_model_name2 = "{}_{:d}.pt".format(env_name, int(next_checkpoint))
            temp_model_path2 = os.path.join(current_dir, "../runs/2019_12_22__14_16_08__threshold5/models", temp_model_name2)
            temp_policy2 = torch.load(temp_model_path2)
            policy_list2.append(temp_policy2)


    # import matplotlib.pyplot as plt
    # import matplotlib.image as mpimg
    # image = env.camera.dump_rgb_array()
    # imgplot = plt.imshow(image, animated=True)
    # plt.show(block=False)

    while step < max_steps:
        step += 1
        obs = torch.from_numpy(obs).float().unsqueeze(0)

        with torch.no_grad():
            if args.use_specialist:
                values = [float(m.get_value(obs, None, None).squeeze()) for m in specialist_models]
                value, action, _, states = specialist_models[4].act(obs, states, masks, deterministic=True)
            else:
                value, action, _, states = actor_critic.act(
                    obs, states, masks, deterministic=True
                )
                ensemble_values = actor_critic.get_ensemble_values(obs, states, masks)
                #print("ensemble_values", ensemble_values, ensemble_values.mean(), ensemble_values.std())
        cpu_actions = action.squeeze().cpu().numpy()

        obs, reward, done, info = env.step(cpu_actions)
        #done = False

        if args.plot_different_value and env.update_terrain:
            temp_states = env.create_temp_states()
            temp_states = torch.from_numpy(temp_states).float()
            size = env.yaw_samples.shape[0]
            values = np.zeros((11, num_values, 2))
            values2 = np.zeros((11, num_values, 2))
            index = 0
            for policy in policy_list:
                with torch.no_grad():
                    value_samples = policy.get_ensemble_values(temp_states, None, None)
                value_samples = value_samples.view(size, size, 2).cpu().numpy()
                #value_samples -= value_samples.mean(axis=2, keepdims=True)
                values[:, index, :] = value_samples[0, :, :]
                index += 1
            values = values.mean(axis=2)
            #values /= values.max(axis=0)
            #print(values[:, 0, :])
            index = 0
            for policy in policy_list2:
                with torch.no_grad():
                    value_samples = policy.get_ensemble_values(temp_states, None, None)
                value_samples = value_samples.view(size, size, 2).cpu().numpy()
                #value_samples -= value_samples.mean(axis=2, keepdims=True)
                values2[:, index, :] = value_samples[0, :, :]
                index += 1
            values2 = values2.mean(axis=2)
            #values2 /= values2.max(axis=0)

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            for i in range(1):
                ax1.plot(values[5, :], 'r')
                for i in range(5):
                    label = "{}".format(i)
                    ax1.plot(values[5-i, :], label=label)
                    ax1.plot(values[5+i, :], label=label)
                #ax1.plot(values[7, : ,:].min(axis=1), 'g')
                #ax1.plot(values[0, : ,:].min(axis=1), 'g')

                #ax1.plot(values2[5, :, :].min(axis=1), 'salmon')
                #ax1.plot(values2[6, : ,:].min(axis=1), 'lightblue')
                #ax1.plot(values2[0, : ,:].min(axis=1), 'lightgreen')
                #ax1.plot(values2[0, : ,:].min(axis=1), 'gold')
            ax1.legend()
            plt.show()

        if args.plot_value and env.update_terrain:
            #print("eps reward", ep_reward)
            temp_states = env.create_temp_states()
            with torch.no_grad():
                temp_states = torch.from_numpy(temp_states).float()
                value_samples = actor_critic.get_value(temp_states, None, None)
            size = env.yaw_samples.shape[0]
            #std = value_samples.std(dim=1).view(size, size).cpu().numpy()
            mean = value_samples.view(env.yaw_sample_size, env.pitch_sample_size).cpu().numpy()
            metric = mean.copy()
            #metric = metric.reshape(size, size)
            curriculum = 5
            # value_filter = np.ones((11, 11)) * -1e5
            # value_filter[5 - curriculum: 5 + curriculum + 1, 5 - curriculum: 5 + curriculum + 1] = 0
            metric = mean# - mean.mean()
            #metric = metric / std
            metric /= np.abs(metric).max()
            #metric = np.square(metric - 0.7)
            print("mean", np.round(mean[:, :], 2))
            #print("metric", np.round(metric, 2))
            #metric = metric.reshape(size*size)
            #metric = (-0.2*metric).softmax(dim=1).view(size, size)
            metric = np.exp(-10*metric)/(np.sum(np.exp(-10*metric)))
            #print("std", np.round(std, 2))
            #print("metric", np.round(metric, 2))
            #print("metric first row", np.round(metric[:, 2], 2))
            #print("mean/std", np.round(metric/metric.max(), 5))
            #metric = (metric) / ((metric).max())
            #v = value_samples.var(dim=1).view(size, size).cpu().numpy()
            #v = (v - v.mean()) / (v.std())
            # vs = value_samples.var(dim=0).view(size, size).cpu().numpy()
            #ax1.pcolormesh(np.power(metric, 0.3), cmap="Reds", norm=matplotlib.colors.Normalize(0, 1))
            #env.update_sample_prob(metric)
            # ax2.pcolormesh(vs)
            #print(np.round(metric, 2))
            fig.canvas.draw()
            #plt.show(block=True)
        
        # temp_states = env.create_temp_states()
        # with torch.no_grad():
        #     temp_states = torch.from_numpy(temp_states).float()
        #     value = -actor_critic.get_value(temp_states, None, None)

        # size = env.yaw_samples.shape[0]
        # value = value.softmax(dim=0).view(size, size).cpu().numpy()
        # env.update_sample_prob(value)


        # threshold sampling
        # temp_states = env.create_temp_states()
        # with torch.no_grad():
        #     temp_states = torch.from_numpy(temp_states).float()
        #     value_samples = actor_critic.get_value(temp_states, None, None)
        # size = env.yaw_samples.shape[0]
        # sample_probs = (-(value_samples - 125)**2/1000).softmax(dim=0)
        # env.update_sample_prob(sample_probs.cpu().numpy())

        ep_reward += reward

        if args.plot:
            contact = env.unwrapped.robot.feet_contact[0] == 1
            strike = not prev_contact and contact
            plotter.update_plot(
                float(value),
                cpu_actions,
                ep_reward,
                done,
                strike,
                env.unwrapped.camera._fps,
            )
            prev_contact = contact

        if args.dump:
            image_sequence.append(env.unwrapped.camera.dump_rgb_array())

        if done:
            if not args.plot:
                print("Episode reward:", ep_reward)
            ep_reward = 0
            obs = env.reset()

        # if step % 5 == 0:
        #     from imageio import imwrite
        #     image = env.camera.dump_rgb_array()
        #     imgplot.set_data(image)
        #     plt.pause(0.001)
        # plt.show(block=True)
        
        #imwrite("out_{:04d}.jpg".format(step), image)

    if args.dump:
        import moviepy.editor as mp
        import datetime

        clip = mp.ImageSequenceClip(image_sequence, fps=60)
        now_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = os.path.join(dump_dir, "{}.mp4".format(now_string))
        clip.write_videofile(filename)




if __name__ == "__main__":
    main()








#####old code

# import argparse
# import os
# from glob import glob

# current_dir = os.path.dirname(os.path.realpath(__file__))
# parent_dir = os.path.dirname(current_dir)
# os.sys.path.append(parent_dir)

# import torch

# from common.envs_utils import make_env
# from common.render_utils import StatsVisualizer


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--env", type=str, default="mocca_envs:Walker3DMocapStepperEnv-v0", help="Environment ID"
#     )
#     parser.add_argument(
#         "--net", type=str, default=None, help="Path to trained network file"
#     )
#     parser.add_argument(
#         "--plot",
#         action="store_true",
#         default=False,
#         help="If true, plot stats in real-time",
#     )
#     parser.add_argument(
#         "--dump",
#         action="store_true",
#         default=False,
#         help="If true, dump camera rgb array",
#     )
#     args = parser.parse_args()

#     env_name = args.env

#     env = make_env(env_name, render=True)
#     env.seed(1093)

#     if args.net is None:
#         best_model = "{}_best.pt".format(env_name)
#         model_path = os.path.join(current_dir, "models", best_model)
#     else:
#         model_path = args.net

#     print("Env: {}".format(env_name))
#     print("Model: {}".format(os.path.basename(model_path)))

#     if args.dump:
#         args.plot = False
#         max_steps = 2000
#         dump_dir = os.path.join(current_dir, "dump")
#         image_sequence = []

#         try:
#             os.makedirs(dump_dir)
#         except OSError:
#             files = glob(os.path.join(dump_dir, "*.png"))
#             for f in files:
#                 os.remove(f)

#     else:
#         max_steps = float("inf")

#     if args.plot:
#         num_steps = env.spec.max_episode_steps
#         plotter = StatsVisualizer(100, num_steps)

#     actor_critic = torch.load(model_path)

#     states = torch.zeros(1, actor_critic.state_size)
#     masks = torch.zeros(1, 1)

#     obs = env.reset()

#     ep_reward = 0
#     prev_contact = False
#     step = 0

#     while step < max_steps:
#         step += 1
#         obs = torch.from_numpy(obs).float().unsqueeze(0)

#         with torch.no_grad():
#             value, action, _, states = actor_critic.act(
#                 obs, states, masks, deterministic=True
#             )
#         cpu_actions = action.squeeze().cpu().numpy()

#         obs, reward, done, _ = env.step(cpu_actions)
#         import time; time.sleep(0.02)
#         ep_reward += reward

#         if args.plot:
#             contact = env.unwrapped.robot.feet_contact[0] == 1
#             strike = not prev_contact and contact
#             plotter.update_plot(
#                 float(value),
#                 cpu_actions,
#                 ep_reward,
#                 done,
#                 strike,
#                 env.unwrapped.camera._fps,
#             )
#             prev_contact = contact

#         if args.dump:
#             image_sequence.append(env.unwrapped.camera.dump_rgb_array())

#         if done:
#             if not args.plot:
#                 print("Episode reward:", ep_reward)
#             ep_reward = 0
#             obs = env.reset()

#     if args.dump:
#         import moviepy.editor as mp
#         import datetime

#         clip = mp.ImageSequenceClip(image_sequence, fps=60)
#         now_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         filename = os.path.join(dump_dir, "{}.mp4".format(now_string))
#         clip.write_videofile(filename)


# if __name__ == "__main__":
#     main()
