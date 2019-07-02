import argparse
import os
from glob import glob

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import torch

from common.envs_utils import make_env
from common.render_utils import StatsVisualizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="Walker3DTerrainEnv-v0", help="Environment ID"
    )
    parser.add_argument(
        "--net", type=str, default=None, help="Path to trained network file"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="If true, plot stats in real-time",
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
        best_model = "{}_best.pt".format(env_name)
        model_path = os.path.join(current_dir, "models", best_model)
    else:
        model_path = args.net

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

    actor_critic = torch.load(model_path)

    states = torch.zeros(1, actor_critic.state_size)
    masks = torch.zeros(1, 1)

    obs = env.reset()

    ep_reward = 0
    prev_contact = False
    step = 0

    while step < max_steps:
        step += 1
        obs = torch.from_numpy(obs).float().unsqueeze(0)

        with torch.no_grad():
            value, action, _, states = actor_critic.act(
                obs, states, masks, deterministic=True
            )
        cpu_actions = action.squeeze().cpu().numpy()

        obs, reward, done, _ = env.step(cpu_actions)
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

    if args.dump:
        import moviepy.editor as mp
        import datetime

        clip = mp.ImageSequenceClip(image_sequence, fps=60)
        now_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = os.path.join(dump_dir, "{}.mp4".format(now_string))
        clip.write_videofile(filename)


if __name__ == "__main__":
    main()
