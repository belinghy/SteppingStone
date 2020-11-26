"""
Helper script used for rendering the best learned policy for an existing experiment.

Usage:
```bash
python enjoy.py --env <ENV> --dir
python enjoy.py --env <ENV> --net <PATH/TO/NET> --len <STEPS>
```
"""
import argparse
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import torch

from common.envs_utils import make_env
from common.misc_utils import EpisodeRunner


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Examples:\n"
            "   python enjoy.py --env <ENV> --net <NET>\n"
            "   (Remote) python enjoy.py --env <ENV> --net <NET> --len 1000 --render 0 --save 1\n"
            "   (Faster) python enjoy.py --env <ENV> --net <NET> --len 1000 --save 1 --ffmpeg 1\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--net", type=str, required=True)
    parser.add_argument("--len", type=int, default=float("inf"))
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--save", type=int, default=0)
    parser.add_argument("--ffmpeg", type=int, default=0)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    # Save options:
    #   1) render=True ffmpeg=False -> Dump frame by frame using getCameraImage, high quality
    #   2) render=True ffmpeg=True -> Use pybullet.connect option
    #   3) render=False -> Use EGL and getCameraImage
    use_egl = args.save and not args.render
    use_ffmpeg = args.render and args.ffmpeg
    env = make_env(
        args.env,
        render=args.render,
        use_egl=use_egl,
        use_ffmpeg=use_ffmpeg,
    )
    env.seed(1093)

    model_path = args.net or os.path.join(args.save_dir, f"{args.env}_latest.pt")

    print("Env: {}".format(args.env))
    print("Model: {}".format(os.path.basename(model_path)))

    actor_critic = torch.load(model_path).to("cpu")

    # Set global no_grad
    torch.set_grad_enabled(False)

    runner_options = {
        "save": args.save,
        "use_ffmpeg": use_ffmpeg,
        "max_steps": args.len,
        "csv": args.csv,
    }

    with EpisodeRunner(env, **runner_options) as runner:

        obs = env.reset()
        ep_reward = 0

        while not runner.done:
            obs = torch.from_numpy(obs).float().unsqueeze(0)
            value, action, _ = actor_critic.act(obs, deterministic=True)
            cpu_actions = action.squeeze().cpu().numpy()

            obs, reward, done, _ = env.step(cpu_actions)
            ep_reward += reward

            if done:
                print("--- Episode reward:", ep_reward)
                obs = env.reset(reset_runner=False)
                ep_reward = 0


if __name__ == "__main__":
    main()
