import os
import json
import random
import argparse
import torch
import numpy as np
from sacred import Experiment
from types import SimpleNamespace
from pprint import pprint

ex = Experiment()


def print_settings(args):
    print("==============SETTINGS================")
    pprint(args.__dict__)
    print("--------------------------------------")


@ex.config
def experiment_config():
    experiment_dir = "."
    log_dir = os.path.join(experiment_dir, "logs")
    save_dir = os.path.join(experiment_dir, "models")
    replicate_num = 1


def load_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str)
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

    with open(os.path.join(args.experiment_dir, "configs.json"), "r") as cfile:
        config = json.load(cfile)
        return SimpleNamespace(net=args.net, plot=args.plot, dump=args.dump, **config)


def init(seed, config, _run):
    # This gives dot access to all paths, hyperparameters, etc
    args = SimpleNamespace(**config)

    if not hasattr(args, "seed"):
        args.seed = seed + args.replicate_num

    # Seed everything
    seed_all(args.seed)

    # Print run settings
    print_settings(args)

    if args.experiment_dir != ".":
        with open(os.path.join(args.experiment_dir, "configs.json"), "w") as cfile:
            json.dump(config, cfile, indent=2, sort_keys=True)
        with open(os.path.join(args.experiment_dir, "run.json"), "w") as cfile:
            json.dump(
                {
                    "experiment_info": _run.experiment_info,
                    "host_info": _run.host_info,
                    "start_time": _run.start_time.timetuple(),
                },
                cfile,
                indent=2,
                sort_keys=True,
            )

    # This gives global access to sacred's '_run' object without having to capture functions
    # args._run = _run

    # Other init stuff here (cuda, etc)
    return args


def seed_all(seed):
    """
    Seed all devices deterministically off of seed and somewhat independently.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
