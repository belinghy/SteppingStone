import math
import time
import argparse


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (as_minutes(s), as_minutes(rs))


def linear_decay(epoch, total_num_epochs, initial_value, final_value):
    return initial_value - (initial_value - final_value) * epoch / float(
        total_num_epochs
    )


def exponential_decay(epoch, rate, initial_value, final_value):
    return max(initial_value * (rate ** epoch), final_value)


def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class StringEnum(tuple):
    def __getattr__(self, attr):
        if attr in self:
            return attr


def str2bool(v):
    """
    Argument Parse helper function for boolean values
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
