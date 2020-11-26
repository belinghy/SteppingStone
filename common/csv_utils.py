import os
import csv
import numpy as np


class CSVLogger(object):
    def __init__(self, log_dir, filename="progress.csv"):
        self.csvfile = open(os.path.join(log_dir, filename), "w")
        self.writer = None

    def init_writer(self, keys):
        if self.writer is None:
            self.writer = csv.DictWriter(self.csvfile, fieldnames=list(keys))
            self.writer.writeheader()

    def log_epoch(self, data):
        if "stats" in data:
            for key, values in data["stats"].items():
                data["mean_" + key] = np.mean(values)
                data["median_" + key] = np.median(values)
                data["min_" + key] = np.min(values)
                data["max_" + key] = np.max(values)
        del data["stats"]

        self.init_writer(data.keys())
        self.writer.writerow(data)
        self.csvfile.flush()

    def __del__(self):
        self.csvfile.close()


class ConsoleCSVLogger(CSVLogger):
    def __init__(self, console_log_interval=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.console_log_interval = console_log_interval

    def log_epoch(self, data):
        super().log_epoch(data)

        flush = data["iter"] % self.console_log_interval == 0
        print(
            (
                f'Updates {data["iter"]}, '
                f'num timesteps {data["total_num_steps"]}, '
                f'FPS {data["fps"]}, '
                f'mean/median reward {data["mean_rew"]:.1f}/{data["median_rew"]:.1f}, '
                f'min/max reward {data["min_rew"]:.1f}/{data["max_rew"]:.1f}, '
                f'entropy {data["entropy"]:.2f}, '
                f'value loss {data["value_loss"]:.2f}, '
                f'policy loss {data["action_loss"]:.3f}'
            ),
            flush=flush,
        )
