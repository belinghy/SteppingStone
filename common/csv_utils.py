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

        if "test_stats" in data:
            for key, values in data["test_stats"].items():
                data["test_mean_" + key] = np.mean(values)
                data["test_median_" + key] = np.median(values)
                data["test_min_" + key] = np.min(values)
                data["test_max_" + key] = np.max(values)
        del data["test_stats"]

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

        if data["iter"] % self.console_log_interval == 0:
            print(
                "Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, test_mean/median reward {:.1f}/{:.1f}, test_min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".format(
                    data["iter"],
                    data["total_num_steps"],
                    data["fps"],
                    data["mean_rew"],
                    data["median_rew"],
                    data["min_rew"],
                    data["max_rew"],
                    data["test_mean_rew"],
                    data["test_median_rew"],
                    data["test_min_rew"],
                    data["test_max_rew"],
                    data["entropy"],
                    data["value_loss"],
                    data["action_loss"],
                ),
                flush=True,
            )
