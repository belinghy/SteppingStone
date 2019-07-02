import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


class StatsVisualizer:
    def __init__(self, window, max_length):

        plt.rcParams["toolbar"] = "None"
        fig = plt.figure(figsize=(5, 5))

        if plt.get_backend() == "Qt5Agg":
            # Set figure window location to left side
            manager = plt.get_current_fig_manager()
            _, _, dx, dy = manager.window.geometry().getRect()
            manager.window.setGeometry(1100, 100, dx, dy)

        fig.set_facecolor("black")
        fig.canvas.mpl_connect("close_event", lambda x: sys.exit())

        nrow = 11
        ncol = 6
        gs = gridspec.GridSpec(
            nrow,
            ncol,
            wspace=0.0,
            hspace=0.0,
            top=1.0 - 0.5 / (nrow + 1),
            bottom=0.5 / (nrow + 1),
            left=0.5 / (ncol + 1),
            right=1 - 0.5 / (ncol + 1),
        )

        vf_axis = fig.add_subplot(gs[0, :])
        vf_axis.set_facecolor("black")
        vf_axis.grid(False)
        vf_axis.set_xlim(0, window)
        vf_axis.set_ylim(1e-2, 500)
        vf_axis.spines["bottom"].set_color("#dddddd")
        vf_axis.spines["top"].set_color("#dddddd")
        vf_axis.spines["right"].set_color("red")
        vf_axis.spines["left"].set_color("red")
        vf_axis.tick_params(axis="y", colors="#dddddd")

        action_axes = [
            fig.add_subplot(gs[1, 0:2], yticks=[], yticklabels=[]),  # abdomen z
            fig.add_subplot(gs[1, 2:4], yticks=[], yticklabels=[]),  # abdomen y
            fig.add_subplot(gs[1, 4:6], yticks=[], yticklabels=[]),  # abdomen x
            fig.add_subplot(gs[2, 3:6], yticks=[], yticklabels=[]),  # right_hip_x
            fig.add_subplot(gs[3, 3:6], yticks=[], yticklabels=[]),  # right_hip_z
            fig.add_subplot(gs[4, 3:6], yticks=[], yticklabels=[]),  # right_hip_y
            fig.add_subplot(gs[5, 3:6], yticks=[], yticklabels=[]),  # right_knee
            fig.add_subplot(gs[6, 3:6], yticks=[], yticklabels=[]),  # right_ankle
            fig.add_subplot(gs[2, 0:3]),  # left_hip_x
            fig.add_subplot(gs[3, 0:3], yticks=[], yticklabels=[]),  # left_hip_z
            fig.add_subplot(gs[4, 0:3]),  # left_hip_y
            fig.add_subplot(gs[5, 0:3], yticks=[], yticklabels=[]),  # left_knee
            fig.add_subplot(gs[6, 0:3]),  # left_ankle
            fig.add_subplot(gs[7, 3:6], yticks=[], yticklabels=[]),  # right_shoulder_x
            fig.add_subplot(gs[8, 3:6], yticks=[], yticklabels=[]),  # right_shoulder_z
            fig.add_subplot(gs[9, 3:6], yticks=[], yticklabels=[]),  # right_shoulder_y
            fig.add_subplot(gs[10, 3:6], yticks=[], yticklabels=[]),  # right_elbow
            fig.add_subplot(gs[7, 0:3], yticks=[], yticklabels=[]),  # left_shoulder_x
            fig.add_subplot(gs[8, 0:3]),  # left_shoulder_z
            fig.add_subplot(gs[9, 0:3], yticks=[], yticklabels=[]),  # left_shoulder_y
            fig.add_subplot(gs[10, 0:3]),  # left_elbow
        ]

        for i, ax in enumerate(action_axes):
            ax.set_facecolor("black")
            ax.grid(False)
            ax.set_xlim(0, window)
            ax.set_ylim(-1.2, 1.2)
            ax.spines["bottom"].set_color("#dddddd")
            ax.spines["top"].set_color("#dddddd")
            ax.spines["right"].set_color("red")
            ax.spines["left"].set_color("red")
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.yaxis.label.set_color("red")
            ax.xaxis.label.set_color("red")
            ax.tick_params(axis="y", colors="#dddddd")
            ax.title.set_color("red")

        names = [
            "hip_x",
            "hip_z",
            "hip_y",
            "knee",
            "ankle",
            "shoulder_x",
            "shoulder_z",
            "shoulder_y",
            "elbow",
        ]

        right_joints = [action_axes[i] for i in [3, 4, 5, 6, 7, 13, 14, 15, 16]]

        for ax, name in zip(right_joints, names):
            ax.text(
                0.95,
                0.1,
                name,
                color="#dddddd",
                horizontalalignment="right",
                transform=ax.transAxes,
            )

        plt.ion()
        plt.show()

        fig.canvas.draw()
        self.background = fig.canvas.copy_from_bbox(fig.bbox)

        fake_data = np.zeros(window)

        line_style = {
            "linestyle": "none",
            "marker": "o",
            "markersize": "1",
            "animated": True,
        }

        vf_line = vf_axis.plot(fake_data, **line_style)[0]

        fps_label = vf_axis.text(
            0.1,
            0.6,
            "00.0",
            color="#dddddd",
            horizontalalignment="right",
            transform=vf_axis.transAxes,
            animated=True,
        )

        reward_label = vf_axis.text(
            0.97,
            0.6,
            "0000.0",
            color="#dddddd",
            horizontalalignment="right",
            transform=vf_axis.transAxes,
            animated=True,
        )

        action_lines = []
        for ax in action_axes:
            line = ax.plot(fake_data, **line_style)[0]
            action_lines.append(line)

        cline_style = {
            "x": 0,
            "ymin": 0,
            "ymax": 1,
            "color": "orange",
            "animated": True,
        }
        ax = right_joints[3]
        num_lines = 5
        self.clines = [ax.axvline(**cline_style) for _ in range(num_lines)]
        self.clines_x = (window + 1) * np.ones(num_lines)
        self.clines_active = np.zeros(num_lines)

        for cl in self.clines:
            cl.set_xdata(window + 1)

        self.fig = fig
        self.lines = [vf_line] + action_lines
        self.fps_label = fps_label
        self.reward_label = reward_label

        self.step = 0
        self.window = window
        self.batch = 1

        self.data = -2 * np.ones((len(self.lines), max_length))

    def update_plot(self, value, action, tot_reward, done, contact, fps):

        self.data[0, self.step] = value
        self.data[1:, self.step] = action

        self.fps_label.set_text("{:02.1f}".format(fps))

        self.reward_label.set_text("{:04.1f}".format(tot_reward))

        self.fig.canvas.restore_region(self.background)

        self.fps_label.axes.draw_artist(self.fps_label)
        self.reward_label.axes.draw_artist(self.reward_label)

        # Draw new data
        start = self.step % self.batch
        begin = 0 if self.step < self.window else self.step - 100
        end = begin + 100

        # This whole section is pretty ugly...
        if self.step >= self.window:
            if contact:
                # Find a new line to move if contact
                active_x = self.clines_x[self.clines_active == 1]
                max_active_x = 0 if active_x.size == 0 else active_x.max()
                found = False
                for i, x in enumerate(self.clines_x):
                    if x <= 0:
                        self.clines_active[i] = 0
                        self.clines_x[i] = self.window + 1
                        self.clines[i].set_xdata(self.clines_x[i])
                    if x - max_active_x > 5 and not found:
                        found = True
                        self.clines_active[i] = 1

            # Move every active line left 1
            for i, active in enumerate(self.clines_active):
                if not active:
                    continue

                self.clines_x[i] -= 1
                self.clines[i].set_xdata(self.clines_x[i])
                self.clines[i].axes.draw_artist(self.clines[i])
        else:
            if contact:
                # Find a new line to move if contact
                found = False
                for i, x in enumerate(self.clines_x):
                    if not self.clines_active[i] and not found:
                        found = True
                        self.clines_active[i] = 1
                        self.clines_x[i] = self.step
                        self.clines[i].set_xdata(self.clines_x[i])

            # Still need to redraw because blit
            for i, active in enumerate(self.clines_active):
                if not active:
                    continue
                self.clines[i].axes.draw_artist(self.clines[i])

        for (line, data) in zip(
            self.lines[start :: self.batch], self.data[start :: self.batch]
        ):
            line.set_ydata(data[begin:end])
            line.axes.draw_artist(line)
            self.fig.canvas.blit(line.axes.bbox)

        self.fig.canvas.flush_events()
        self.step += 1

        if done:
            self.step = 0
            self.reward = 0
            self.data.fill(-2)
            self.clines_active.fill(0)
            self.clines_x.fill(self.window + 1)
