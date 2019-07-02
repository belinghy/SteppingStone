import sys

import numpy as np
import vispy
from vispy import scene, app
from vispy.geometry import Rect


try:
    # On vispy < 0.6, this is faster
    # else it is capped at 60 fps
    import PyQt5
    import PyQt5.QtCore

    vispy.use("pyqt5", "gl2")
except ImportError:
    # On vispy > 0.6, this is faster
    vispy.use("pyglet", "gl2")


def event_handler(event):
    print(event, type(event))


def on_close(event):
    sys.exit(0)


class CustomPanZoomCamera(scene.PanZoomCamera):
    def __init__(self, rect, interactive):

        self.x_min = rect[0]
        self.x_max = self.x_min + rect[2]
        self.y_min = rect[1]
        self.y_max = self.y_min + rect[3]

        super().__init__(rect=rect, interactive=interactive)

    def expand_bounds(self, x=None, y=None):
        if y is not None:
            if y > self.y_max:
                delta = (y - self.y_max) * 2
                self.y_max = max(self.y_max * 2, self.y_max + delta)
            elif y < self.y_min:
                delta = (y - self.y_min) * 2
                self.y_min = min(self.y_min * 2, self.y_min + delta)

        if x is not None:
            if x > self.x_max:
                delta = (x - self.x_max) * 2
                self.x_max = max(self.x_max * 2, self.x_max + delta)
            elif x < self.x_min:
                delta = (x - self.x_min) * 2
                self.x_min = min(self.x_min * 2, self.x_min + delta)

        self.rect._pos = (self.x_min, self.y_min)
        self.rect._size = (self.x_max - self.x_min, self.y_max - self.y_min)


class Plot:
    def __init__(self, nrows=1, ncols=1, parent=None, grid_options=None, **kwargs):
        self._in_use = np.zeros((nrows, ncols), dtype=np.bool)

        if parent is None:
            self.parent = self

            kwargs.setdefault("keys", "interactive")
            kwargs.setdefault("show", True)
            kwargs.setdefault("size", (600, 600))

            self.canvas = scene.SceneCanvas(**kwargs)
            # self.canvas.measure_fps()

            self.canvas.on_close = on_close
            # https://github.com/vispy/vispy/issues/1201
            self.canvas.native.closeEvent = on_close
            # self.canvas.events.connect(event_handler)

            grid_options = {} if grid_options is None else grid_options
            grid_options.setdefault("spacing", 0)

            self.grid = self.canvas.central_widget.add_grid(**grid_options)
        else:
            self.parent = parent

    def _get_subplot(
        self, row=None, col=None, row_span=1, col_span=1, view_options=None
    ):

        if row is None and col is None:
            row, col = np.unravel_index(self._in_use.argmin(), self._in_use.shape)
            assert not self._in_use[row, col], "Oops, ran out of space to put new plot"

        view_options = {} if view_options is None else view_options
        view_options.setdefault("border_color", (0.5, 0.5, 0.5, 1))

        view = self.grid.add_view(row, col, row_span, col_span, **view_options)

        self._in_use[slice(row, row + row_span), slice(col, col + col_span)] = True

        return view, row, col


class TimeSeriesPlot(Plot):
    def __init__(
        self,
        parent=None,
        rows=None,
        cols=None,
        num_lines=1,
        window_size=1000,
        ylim=[-1.2, 1.2],
        view_options=None,
        plot_options=None,
        x_axis_options=None,
        y_axis_options=None,
    ):
        super().__init__(parent=parent)

        if isinstance(rows, slice) and isinstance(cols, slice):
            row, row_span = rows.start, rows.stop - rows.start
            col, col_span = cols.start, cols.stop - cols.start
        else:
            row = col = None
            row_span = col_span = 1

        self.view, row, col = self.parent._get_subplot(
            row, col, row_span, col_span, view_options
        )
        self.view.camera = CustomPanZoomCamera(
            rect=(0, ylim[0], window_size, ylim[1] - ylim[0]), interactive=False
        )

        plot_options = {} if plot_options is None else plot_options
        plot_options.setdefault("antialias", False)
        plot_options.setdefault("method", "gl")
        plot_options.setdefault("parent", self.view.scene)

        self.window_size = window_size
        x = np.arange(window_size)
        y = np.zeros(window_size)

        self.lines = [
            scene.visuals.Line(np.stack((x, y), axis=1), **plot_options)
            for _ in range(num_lines)
        ]
        self.steps = np.zeros(num_lines, dtype=np.int32)

        if y_axis_options is not None:
            y_axis_options.setdefault("orientation", "right")
            y_axis_options.setdefault("axis_font_size", 12)
            y_axis_options.setdefault("axis_label_margin", 50)
            y_axis_options.setdefault("tick_label_margin", 5)

            self.yaxis = scene.AxisWidget(**y_axis_options)
            self.parent.grid.add_widget(
                self.yaxis, row=row, col=col, row_span=row_span, col_span=col_span
            )
            self.yaxis.link_view(self.view)

        if x_axis_options is not None:
            x_axis_options.setdefault("orientation", "top")
            x_axis_options.setdefault("axis_font_size", 12)
            x_axis_options.setdefault("axis_label_margin", 50)
            x_axis_options.setdefault("tick_label_margin", 5)

            self.xaxis = scene.AxisWidget(**x_axis_options)
            self.parent.grid.add_widget(
                self.xaxis, row=row, col=col, row_span=row_span, col_span=col_span
            )
            self.xaxis.link_view(self.view)

    def add_point(self, y, line_num=0, options=None, redraw=False):

        self.view.camera.expand_bounds(y=y)

        line = self.lines[line_num]
        step = self.steps[line_num]

        shift = 1 if isinstance(y, (int, float)) else len(y)

        if step < self.window_size:
            self.steps[line_num] = step + shift
        else:
            # shift to left by length y
            line.pos[:-shift, 1] = line.pos[shift:, 1]
            step = self.window_size - shift

        line.pos[step : step + shift, 1] = y

        options = {} if options is None else options
        line.set_data(line.pos, **options)

        if redraw:
            app.process_events()


class ScatterPlot(Plot):
    def __init__(
        self,
        parent=None,
        rows=None,
        cols=None,
        xlim=[-1, 1],
        ylim=[-1, 1],
        view_options=None,
        plot_options=None,
        projection=None,
        x_axis_options=None,
        y_axis_options=None,
    ):
        super().__init__(parent=parent)

        if isinstance(rows, slice) and isinstance(cols, slice):
            row, row_span = rows.start, rows.stop - rows.start
            col, col_span = cols.start, cols.stop - cols.start
        else:
            row = col = None
            row_span = col_span = 1

        self.view, row, col = self.parent._get_subplot(
            row, col, row_span, col_span, view_options
        )
        self.view.camera = CustomPanZoomCamera(
            rect=(xlim[0], ylim[0], xlim[1] - xlim[0], ylim[1] - ylim[0]),
            interactive=False,
        )

        plot_options = {} if plot_options is None else plot_options
        plot_options.setdefault("parent", self.view.scene)

        self.scatter = scene.visuals.Markers(**plot_options)

        if projection == "3d":
            self.scatter.set_data(np.zeros((1, 3)))
            self.view.camera = "turntable"
            self.axis = scene.visuals.XYZAxis(parent=self.view.scene)
        else:
            if y_axis_options is not None:
                y_axis_options.setdefault("orientation", "right")
                y_axis_options.setdefault("axis_font_size", 12)
                y_axis_options.setdefault("axis_label_margin", 50)
                y_axis_options.setdefault("tick_label_margin", 5)

                self.yaxis = scene.AxisWidget(**y_axis_options)
                self.parent.grid.add_widget(
                    self.yaxis, row=row, col=col, row_span=row_span, col_span=col_span
                )
                self.yaxis.link_view(self.view)

            if x_axis_options is not None:
                x_axis_options.setdefault("orientation", "top")
                x_axis_options.setdefault("axis_font_size", 12)
                x_axis_options.setdefault("axis_label_margin", 50)
                x_axis_options.setdefault("tick_label_margin", 5)

                self.xaxis = scene.AxisWidget(**x_axis_options)
                self.parent.grid.add_widget(
                    self.xaxis, row=row, col=col, row_span=row_span, col_span=col_span
                )
                self.xaxis.link_view(self.view)

    def update(self, points, options=None, redraw=False):

        if isinstance(self.view.camera, CustomPanZoomCamera):
            self.view.camera.expand_bounds(points[:, 0].min(), points[:, 1].min())
            self.view.camera.expand_bounds(points[:, 0].max(), points[:, 1].max())

        options = {} if options is None else options
        options.setdefault("edge_color", None)
        options.setdefault("face_color", (1, 1, 1, 0.5))
        options.setdefault("size", 5)

        self.scatter.set_data(points, **options)

        if redraw:
            app.process_events()


if __name__ == "__main__":
    plot = Plot(nrows=2, ncols=3)

    # rows and cols can be (both) integers, slices, or None
    # if None, then will occupy any available space of size 1
    lp1 = TimeSeriesPlot(
        parent=plot,
        rows=slice(0, 1),
        cols=slice(0, 2),
        num_lines=2,
        # x_axis_options={},
        y_axis_options={},
    )
    lp1_options = {"color": (1, 0, 0, 1)}

    # x-axis range is defined by window_size and xlim
    # window_size also defines the number of points we are plotting
    # as long as it is constant (and small-ish), we can plot relatively fast
    T = 100
    lp2 = TimeSeriesPlot(parent=plot, num_lines=1, window_size=T)

    # for scatter plot, we always take 3D points
    # but we can choose to visualize in 3d or 2d using `projection`
    sc1 = ScatterPlot(parent=plot, projection="3d")

    # Turning off axis will make plotting faster
    sc2 = ScatterPlot(
        parent=plot,
        rows=slice(1, 2),
        cols=slice(1, 3),
        x_axis_options={},
        y_axis_options={},
    )

    # Currently doesn't suppose multiple figures
    # Will not be updated unless triggered by mouse or other events
    # plot2 = Plot(nrows=1, ncols=1)
    # lp3 = TimeSeriesPlot(parent=plot2, num_lines=1, window_size=T)

    x = 0
    while True:
        x += 1
        y1 = np.sin(2 * np.pi * x / (T * 20)) * np.sin(2 * np.pi * x / T)
        y2 = np.sin(2 * np.pi * x / (T * 20)) * np.cos(2 * np.pi * x / T)

        lp1.add_point(y1, 0, options=lp1_options)
        lp1.add_point(y2, 1)

        y3 = np.random.rand()
        lp2.add_point(y3, 0)

        # lp3.add_point(y3, 0, redraw=True)

        pos = np.random.normal(size=(10000, 3), scale=1.2)
        rgba = np.random.uniform(0, 1, size=(10000, 4))
        options = {"face_color": rgba}
        sc1.update(pos, options=options)

        # Drawing once at the end will redraw every plot
        sc2.update(pos, options=options, redraw=True)
