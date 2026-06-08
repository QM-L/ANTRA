from matplotlib.image import AxesImage

class MplContrastHelper:
    ''''Handles contrast changing (window width/level) with left click drag for a single axes'''
    def __init__(self, img: AxesImage):
        self.img = img

        self.HU_rate = 1
        self.x_on_click = 0
        self.y_on_click = 0
        self.width_on_click = 400
        self.level_on_click = 40

        self.initial_left = int(self.level_on_click - self.width_on_click/2)
        self.initial_right = int(self.level_on_click + self.width_on_click/2)
        self.on_home(None)

        img.axes.figure.canvas.mpl_connect('button_press_event', self.on_button_press)
        img.axes.figure.canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_home(self, event):
        self.img.set_clim(self.initial_left, self.initial_right)
        self.img.figure.canvas.draw_idle()

    def on_key_press(self, event):
        if event.key == 'h': self.on_home(event)

    def on_button_press(self, event):
        if event.inaxes is None: return
        if event.button != 1: return

        self.x_on_click = event.xdata
        self.y_on_click = event.ydata
        self.width_on_click  = self.img.get_clim()[1] - self.img.get_clim()[0]
        self.level_on_click  = (self.img.get_clim()[1] + self.img.get_clim()[0]) / 2

    def on_move(self, event):
        if event.inaxes is None: return
        if event.button != 1: return

        dx = event.xdata - self.x_on_click
        dy = event.ydata - self.y_on_click
        width = max(1, self.width_on_click + dx * self.HU_rate)
        level = self.level_on_click + dy * self.HU_rate

        # set clim based on WW/WL values
        left  = int(level - width/2)
        right = int(level + width/2)
        self.img.set_clim(left, right)

        self.img.figure.canvas.draw_idle()
