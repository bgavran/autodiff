import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

plt.ion()


class Plotter:
    def __init__(self):
        self.fig = plt.figure()
        # self.ax1 = self.fig.add_subplot(111)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212, projection="3d")
        self.fig.show()
        plt.tight_layout()
        plt.pause(0.1)

    def plot_stream(self, wstart, wend, wrt):
        """
        Stream plot of gradients with respect to weights
        :param wstart:
        :param wend:
        :param wrt:
        :return:
        """
        assert len(wstart) == len(wend)
        assert len(wstart) in [2]  # later make it plottable in 3 dimensions

        self.ax1.clear()
        self.ax1.set_xlabel(wrt[0]), self.ax1.set_ylabel(wrt[1])
        # reverse the values because we're minimizing the function
        norm = np.sqrt(wend[0] ** 2 + wend[1] ** 2)
        self.ax1.streamplot(wstart[0], wstart[1], -wend[0], -wend[1], density=1.5,
                            color=norm,
                            cmap=plt.cm.winter)
        plt.pause(0.0001)

    def plot_value(self, x, y, z, wrt):
        self.ax2.clear()
        self.ax2.set_xlabel(wrt[0]), self.ax2.set_ylabel(wrt[1])
        self.ax2.plot_surface(x, y, z, cmap=plt.cm.coolwarm)
        plt.pause(0.0001)
