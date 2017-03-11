import matplotlib.pyplot as plt
import seaborn as sns

plt.ion()


class Plotter:
    def __init__(self):
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111)
        # self.ax1 = self.fig.add_subplot(211)
        # self.ax2 = self.fig.add_subplot(212)
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
        l0 = wstart[0] - wend[0]
        l1 = wstart[1] - wend[1]
        import numpy as np
        self.ax1.streamplot(wstart[0], wstart[1], -wend[0], -wend[1], density=1.5, color=np.sqrt(l0 ** 2 + l1 ** 2),
                            cmap=plt.cm.winter)
        plt.pause(0.0001)

    def plot_value(self):
        """

        :return:
        """
