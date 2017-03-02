import matplotlib.pyplot as plt
import seaborn as sns

plt.ion()


class Plotter:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.fig.show()
        plt.pause(0.1)

    def plot_stream(self, wstart, wend, wrt):
        assert len(wstart) == len(wend)
        assert len(wstart) in [2]  # later make it plottable in 3 dimensions

        self.ax.clear()
        plt.xlabel(wrt[0]), plt.ylabel(wrt[1])
        # reverse the values because we're minimizing the function
        self.ax.streamplot(wstart[0], wstart[1], -wend[0], -wend[1])
        plt.pause(0.0001)
