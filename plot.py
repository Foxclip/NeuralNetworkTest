import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


plot_queue = None


def start(queue):
    process = multiprocessing.Process(target=anim, args=(1, queue))
    process.start()


fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [])


def init():
    ax.set_ylim(0, 1)


def update(frame):
    while not plot_queue.empty():
        xdata.append(len(xdata) + 1)
        ydata.append(plot_queue.get())
        ln.set_data(xdata, ydata)
        ax.set_xlim(0, max(len(xdata), 1))


def anim(name, queue):
    global plot_queue
    plot_queue = queue
    ani = FuncAnimation(fig, update, init_func=init, interval=10)
    plt.show()
