import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import collections as mc


MAX_POINTS = 200     # limiting amount of displayed points for performance purposes
DECREASE_FACTOR = 2  # if MAX_POINTS is reached, point list is shrunk by this factor

max_index = 1         # index of next point
add_every = 1         # add point every x frames

plot_queue = None     # multiprocessing.queue from which new points come


def start(queue):
    process = multiprocessing.Process(target=anim, args=(1, queue))
    process.start()


fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [])
colors = []


def init():
    ax.set_ylim(0, 1)


def get_points():
    lines = []
    for i in range(1, len(xdata)):
        line = [(xdata[i], ydata[i]), (xdata[i - 1], ydata[i - 1])]
        lines.append(line)
    return lines


def update(frame):

    while not plot_queue.empty():

        global xdata
        global ydata
        global max_index
        global add_every

        if max_index % add_every == 0:

            if len(xdata) >= MAX_POINTS:
                xdata = xdata[0::DECREASE_FACTOR]
                ydata = ydata[0::DECREASE_FACTOR]
                add_every *= DECREASE_FACTOR

            queue_data = plot_queue.get()
            xdata.append(max_index)
            ydata.append(queue_data[0])
            colors.append(queue_data[1])
            if frame > 1:
                lc = mc.LineCollection(get_points(), colors=colors)
                ax.clear()
                ax.add_collection(lc)
            # ln.set_data(xdata, ydata)

        else:
            plot_queue.get()

        ax.set_xlim(0, max_index)
        ax.set_ylim(0, max(ydata))
        max_index += 1


def anim(name, queue):
    global plot_queue
    plot_queue = queue
    ani = FuncAnimation(fig, update, init_func=init, interval=10)
    plt.show()
