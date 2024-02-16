import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

N = 500  # number of steps
x1 = np.zeros((N, 3))
x2 = np.zeros((N, 3))
x3 = np.zeros((N, 3))
t = np.linspace(0, 20, N)
x1[:,0] = np.sin(t)
x1[:,1] = np.cos(t)
x1[:,2] = t
x2[:,0] = np.sin(t+1)
x2[:,1] = np.cos(t+1)
x2[:,2] = t
x3[:,0] = np.sin(t+2)
x3[:,1] = np.cos(t+2)
x3[:,2] = t

def generate_lines_from_x123(length, x1, dims=3):
    lineData = np.empty((dims, length))
    lineData[:,0] = x1[0]
    for index in range(1, length) :
        lineData[:, index] = x1[index]
    return lineData

def updatelines(num, dataLines, lines) :
     for line, data in zip(lines, dataLines) :

         line.set_data(data[0:2, :num])
         line.set_3d_properties(data[2,:num])
     return lines

fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection = '3d')
ax.set_title('3 Body Problem')

do_animation = True
if do_animation:
    data = [generate_lines_from_x123(N, x, 3) for x in (x1, x2, x3)]
    lines = [ax.plot(x[:,0], x[:,1], x[:,2])[0] for x in (x1, x2, x3)]
    line_ani = animation.FuncAnimation(fig, updatelines, N, fargs=(data, lines),
                          interval=100, blit=False)
else:
    # just plot the 3 curves

    ax.plot(x1[:,0],x1[:,1],x1[:,2],color = 'b')
    ax.plot(x2[:,0],x2[:,1],x2[:,2],color = 'm')
    ax.plot(x3[:,0],x3[:,1],x3[:,2],color = 'g')

    ax.scatter(x1[-1,0],x1[-1,1],x1[-1,2],color = 'b', marker = 'o', s=30, label = 'Mass 1')
    ax.scatter(x2[-1,0],x2[-1,1],x2[-1,2],color = 'm', marker = 'o',s=90, label = 'Mass 2')
    ax.scatter(x3[-1,0],x3[-1,1],x3[-1,2],color = 'g', marker = 'o',s=60, label = 'Mass 3')
    ax.legend()

plt.show()